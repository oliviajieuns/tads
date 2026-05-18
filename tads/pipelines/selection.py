"""Per-epoch sample selection dispatch.

Wraps the four selection methods behind a single function. For data_agent
and tads the heavy collect_episode runs on rank 0 only; other ranks share
the resulting indices through a filesystem sentinel + poll, NOT through
an NCCL barrier — that was the deadlock that crashed runs after epoch 1.

Why polling, not dist.barrier:
    Rank 0 spends 30+ minutes inside collect_episode (52K samples × 32
    decoder layers × chunked rewards). While that runs, the other DDP
    ranks would be stuck inside dist.barrier() inside _broadcast_selection,
    and any of them hitting the NCCL collective watchdog (120 min default
    now, less previously) tears down the communicator. The next forward
    pass then fails on every rank, and rank 0 — which never reached the
    barrier — exits before saving any checkpoint.

    The fix is to remove the NCCL barriers from this path entirely. Rank 0
    writes the selection atomically (tmp + fsync + rename), then writes
    a separate `.ready` sentinel; workers poll on disk for the sentinel
    and read once it appears. The only collective in this module is a
    single barrier at the very end, after everyone has the data — so it
    always completes immediately.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ..core.agent import PPOAgent
from ..core.selector import collect_episode
from ..core.trajectory_anchor import TrajectoryAnchor
from ..core.utils import is_main_process, local_rank, rank, world_size

logger = logging.getLogger(__name__)


# Workers poll this often while waiting for rank-0's collect_episode.
_POLL_INTERVAL_SEC = 2.0
# Hard ceiling on how long workers will wait. Set generously — episodes
# can legitimately take an hour at the 7B scale.
_POLL_TIMEOUT_SEC = 6 * 60 * 60  # 6 hours


def _random_indices(n_total, ratio, seed, epoch):
    g = torch.Generator()
    g.manual_seed(seed + epoch * 100)
    perm = torch.randperm(n_total, generator=g).tolist()
    k = max(1, int(n_total * ratio))
    return perm[:k]


def _broadcast_selection(selected, *, epoch=0, output_dir=None):
    """NCCL broadcast of selected indices from rank 0 to all ranks.

    Earlier this function used a filesystem sentinel + poll loop on the
    workers. That avoided NCCL during rank 0's long collect_episode, but
    on the SPACE group-volume it surfaced a sporadic race: rank 0 would
    write the ``.ready`` sentinel atomically, one worker would observe
    it, and the remaining workers would not -- they spun in the poll
    forever while the rest of training deadlocked. (Concretely:
    1108319's tads_10 retry on 2026-05-18: rank 2 saw the sentinel and
    advanced to SFT step 0, ranks 1 and 3 polled for 57+ minutes.)

    The new path is the obvious one: a single
    ``dist.broadcast_object_list``. Ranks 1..N enter the collective
    at the same time rank 0 begins collect_episode and block on it
    cleanly. The collective only completes once rank 0 finishes
    collect_episode and reaches the broadcast -- which is exactly the
    synchronisation we want. The NCCL collective watchdog is bumped
    to 6 h in ``tads/train.py::ensure_ddp_initialized`` so this is
    comfortable for any plausible rank-0 workload.

    Rank 0 still drops a JSON copy of the selection alongside the
    checkpoint for the resume-time cache reuse path; that copy is
    purely informational and never read by other ranks.
    """
    if not dist.is_initialized():
        if hasattr(selected, "tolist"):
            return selected.tolist()
        return list(selected) if not isinstance(selected, list) else selected

    r = dist.get_rank()
    SRC = 0

    if r == SRC:
        if hasattr(selected, "tolist"):
            selected = selected.tolist()
        elif not isinstance(selected, list):
            selected = list(selected)
        selected = [int(x) for x in selected]
        logger.info(
            "[sel-share] rank=0 normalized selection | len=%d | first5=%s",
            len(selected), selected[:5],
        )
        obj_list = [selected]
    else:
        obj_list = [None]

    # NCCL collective. Holds ranks 1..N-1 for the entire duration of
    # rank 0's collect_episode + scoring. The init_process_group timeout
    # is 6 h (see train.py), well beyond any per-epoch rank-0 work.
    dist.broadcast_object_list(obj_list, src=SRC)
    result = obj_list[0]

    if r != SRC:
        if not isinstance(result, list):
            raise RuntimeError(
                f"[rank {r}] broadcast returned wrong shape: "
                f"type={type(result).__name__}",
            )
        logger.info(
            "[sel-share] rank=%d received selection broadcast | len=%d",
            r, len(result),
        )

    # Informational on-disk copy for cache-reuse on rank 0 only.
    # Other ranks never read this; it is only consulted by the cache-
    # reuse branch in select_indices() on a future resume.
    if r == SRC and output_dir is not None:
        base = Path(output_dir)
        try:
            base.mkdir(parents=True, exist_ok=True)
            sel_path = base / f"_selection_epoch{epoch}.json"
            tmp_path = sel_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(result, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, sel_path)
            logger.info(
                "[sel-share] rank=0 persisted selection to %s for resume cache",
                sel_path,
            )
        except Exception as e:
            # Persistence failure must not break training; just log.
            logger.warning(
                "[sel-share] could not persist selection to %s: %s",
                output_dir, e,
            )

    return result


def select_indices(method, *, model, agent, anchor, dataset, cfg, epoch, seed, device):
    """Return (selected_indices, extras) for the given epoch."""
    n_total = len(dataset)
    ratio = float(cfg["selection_ratio"])
    extras = {}

    if method == "full":
        selected = list(range(n_total))
        logger.info("Full dataset selection | k=%d", len(selected))
        return selected, extras

    if method == "random":
        selected = _random_indices(n_total, ratio, seed, epoch)
        logger.info("Random selection | k=%d/%d", len(selected), n_total)
        return selected, extras

    if method not in ("tads", "data_agent"):
        raise ValueError("Unknown method: " + repr(method))

    # ---------- selection cache: skip collect_episode if a prior run
    # ---------- already produced selected_indices_epoch{N}.json
    # collect_episode for tads/data_agent takes 30+ min on 7B. If a previous
    # run made it through scoring but hung in the post-broadcast NCCL step,
    # the indices already exist on disk and we can reuse them directly. This
    # path is ONLY hit when the file is present; a fresh start still runs
    # the full episode.
    _output_dir_raw = cfg.get("output_dir") or cfg.get("output_root")
    if _output_dir_raw is not None:
        _cached_path = Path(_output_dir_raw) / f"selected_indices_epoch{epoch}.json"
        if _cached_path.exists():
            try:
                with open(_cached_path) as _f:
                    _cached = json.load(_f)
                if isinstance(_cached, list) and len(_cached) > 0:
                    logger.info(
                        "REUSING cached selection from %s (%d indices) — "
                        "skipping collect_episode for epoch %d.",
                        _cached_path, len(_cached), epoch,
                    )
                    # Broadcast the cached indices to all ranks via the same
                    # file-polling mechanism so workers also get them.
                    if is_main_process():
                        selected = [int(x) for x in _cached]
                    else:
                        selected = []
                    selected = _broadcast_selection(
                        selected, epoch=epoch,
                        output_dir=_output_dir_raw,
                    )
                    extras["selection_cache_reused"] = True
                    return selected, extras
            except Exception as _e:
                logger.warning(
                    "Could not reuse cached selection at %s (%s); "
                    "running full collect_episode.",
                    _cached_path, _e,
                )

    if is_main_process():
        print("[trace] rank=0 ENTER main branch | method=" + method
              + " | anchor=" + ("set" if anchor is not None else "None"), flush=True)
        import traceback as _tb
        try:
            if method == "tads" and anchor is not None:
                logger.info("Updating trajectory anchor ...")
                print("[trace] rank=0 BEFORE anchor.update", flush=True)
                anchor_stats = anchor.update(
                    model=model, dataset=dataset, seed=seed, epoch=epoch,
                )
                _akeys = list(anchor_stats.keys()) if anchor_stats else None
                print("[trace] rank=0 AFTER anchor.update | stats_keys="
                      + str(_akeys), flush=True)
                extras["anchor_stats"] = anchor_stats

            tads_cfg = cfg.get("tads", {}) or {}
            exp_tag = str(cfg.get("model_key", "?")) + "/alpaca/" + method

            print("[trace] rank=0 BEFORE collect_episode", flush=True)
            episode = collect_episode(
                model=model,
                agent=agent,
                dataset=dataset,
                selection_ratio=ratio,
                trajectory_anchor=anchor if method == "tads" else None,
                lam=float(tads_cfg.get("lam", 0.0)),
                use_anchor=bool(tads_cfg.get("use_anchor", False)) and method == "tads",
                batch_size=int(cfg.get("episode_batch_size", 1)),
                device=str(device),
                seed=seed,
                epoch=epoch,
                exp_tag=exp_tag,
            )
            print("[trace] rank=0 AFTER collect_episode | episode_keys="
                  + str(list(episode.keys())), flush=True)
            selected = episode["selected_indices"]
            _slen = len(selected) if hasattr(selected, "__len__") else "?"
            print("[trace] rank=0 selected=" + type(selected).__name__
                  + " len=" + str(_slen), flush=True)

            extras.update({
                "r_loss_mean": episode["r_loss_mean"],
                "r_entropy_mean": episode["r_entropy_mean"],
                "r_weight": episode["r_weight"],
                "rdiff_mean": episode["rdiff_mean"],
                "rconf_mean": episode["rconf_mean"],
                "lam": episode["lam"],
                "use_anchor": episode["use_anchor"],
                "align_mean": episode["align_mean"],
                "align_std": episode["align_std"],
            })

            if agent is not None:
                actor_loss, critic_loss = agent.update(
                    states=episode["states"],
                    actions=episode["actions"],
                    old_log_probs=episode["log_probs"],
                    rewards=episode["rewards"],
                )
                extras.update({"actor_loss": actor_loss, "critic_loss": critic_loss})
                logger.info(
                    "PPO update | actor_loss=%.4f | critic_loss=%.4f",
                    actor_loss, critic_loss,
                )
        except Exception as _e:
            print("[trace] rank=0 EXCEPTION in main branch: "
                  + type(_e).__name__ + ": " + str(_e), flush=True)
            _tb.print_exc()
            import sys as _sys
            _sys.stdout.flush()
            _sys.stderr.flush()
            raise
    else:
        selected = []

    _output_dir = (
        cfg.get("output_dir")
        or cfg.get("output_root")
        or "/tmp/tads_selection_share"
    )
    selected = _broadcast_selection(
        selected, epoch=epoch, output_dir=_output_dir,
    )
    return selected, extras


def save_selection(output_dir, epoch, selected):
    """Persist the per-epoch selection for resume-time cache reuse.

    Atomic tmp + fsync + rename so a crash mid-write does not leave a
    truncated JSON behind — the cache-reuse path in ``select_indices``
    would otherwise hit json.JSONDecodeError on the next run and fall
    back to the 30-min collect_episode unnecessarily.
    """
    if not is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"selected_indices_epoch{epoch}.json"
    tmp = final.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(selected, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
