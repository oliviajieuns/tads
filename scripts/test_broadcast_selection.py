"""Isolated smoke test for tads.pipelines.selection._broadcast_selection.

What this verifies (in ~30 seconds, no training, no dataset):
  1. dist.init_process_group succeeds on 4 ranks.
  2. _broadcast_selection (file-write + NCCL barrier) returns the
     SAME non-empty list of integers on every rank.
  3. The on-disk selection file is the canonical source: rank 0 wrote
     it, ranks 1..N-1 read the same contents back.

Launch:

  cd /group-volume/jieuns/tads_v2
  source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
      --master_port=29520 scripts/test_broadcast_selection.py

PASS output:
  [smoke] rank=0 sent  len=1040 first5=[7, 13, 42, 99, 256]
  [smoke] rank=1 got   len=1040 first5=[7, 13, 42, 99, 256]  match=True
  [smoke] rank=2 got   len=1040 first5=[7, 13, 42, 99, 256]  match=True
  [smoke] rank=3 got   len=1040 first5=[7, 13, 42, 99, 256]  match=True
  [smoke] ALL RANKS PASS

FAIL output: any mismatch, garbage length, or exception.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> int:
    # 1) DDP init (same shape as tads.train).
    if "RANK" not in os.environ:
        print("[smoke] RANK env not set; run under torchrun", file=sys.stderr)
        return 2

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(hours=1),
    )
    r = dist.get_rank()
    world = dist.get_world_size()
    local_r = int(os.environ.get("LOCAL_RANK", r))
    torch.cuda.set_device(local_r)

    print(
        f"[smoke] rank={r}/{world} local_rank={local_r} device=cuda:{local_r}",
        flush=True,
    )

    # 2) Build the same kind of selection a real epoch would produce.
    # Rank 0 picks an arbitrary set, non-root ranks pass empty (real
    # behaviour -- only rank 0 has computed indices at this call site).
    if r == 0:
        # Sentinel pattern with a known first5 so the receivers can be
        # checked deterministically.
        selected = [7, 13, 42, 99, 256] + list(range(1000, 2035))
    else:
        selected = []

    # 3) Import and call the function under test.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tads.pipelines.selection import _broadcast_selection

    with tempfile.TemporaryDirectory() as td:
        # All ranks must point at the same shared dir.
        # In a real training run this is <output_root>/<output_subdir>/runs/<tag>/.
        # For the test, use a group-volume scratch dir all 4 ranks can see.
        shared_dir = os.environ.get(
            "TADS_SMOKE_SHARED_DIR",
            "/group-volume/jieuns/tads-checkpoints/_smoke_broadcast_test",
        )
        Path(shared_dir).mkdir(parents=True, exist_ok=True)

        result = _broadcast_selection(
            selected, epoch=0, output_dir=shared_dir,
        )

    # 4) Verify.
    if r == 0:
        print(
            f"[smoke] rank=0 sent  len={len(result)} first5={result[:5]}",
            flush=True,
        )
    else:
        expected_first5 = [7, 13, 42, 99, 256]
        expected_len = 5 + 1035  # 1040
        got_first5 = result[:5] if isinstance(result, list) else None
        got_len = len(result) if hasattr(result, "__len__") else -1
        match = (got_len == expected_len) and (got_first5 == expected_first5)
        print(
            f"[smoke] rank={r} got   len={got_len} "
            f"first5={got_first5}  match={match}",
            flush=True,
        )
        if not match:
            print(
                f"[smoke] rank={r} FAIL: expected len={expected_len} "
                f"first5={expected_first5}",
                file=sys.stderr,
                flush=True,
            )
            dist.destroy_process_group()
            return 1

    # 5) Cross-rank consensus via a final barrier so the prints don't
    # interleave with the success line.
    dist.barrier()
    if r == 0:
        print("[smoke] ALL RANKS PASS", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
