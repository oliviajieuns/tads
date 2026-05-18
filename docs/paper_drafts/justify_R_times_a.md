# Why TADS multiplies R · a — theoretical justification

작성자: jieuns | 2026-05-18 | for 동료 리뷰용

## TL;DR

TADS score $s_i^{(t)} = R_i^{(t)} \cdot a_i^{(t)} \cdot (1 + \lambda\,\widetilde{\mathrm{align}}_i^{(t)})$
에서 $R \cdot a$ 곱셈은 Data Agent 원논문엔 없는 **TADS의 확장**.
근거 네 줄:

1. **Expected utility decomposition** — bandit/causal inference 문헌의 표준
   (doubly-robust estimator: propensity × reward).
2. **Adaptive curriculum** — early R-dominated explore, late a-sharpened exploit.
3. **Variance reduction** — bounded $a \in (0,1)$ 가 noisy $R$ 위의 learned soft mask.
4. **Information channels 분리** — text-level $R$ × representation-level $a$.

---

## 1. PPO actor 정확한 정의 (`tads/core/agent.py`)

```
state    = h_i  (마지막 토큰 hidden under θ_t)
trunk    = MLP(state_dim → 256 → 256)
heads    = α_head, β_head, value_head
dist     = Beta(α_t(h_i), β_t(h_i))
action   = a_i^(t) ~ Beta(·,·)  ∈ (0, 1)
training = clipped surrogate PPO on R_i^(t)
           (group-relative advantage by default)
```

해석: actor는 **state $h_i$ → "이 샘플이 학습에 도움될 정도"** $\in (0,1)$ 을
출력하는 학습된 routing function. PPO는 $\mathbb{E}[R]$ maximizing 방향으로
$\theta_\pi$ 업뎃.

즉 $a_i^{(t)}$ 는 **"actor가 학습한 sample-wise posterior of usefulness"** 로
읽을 수 있음.

---

## 2. 근거 (1): Expected utility / doubly-robust

Decision theory:
$$
U(\text{select } i) = P(i \text{ worth selecting} \mid h_i) \cdot \text{value}(i)
$$

TADS에서:
- $a_i^{(t)} \in (0,1)$ ≈ actor's learned posterior of "select i is useful under $\theta_t$"
- $R_i^{(t)}$ = realized value of selecting $i$ at current model state

→ $R \cdot a$ 는 **per-sample expected utility의 plug-in estimator**.

이건 bandit / off-policy evaluation literature의 표준 형태:

- Robins & Rotnitzky (1995): doubly-robust estimator $\hat{V} = \hat{\pi}(a|s) \cdot \hat{r}(s,a)$
- Dudík, Langford, Li (2011) "Doubly Robust Policy Evaluation"
- Recommendation/IPS literature: propensity score × outcome

**리뷰어 비판 "곱하는 페이퍼 없다" 는 좁은 시각**.
RL/instruction-tuning 페이퍼에서 흔치 않은 건 맞지만,
**bandit/causal inference 에선 표준 형태**.

---

## 3. 근거 (2): Adaptive curriculum (학습 단계별 selection 양상)

| Epoch t | actor a 상태 | R 상태 | s = R·a 의 양상 |
|---|---|---|---|
| Early | 거의 uniform (미학습) | 모델이 못 푸는 샘플에 high | R 지배 → **hard sample explore** |
| Mid | partial peakedness | 점점 잘 푸는 영역 생김 | R·a 의 일부 합의 |
| Late | sharpened | 새 frontier 좁아짐 | **합의 영역만 통과 = curriculum 좁힘** |

- 곱셈은 **phase transition** 만듦 (early R-driven → late a-gated)
- additive ($R + \lambda a$) 였으면 두 신호가 independent로 작용해 이런 동작 안 나옴
- 실험적으로 epoch 후반에 selected subset diversity가 줄어드는 게 이 mechanism의 증거

---

## 4. 근거 (3): Variance reduction / learned soft mask

| 항 | range | smoothness |
|---|---|---|
| R_i | unbounded, σ_R 큼 (CE loss 합성) | noisy across samples |
| a_i ∈ (0,1) | bounded | smooth (Beta dist, PPO clip 으로 epoch 간 점진적) |

- $a$ 가 bounded learned mask → noisy $R$ 위에서 **soft attention** 처럼 작동
- additive ($R + a$): scale mismatch 때문에 $\lambda$ tuning 지옥, bounded 보장 깨짐
- multiplicative: $R \cdot a$ 의 분포 자동 안정화 (a 가 0 근처면 outlier R 도 작아짐)

---

## 5. 근거 (4): Information channels 분리 + AND-gate

| 신호 | 무엇을 본다 | 채널 |
|---|---|---|
| R_i (loss + entropy) | **무엇을 출력하느냐** | output token level |
| a_i (state = h_i) | **모델이 어떻게 표현하느냐** | hidden representation level |
| align_i (anchor) | **표현이 어디로 움직이느냐** | trajectory geometry level |

세 채널이 본질적으로 다른 정보 → 곱셈 = **AND-gate consensus**:
세 신호 모두 nontrivial 일 때만 top-$k$ 진입.

→ 단일 채널 noise/bias 에 robust.

---

## 6. additive 대비 multiplicative 의 우위 (정리)

| 속성 | additive R + λa | multiplicative R·a |
|---|---|---|
| Expected utility 해석 | 없음 | 있음 (doubly-robust) |
| Scale invariance | 깨짐 (R, a 스케일 맞춰야) | 자동 |
| Phase transition (curriculum) | 없음 | 있음 |
| AND-gate consensus | 없음 (둘 중 하나 큰 거 통과) | 있음 |
| λ=0 reduction | R 만 남음 (a 사라짐) | R·a 남음 (RL+CR 그대로) |
| Anchor 부착 | additive 면 모순 (또 더하기?) | natural (또 곱하기) |

**마지막 행이 결정적**: anchor 를 동일 multiplicative 슬롯에 끼우는 형태가
가능하려면 base score 자체가 multiplicative 여야 함. TADS의 anchor 항
$(1 + \lambda\,\widetilde{\mathrm{align}})$ 가 깔끔하게 들어가는 이유.

---

## 7. 솔직히 인정해야 할 부분

- Data Agent 원논문 selection rule = top-$k(a)$ 뿐. R 은 PPO loss 안에서만.
- TADS의 $R \cdot a$ 는 **Data Agent 의 단순 차용이 아니라 의도적 확장**.
- 곱셈을 RL/instruction-tuning literature 안에서만 찾으면 직접 선례 없음.
- 정당화의 무게는 **bandit/causal literature + empirical ablation** 에 둠.

---

## 8. Empirical defense — 추가 ablation 필요

다음 셀들 0.5B sanity 에서 추가 (각 ~30분):

| cell | score | 목적 |
|---|---|---|
| s = R only | $R_i$ | actor 무용성 체크 |
| s = a only | $a_i$ | Data Agent 정확 재현 |
| s = R + a | $R_i + \lambda a_i$ | additive 대안 |
| s = R · a | $R_i \cdot a_i$ | **TADS base** (현재) |
| s = R · a · boost | full | TADS |

기대: R·a > R-only, R·a > a-only, R·a ≥ R+a.
→ 4가지 근거의 empirical leg 완성.

---

## 9. 페이퍼에 박을 framing (intro/method)

> The score is a per-sample **expected-utility estimator**:
> $a_i^{(t)} \in (0,1)$ is the PPO actor's learned posterior that
> sample $i$ contributes to training under representation $h_i^{(t)}$,
> and $R_i^{(t)}$ is the composite reward valuing that contribution
> under $\theta_t$. Their product is the standard form of doubly-robust
> off-policy value estimation (Dudík et al., 2011), instantiated at the
> sample level. The multiplicative form also yields an adaptive
> curriculum — early epochs are R-dominated (explore the model's
> current frontier of difficult samples) and later epochs are a-gated
> (exploit the actor's learned posterior) — and gives the anchor term
> $(1 + \lambda\,\widetilde{\mathrm{align}})$ a natural multiplicative
> slot complementary to the text-level $R$ and the action-level $a$.

---

## 10. 인용해야 할 reference

- Robins, J. M., & Rotnitzky, A. (1995). Semiparametric efficiency in multivariate regression models with missing data.
- Dudík, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and learning. ICML.
- Schulman et al. (2017). Proximal Policy Optimization. (PPO clipped surrogate)
- Data Agent paper (RL+CR base selector — TADS 의 출발점, top-k(a) rule)

---

## 11. 동료 검토 요청 사항

1. doubly-robust 비유가 너무 stretch 인지? (a 가 propensity 가 아니라 action 자체라는 점에서)
2. AND-gate consensus framing 이 reviewer 한테 통할지?
3. Ablation 4 셀 (R-only, a-only, R+a, R·a) 우선순위 동의?
4. $\lambda=0$ cell 을 "Data Agent baseline" 으로 부를지, "RL+CR" 로 부를지 통일?
