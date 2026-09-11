# AR Gap이 rate와 함께 오르는 이유 (9% → 22%)



Fig.5 Gap 그래프에서 논문 AR(파랑 점선) 은 rate 0%에서 9%, rate 100%에서 22%로 **일관되게 상승**합니다.



---



## 1. Gap 정의



$$\text{Gap} = 100 \times \frac{\sum \text{oracle} - \sum \text{realized}}{\sum \text{oracle}}$$



 Gap이 커지려면:

- $\sum$realized가 **작아지거나**,

- $\sum$oracle이 **커지거나**



---



## 2. 핵심 — AR 예측은 rate와 무관



$$\hat{S} = f(\text{전날 발전량만})$$



AR 회귀는 전날 발전량으로만 학습; market price(DA/RT/PC)를 입력으로 받지 않습니다.

따라서 rate가 0%든 100%든 **동일한 $\hat{S}$** 를 출력합니다.



---



## 3. realized — rate가 커질수록 떨어짐



$$\begin{align*}

\text{realized} &= \text{DA} \cdot \hat{S} + \text{RT} \cdot \text{surplus} - (\text{RT} + \text{PC}) \cdot \text{shortage} \\

&= \text{DA} \cdot \hat{S} + \text{RT} \cdot \text{surplus} - (\text{RT} + \text{rate} \cdot \text{DA}) \cdot \text{shortage}

\end{align*}$$



rate를 $\rho$로 두고 두 경우로 나눕니다:



| 상황 | surplus / shortage | realized |

|---|---|---|

| $\hat{S} \leq S$ (under-commit) | surplus | $\text{DA} \cdot \hat{S} + \text{RT} \cdot (S - \hat{S})$ — **rate와 무관** |

| $\hat{S} > S$ (over-commit) | shortage | $\text{DA} \cdot \hat{S} - (\text{RT} + \rho \cdot \text{DA}) \cdot (\hat{S} - S)$ — **rate에 의존** |



over-commit 상황에서:



$$\frac{\partial \ \text{realized}}{\partial \rho} = -\text{DA} \cdot (\hat{S} - S) < 0$$



→ **$\rho$가 커질수록 realized는 선형적으로 감소**.



---



## 4. oracle — rate가 커져도 잘 견딤



$$\text{oracle} = \max \left\{ \ \underbrace{\text{RT} \cdot S}_{\text{(a) commit=0}}, \ \underbrace{\text{DA} \cdot S}_{\text{(b) commit=S}}, \ \underbrace{\text{DA} \cdot 1 - (\text{RT} + \rho \cdot \text{DA}) \cdot (1 - S)^+}_{\text{(c) commit=1}} \ \right\}$$



rate가 커지면 (c) 의 가치가 줄어듭니다. 하지만 oracle은 **최대값** 이므로:



$$\text{rate} \uparrow \implies \text{(c) 가치} \downarrow \implies \text{oracle} = \max\{\text{(a)}, \text{(b)}\}$$



(a) 나 (b) 로 **fallback** 합니다. 둘 다 rate와 무관하므로:



$$\text{rate가 충분히 커지면 oracle} \to \text{const}$$



---



## 5. 두 현상이 합쳐지면



$$\begin{align*}

\text{rate} \uparrow \implies \sum \text{oracle} &: \text{거의 일정 (최대 5~10% 감소)} \\

\text{rate} \uparrow \implies \sum \text{realized} &: \text{선형적으로 감소} \\[8pt]

\text{Gap} = \frac{\text{oracle} - \text{realized}}{\text{oracle}} &: \text{분자가 커지고 분모는 일정} \implies \textbf{상승}

\end{align*}$$



---



## 6. 직관적인 설명



> AR은 **시장 가격에 귀를 닫은 예측기**입니다.

>

> rate=0% → 패널티 없음 → over-commit해도 손해 없음 → Gap 9%

>

> rate=50% → 패널티 등장 → over-commit하면 벌금 → realized 하락 → Gap 15%

>

> rate=100% → 패널티가 DA와 동급 → over-commit이 큰 손해 → realized 크게 하락 → Gap 22%

>

> 하지만 AR은 "rate가 높아졌으니 commit을 줄이자"라고 생각할 수 없습니다. **동일한 예측**을 반복하므로, rate가 커질수록 그 예측이 **더 나빠집니다**.



---



## 7. 왜 제안모형은 수평일까?



제안모형은 rate를 학습에 반영합니다:



$$\rho \uparrow \implies \text{over-commit 패널티 큼} \implies \hat{S} \text{를 줄임} \implies \text{shortage} \downarrow$$



→ realized가 떨어지지 않으므로 Gap이 일정하게 유지됩니다.



---



## 8. 재현 Gap 구조 변경 분석 (2026-09-11)



### 8.1 변경 내용



논문 Gap 구조를 재현에 적용:



- **PC 정의**: `PC = rate × DA` → `PC = RT + rate × DA` (사용자 확인: 맞음)

- **oracle**: 3후보 `max(RT·S, DA·S, p1_full)` → 2후보 `max(DA·S, RT·S)`

- **realized**: `DA·x + RT·surplus - RT·shortage - rate·DA·shortage` → `DA·x + RT·surplus - PC·shortage`



### 8.2 Gap 평가만 변경 시 (v1, 분무 유지)



| 지표 | 이전 | v1(논문 Gap) | 논문 |

|---|---|---|---|

| AR base Gap(r=0.5) | 11.5% | **14.9%** | 15.04% |

| 제안 nRMSE(r=0.5) | 35.7% | 37.5% | 34.9% |

| 제안 Gap(r=0.5) | 12.0% | 13.7% | 11.4% |



AR base Gap이 논문과 **±0.2%p** 로 거의 일치. Gap 평가 구조 변경이 성공적.



**하지만 재현 제안모형의 Gap이 여전히 rate와 함께 상승**(6% → 18%)



### 8.3 분무(denom) 제거 실험 (v2)



원본 Gurobi 코드가 oracle.sum() 분무를 쓰지 않는다는 것을 발견 → 분무 제거 실험 진행.



**결과: nRMSE가 37% → 68%로 급증. 실패.**



#### 왜 nRMSE가 증가하나



분무가 있을 때 shortage 패널티 계수:

```

yc = W1 × scale × PC / Σoracle + W2/N = 2400/120000 + 0.011 = 0.031

```



분무가 없을 때:

```

yc = W1 × scale × PC + W2 = 2400 + 20 = 2420

```



**shortage 패널티가 10만 배 커짐**. 모델이 shortage(예측값 > 실제값)를 극도로 피하며 commit을 과도하게 낮게 예측 → 예측오차(nRMSE) 증가.



#### 왜 원본 Gurobi 코드는 분무 없이 작동하나



원본의 PC = rate×DA ≈ 20 (작음)

→ shortage 패널티 = 30 × 20 = 600 (작음)

→ W1 항이 지배적이라도 shortage 회피가 심하지 않음



v2의 PC = RT+rate×DA ≈ 80 (큼)

→ shortage 패널티 = 30 × 80 = 2400 (큼)

→ shortage 회피가 과도하게 심해짐



→ **PC 값이 커진 것이 분무 없는 구조의 취약점을 드러냄**



### 8.4 이론적 분석: 분무는 맞다



논문의 Eq.(9a):



$$\min \ W1 \cdot \frac{\sum(\text{oracle} - \pi(x))}{\sum(\text{oracle})} + W2 \cdot \frac{\sum|x - S|}{N \cdot \text{CAPACITY}}$$



분무가 명시되어 있습니다. 분무의 역할:



$$\begin{align*}

\text{분무 있음: } & \text{W1 항} \approx 1.0, \ \text{W2 항} \approx 20.0 \implies \text{W1/W2=1/20 의도 반영} \\

\text{분무 없음: } & \text{W1 항} \approx 108,000, \ \text{W2 항} \approx 1,800 \implies \text{W1이 60배 우세, W2 무시}

\end{align*}$$



**분무(denom = Σoracle)가 이론적으로 정확합니다.**



### 8.5 남은 문제: 재현 제안모형 Gap의 상승



Gap 평가 구조를 논문과 맞추고 학습 구조도 이론적으로 타당한 v1에서조차, **재현 제안모형의 Gap이 rate 0%→100%에서 6%→18%로 상승**합니다. 논문 제안모형은 ≈11%로 수평입니다.



nRMSE는 평탄(36~38%) — 모델이 rate에 따른 commit 조정을 충분히 하지 못하고 있습니다.



---



## 9. rate-의존 분무 상쇄 분석 (2026-09-11)



### 9.1 발견



rate-의존 분무 `denom = Σoracle × max(rate, 0.05)`가 **rate 효과를 상쇄**합니다.



목적함수 계수를 rate가 어떻게 바꾸는지 추적:



| 계수 | 분자 (rate 포함) | 분무(rate)로 나눈 후 |

|---|---|---|

| DA 비용(`obj_x`) | `W1·scale·DA` — rate와 무관 | **rate↑ → 0으로 수렴** |

| surplus 비용(`sc`) | `-W1·scale·RT + W2/N` — rate와 무관 | **rate↑ → 0으로 수렴** |

| shortage 비용(`yc`) | `W1·scale·(RT+rate×DA) + W2/N` | `≈ W1·(Σpc/Σoracle) + W2/(N·rate)` → **거의 상수** |



**결과: 목적함수 계수가 rate와 무관해져, 모델이 rate에 따라 다른 계수를 학습하지 않음.**



rate=0.1이든 rate=1.5이든 → 거의 동일한 β 학습 → 동일한 예측출력 → realized가 rate↑에 따라 떨어짐 → Gap 상승



### 9.2 수정: rate-의존 분무 제거



`denom = oracle_train.sum() * max(penalty_rate, 0.05)` → `denom = oracle_train.sum()`



#### 수정 후 결과 (W1=W2=1)



| Rate | AR nRMSE (기존) | AR nRMSE (수정) | AR Gap (기존) | AR Gap (수정) |

|---|---|---|---|---|

| 0.0 | 95.69% | 57.92% | 7.58% | 6.58% |

| 0.5 | 39.92% | 37.46% | 12.95% | 13.7% |

| 1.0 | 38.96% | 38.96% | 17.97% | 17.97% |

| 1.5 | 39.73% | 42.63% | 24.69% | 22.67% |



rate=0에서 nRMSE가 95% → 58%로 개선되었으나, **Gap은 여전히 상승**.



#### MLR은 방향이 맞음



| Rate | MLR nRMSE | MLR Gap |

|---|---|---|

| 0.1 | 23.47% | 8.15% |

| 0.5 | 23.49% | 10.44% |

| 1.0 | 24.04% | 12.80% |

| 1.5 | 24.44% | 14.99% |



nRMSE는 rate↑에 따라 **약하게 증가** (23% → 24%) — 논문과 같은 방향.

하지만 nRMSE 변화량이 너무 작아 Gap은 여전히 증가.



### 9.3 근본 원인: nRMSE 변화량 부족



논문의 제안모형: rate 0%→100%에서 nRMSE가 35% → 50% (+15%p)

우리 재현: rate 0%→100%에서 nRMSE가 37% → 43% (+6%p) — **불충분**



모델이 예측을 낮추는 양이 부족 → shortage가 충분히 줄지 않음 → realized가 oracle보다 빠르게 떨어짐 → Gap 증가



---



## 10. 원본 Gurobi 코드 구조 분석 (2026-09-11)



### 10.1 결정적 차이 발견



`CodefromJiWon/model_proposed_ar_profit_change_v3.py`를 분석한 결과, **목적함수 계수 구조**가 완전히 다름을 확인.



#### 원본 Gurobi 코드



```python

objective_expr = (

    (-W1 * da_this_hour) @ x_var       # raw W1, scale 없음, 정규화 없음

    + surplus_cost @ yplus_var

    + shortage_cost @ yminus_var

)



surplus_cost[i]  = (-W1 * rt_i) + W2              # magnitude ≈ 10

shortage_cost[i] = (W1 * (penalty_i + rt_i)) + W2  # magnitude ≈ 40

```



- `denom` 없음

- `scale=30`을 목적함수에 곱하지 않음 (오라클 계산에만 사용)

- `W2/n_obs`가 아닌 raw `W2` 사용



#### 우리 코드 (수정 전)



```python

obj[x_s + i] = -W1 * scale * da[i] / denom        # magnitude ≈ -0.001

sc[i] = (-W1 * scale * rt[i] / denom) + (W2 / n_obs)  # magnitude ≈ -0.01

yc[i] = (W1 * scale * pc / denom) + (W2 / n_obs)     # magnitude ≈ 0.03

```



- `denom = Σoracle`으로 모든 계수 나눔 → W1 항이 1000배 축소

- `scale=30`을 목적함수 계수에 곱함

- `W2/n_obs`로 나눔 → W2 항이 3600배 축소



**W1 항이 완전히 drowned out되어 MAE 최소화만 하는 baseline과 동일하게 동작.**



### 10.2 원본 계수 구조로 변경 실험



```python

obj[x_s + i] = -W1 * da[i]           # raw W1, scale 없음, denom 없음

sc[i] = (-W1 * rt[i]) + W2           # raw W2

yc[i] = (W1 * pc) + W2               # PC = RT + rate×DA, raw W2

```



#### 실험 결과



**AR (W1=W2=1):**

- nRMSE: **59~73%** — 너무 높음. W1(economic)이 W2(MAE)보다 20배 강해 정확도 무시

- Gap: 7% → 13% — 방향은 맞지만 nRMSE가 너무 높아서 의미 없음



**MLR (W1=W2=1):**

- nRMSE: 23% → 61% — rate↑에 따라 증가하는 방향은 맞지만 절대값이 높음

- Gap: 10% → 14% — 증가 but 완만



**AR baseline Gap(14.94%)이 논문(15.04%)과 ±0.1%p 일치** — Gap 평가 구조 완벽 재현 ✓



### 10.3 문제 진단: W1/W2 magnitude 불균형



```

W1 계수 magnitude ≈ price ≈ 20

W2 계수 magnitude = 1

→ W1이 W2보다 20배 강함 → economic만 추구 → nRMSE 59~73%

```



논문의 Eq.(9)에서는 정규화(denom)가 W1/W2 균형을 만듭니다. raw 계수로 동일하게 만들려면 W2에 가중치(Weight)를 적용해야 합니다.



---



## 11. W2 Weight 균형 보정 (2026-09-11)



### 11.1 아이디어



W1 계수 magnitude(≈20)와 W2 계수 magnitude(=Weight)를 맞추기 위해 W2에 Weight를 곱합니다.



```python

W2_eff = W2 * Weight



obj[x_s + i] = -W1 * da[i]

sc[i] = (-W1 * rt[i]) + W2_eff

yc[i] = (W1 * pc[i]) + W2_eff

```



### 11.2 Weight=15 실험 결과



#### AR (W1=W2=1, Fig.5)



| Rate | baseline nRMSE | baseline Gap | 제안 nRMSE | 제안 Gap | 논문 nRMSE |

|---|---|---|---|---|---|

| 0.0 | 36.11% | 6.42% | **39.02%** | 5.60% | ≈35% |

| 0.5 | 36.11% | 14.94% | **37.05%** | 13.74% | 44.45% |

| 1.0 | 36.11% | 23.46% | **40.5%** | 16.06% | ≈55% |

| 1.5 | 36.11% | 31.98% | **44.8%** | 15.59% | — |



**nRMSE가 59% → 37%로 대폭 감소 ✓**

Gap: 5.6% → 16% — acceptable ✓

하지만 논문 대비 nRMSE가 낮음 (37% vs 44%). Weight=15가 너무 강함.



#### MLR (W1=W2=1, Fig.8)



| Rate | baseline nRMSE | 제안 nRMSE | 제안 Gap | 논문 nRMSE |

|---|---|---|---|---|

| 0.5 | 24.43% | **23.07%** | 10.79% | 22.01% |

| 1.0 | 24.43% | **23.62%** | 13.1% | ≈25% |



MLR은 논문과 근접 ✓.



#### AR (W1/W2 sweep, rate=0.5)



| Label | nRMSE | Gap | 논문 nRMSE | 논문 Gap |

|---|---|---|---|---|

| AR/MLR | 36.11% | 14.94% | 34.76% | 15.04% |

| 1/20 | **35.6%** | **14.47%** | 34.89% | 13.91% |

| 1/10 | 35.27% | 14.02% | 35.14% | 13.42% |

| 1/5 | 36.5% | 14.64% | 36.28% | 12.71% |

| 1/1 | **37.05%** | **13.74%** | 44.95% | 11.44% |

| 20/1 | 61.99% | 10.75% | 50.07% | 11.36% |



**W1/W2=1/20에서 논문과 ±1%p로 매우 근접 ✓**

**W1/W2=1/1에서 논문 nRMSE(45%)보다 낮음(37%)** — Weight가 너무 강해 economic signal이 약해짐



#### MLR (W1/W2 sweep, rate=0.5)



| Label | nRMSE | Gap | 논문 nRMSE | 논문 Gap |

|---|---|---|---|---|

| AR/MLR | 24.43% | 11.34% | 21.76% | 12.59% |

| 1/20 | **24.07%** | **11.49%** | 21.92% | 11.91% |

| 1/1 | **23.07%** | **10.79%** | 22.01% | 10.28% |



MLR은 W1/W2=1/1에서도 논문과 근접 ✓.



---



## 12. 종합 분석



### 12.1 달성된 것



1. **Gap 평가 구조 재현**: baseline Gap이 논문과 ±0.1%p 일치 ✓

2. **W1/W2=1/20 조건 재현**: nRMSE가 논문과 ±1~3%p로 근접 ✓

3. **MLR 재현**: nRMSE와 Gap이 논문과 ±2%p 이내 ✓

4. **Gap behavior**: rate↑에 따라 증가(5.6% → 16%) — 육안으로 acceptable ✓



### 12.2 남은 문제



1. **W1/W2=1/1에서 AR nRMSE가 너무 낮음**: 37% vs 논문 45%

   - Weight=15가 너무 강해 economic signal이 약해짐

   - Weight를 낮추면(8附近) 논문과 더 근접할 것으로 예상



2. **rate↑에 따른 nRMSE 증가량이 논문보다 작음**: +8%p vs +15%p

   - 경제 최적화 조절 폭이 좁음



3. **Gap이 수평이 아님**: 5.6% → 16% (증가) vs 논문 8% → 11.5% (수평)

   - 모델이 rate에 따른 예측 조정이 충분하지 않음



### 12.3 다음 단계



1. **Weight=8로 재실행** — AR nRMSE를 논문과 더 근접하게 (42-45% 예상)

2. **Weight 스윕(5, 8, 10, 12, 15)** — 최적 Weight 탐색

3. **블록9(400일) 데이터로 재실행** — 학습 데이터 3배 증가



---



## 13. 핵심 인사이트 정리



### 문제 1: rate-의존 분무 상쇄

`denom = Σoracle × max(rate, 0.05)` → rate가 목적함수 계수에서 상쇄됨 → 모델이 rate에 반응하지 않음

**해결**: 분무에서 rate 의존성 제거 ✓



### 문제 2: W1/W2 magnitude 불균형

raw 계수에서 W1(≈20)이 W2(=1)보다 20배 강함 → economic만 추구 → nRMSE 59~73%

**해결**: W2에 Weight(=15) 곱하여 균형 보정 ✓



### 문제 3: Gap 구조 재현

PC = RT + rate×DA, oracle 2후보(max(DA·S, RT·S))로 변경 → baseline Gap 논문과 ±0.1%p 일치 ✓



---



## 14. Weight 스윕 완료 — 최적값 확정 (Claude, 세션 이어받음)



### 14.1 방법



§12.3의 "다음 단계"(Weight=8 재실행, Weight 스윕 5/8/10/12/15)를 이어서 실행하되, 범위를 더 낮은 쪽까지 확장(1~15)해서

`방법2_4term_ar_mlr_z03_block18.py`의 raw 계수 구조(§10.2/§11.1)로 AR Fig.5(rate 0/10/30/50/70/100%)를

직접 재현해 논문 판독값과 SSE(nRMSE+Gap 오차 제곱합) 비교.



### 14.2 결과



| Weight | 1 | 2 | 3 | **4** | 5 | 6 | 8 | 10 | 12 | 15 |

|---|---|---|---|---|---|---|---|---|---|---|

| SSE | 4467 | 941 | 380 | **369** | 431 | 568 | 983 | 1070 | 1134 | 1220 |



**Weight=4가 최적** (Weight=3도 거의 근접). Weight=15는 §11.2에서 이미 "너무 강함"으로 진단된 대로 SSE가

가장 나쁜 축에 속함 — §12.3에서 예상했던 방향(Weight를 낮추면 더 근접)이 맞았다.



Weight=4, rate=0/10/30/50/70/100%에서:

```

nRMSE: 46.60 | 43.42 | 45.65 | 44.64 | 48.97 | 55.03   (논문: 46 | 34 | 36 | 44.45 | 55 | 67)

Gap  :  6.48 |  7.90 | 10.65 | 11.79 | 11.94 | 12.19   (논문:  8 | 10 | 11 | 11.44 | 11.5 | 11.5)

```

rate=0%, 50%에서 nRMSE가 논문과 거의 정확히 일치(±0.6%p). Gap도 §11.2의 Weight=15 결과(5.6%→16%,

너무 가파름)보다 훨씬 평평해져서 논문의 "수평" 특성에 더 가까워짐.



### 14.3 KPI 기준점(W1=1,W2=20,rate=50%) 및 그리드 재검증 (Weight=4)



```

baseline nRMSE=36.11%(논문 34.76%), baseline Gap=14.94%(논문 15.04%, ±0.1%p)

W1/W2=1/20: 재현 nRMSE=36.32%(논문 34.89%)  재현 Gap=14.55%(논문 13.91%)

W1/W2=1/10: 재현 nRMSE=36.67%(논문 35.14%)  재현 Gap=14.58%(논문 13.42%)

W1/W2=1/1 : 재현 nRMSE=44.64%(논문 44.95%)  재현 Gap=11.79%(논문 11.44%)  ← 거의 정확히 일치

W1/W2=20/1: 재현 nRMSE=67.03%(논문 50.07%)  재현 Gap=10.69%(논문 11.36%)  ← 이 극단점만 크게 벗어남

```



1/20~1/1 구간(Fig.3/5/8이 실제로 쓰는 범위)에서는 전부 논문과 근접. 20/1처럼 W2가 극단적으로 작은

지점만 아직 어긋난다 — 이 구간은 이번 스코프(Fig.5/8 재현) 밖이라 더 파고들지 않음.



### 14.4 반영



`방법2_4term_ar_mlr_z03_block18.py`에 raw 계수 구조 + `W2_BALANCE_WEIGHT=4` + oracle 2후보 Gap을

전부 반영하고 재실행 완료. 방법2_fig3/5/6/8 PNG 갱신됨(`results/방법2_fig{3,5,6,8}_*.png`).



# END

