# Optimization-driven Uncertainty Forecasting: 논문 요약



**저자:** Sajad Karimi (Binghamton University, SUNY), Soongoo Kwon (Yonsei University)

**저널:** Applied Energy 326 (2022) 119929



---



## 문제 정의



재생에너지(태양광) 발전사의 day-ahead 전력 시장 참여 시, 발전량 불확실성으로 인해 최적 commitment 결정이 어렵습니다. 기존 접근법은 예측과 최적화를 분리하여 처리함으로써 최적화 결과를 예측 모델에 반영하지 못하는 한계가 있습니다.



## 기존 접근의 한계 (Forecasting-First Optimization-Second)



1. Step 1: AR/MLR 모델로 재생에너지 발전량 예측 (예측 오차 최소화만)

2. Step 2: 예측 값을 최적화 모델에 대입하여 commitment 결정

3. **문제:** 예측과 최적화가 분리되어 있어, 최적화 모델의 정보가 예측 단계에 반영되지 않음



## 제안 방법 (Optimization-Driven Forecasting)



예측과 최적화를 통합한 단일 최적화 문제로 공식화:



**목적함수:**

```

min Σ { W₁ · [optimality gap] + W₂ · [forecasting error] }

```



- W₁, W₂: 정규화 파라미터 (optimality gap vs forecasting error 트레이드오프 조정)

- W₁/W₂ 비율 ↑ → optimality gap 최소화 우선

- W₁/W₂ 비율 ↓ → forecasting error 최소화 우선



**제약조건:**

- AR/MLR 모델의 예측 식을 제약조건으로 포함

- Day-ahead commitment 최적화 모델의 제약조건 포함



## 검증 결과



### Endogenous 예측 (AR 기반)

| 모델 | nRMSE | Optimality Gap |

|------|-------|----------------|

| AR (기존) | 34.76% | 15.04% |

| 제안 (W₁=1, W₂=1) | 44.85% | 11.44% |



→ 예측 오차는 10%p 증가했지만, optimality gap은 24% 감소



### Exogenous 예측 (MLR 기반)

| 모델 | nRMSE | Optimality Gap |

|------|-------|----------------|

| MLR (기존) | 21.76% | 12.59% |

| 제안 (W₁=1, W₂=1) | 22.01% | 10.28% |



→ 예측 오차는 1%p 증가했지만, optimality gap은 18% 감소



## 핵심 통찰



1. **Penalty cost rate에 따른 적응:** 기존 AR/MLR은 penalty rate과 무관하게 같은 예측 값을 반환하지만, 제안 방법은 penalty rate에 따라 commitment 전략을 동적으로 조정

   - Penalty rate 0% → commitment 증가 (shortage 비용 없음)

   - Penalty rate 30-50% → 예측 발전량 수준으로 commitment

   - Penalty rate >50% → commitment 감소 (shortage 위험 회피)



2. **Trayd-off:** 예측 오차(nRMSE)는 약간 증가하지만, optimality gap은 크게 감소하여 전체적인 이익 최대화



3. **실용성:** 실제 시장에서는 예측 정확도보다 이익 최대화가 더 중요함. 제안 방법은 시장 조건(가격, penalty)에 따라 예측 모델을 조정할 수 있음



## 기여점



1. 예측과 최적화를 통합한 새로운 프레임워크 제시

2. 정규화 파라미터(W₁, W₂)를 통해 트레이드오프 조정 가능한 모델

3. penalty cost rate 변화에 대응하는 적응형 commitment 전략

4. 두 가지 예측 접근법(endogenous AR, exogenous MLR) 모두에서 검증



---



*원문: references/0.jpg ~ 10.jpg*