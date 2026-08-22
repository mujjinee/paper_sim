
# 논의: 가격 예측값을 인자로 사용할 경우



## 배경



논문(Karimi & Kwon, 2022)에서 새로 제안한 방식은 **최적화 기반 예측(Optimization-Driven Forecasting)**입니다. 핵심은 예측 정확도(nRMSE)가 아니라 실제 이익(profit)을 극대화하는 것입니다.



## 현재 논문 모델이 사용하는 것



```

이익 = (DA 실제가격 × commitment) + (RT 실제가격 × 여유분) - (패널티 × 부족분)

```



## 문제점



1일 전에 commitment할 때 **실제 내일 가격을 알 수 없습니다**. 논문에서는 실제 가격으로 계산하므로, **미래를 아는 것처럼** 시뮬레이션 한 것입니다.



## 제안: 가격 예측값을 인자로 넣기



```

이익 = (DA 가격 × commitment) + (RT 예측가격 × 여유분) - (패널티 × 부족분)

```



- DA 가격은 1일 전에 이미 나오므로 실제값 사용 가능

- RT 가격만 예측값으로 대체 (실제 상황과 같아짐)



## 장점



1. **더 현실적** — 실제로는 미래 가격을 알 수 없음

2. **더 견고한 모델** — 가격 예측 오차까지 고려한 commitment

3. **실제 운영에 더 가까움** — 예측된 가격으로 판단하므로 실제와 같음



## 단점



1. **예측 오차 누적** — 발전량 예측 오차 + 가격 예측 오차

2. **모델 복잡도 증가** — 가격 예측 모델 추가 필요

3. **계산 시간 증가** — RT 가격 예측 모델을 학습/예측해야 함



## 구현 방향



```python

class PriceForecaster:

    def predict_rt_price(self, day_ahead_price, weather_features):

        # 간단한 접근: DA 가격과 기상 조건으로 RT 가격 예측

        pass



# 기존 최적화 모델에 통합

def optimization_with_price_prediction():

    rt_pred = price_forecaster.predict_rt_price(da_actual, weather)

    profit = da_actual * commitment + rt_pred * surplus - penalty * shortage

```



**첫 단계**: DA 가격은 실제값으로, RT 가격만 예측값으로 바꾸는 것이 현실적입니다.



## RT 가격 예측 모델 옵션



| 모델 | 복잡도 | 정확도 | 참고 |

|------|--------|--------|------|

| 단순 평균 | 낮음 | 낮음 | 과거 RT 가격 평균 |

| DA 기반 | 낮음 | 중간 | DA 가격 + 과거 편차 |

| 시계열 | 중간 | 중간 | ARIMA, Prophet |

| ML 기반 | 높음 | 높음 | XGBoost, LSTM |



## 고려할 것



- RT 가격 예측 오차가 profit에 미치는 영향 크기를 먼저 살펴봐야 합니다.

- 논문이 밝히지 않은 부분이지만, 실제 시장에서 RT 가격 예측은 이미 존재하는 문제입니다.

- 발전량 예측보다 RT 가격 예측이 더 쉬울 수 있습니다 (과거 패턴 반복).



## 다음 단계



1. DA 가격만으로 RT 가격 예측 (가장 간단한 기준선)

2. 기상 조건까지 추가한 RT 가격 예측

3. 기존 최적화 모델에 RT 예측가격 통합

4. nRMSE와 optimality gap 변화를 기존 논문 결과와 비교

