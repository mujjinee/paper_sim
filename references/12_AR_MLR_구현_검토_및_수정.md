# APEN 논문 AR·MLR 구현 검토 및 수정

검토 대상은 Karimi & Kwon (2022), *Optimization-driven uncertainty
forecasting: Application to day-ahead commitment with renewable energy
resources*와 `run_simulation.py`이다. 이 문서는 코드 수정 전 확인된 오류와
새 기준선 실행 파일의 수정 범위를 기록한다.

## 확정된 구현 오류

### 1. AR 테스트 이력 인덱스 오류

기존 `ARModel.predict_next_day()`는
`past_daily[self.n_lags - i - 1, h]`를 사용한다. 호출부는 학습 마지막 12일
뒤에 이미 관측된 테스트 일자를 계속 덧붙이지만, 위 인덱스는 언제나 처음 12개
행만 읽는다. 따라서 테스트 두 번째 날 이후에도 이전 테스트 관측값은 AR 입력에
전혀 사용되지 않고, 모든 테스트 일자의 예측이 같은 이력에서 나온다.

수정 구현은 `history_daily[-n_lags:]`로 가장 최근 12일을 얻는다. 예측 후 그날의
실측을 이력에 추가하되 회귀계수는 재학습하지 않는다. 즉, 다음 날 예측 시점에
이미 알 수 있는 과거 발전량만 사용한다.

### 2. MLR에서 Hour를 상수로 만든 구조

논문의 MLR 최종 변수 집합은 `SSRD`, `TSR`, `Hour`이다. 기존 구현은 시간대별
12개 회귀를 따로 학습하고, 각 모델에 `hour=h`를 넣는다. 한 모델 내부에서는
Hour가 모든 행에서 같은 상수이며 intercept와 완전 공선이다. 따라서 Hour의
계수를 식별하거나 시간대 효과를 학습할 수 없다.

수정 구현은 모든 날짜·시간대 행을 하나로 모은 pooled MLR을 학습한다. 그러므로
Hour가 0~11로 변하고, SSRD·TSR와 함께 논문의 회귀식의 설명변수가 된다.

### 3. 논문 MLR과 12변수 MLR의 비교 혼동

기존 출력은 12개 기상변수 MLR의 결과에 논문의 3변수 MLR 기준값
(nRMSE 21.76%, penalty 50%에서 optimality gap 12.59%)을 붙여 비교한다. 입력
변수와 모델 구조가 달라 해당 비교는 유효하지 않다. 새 실행 파일은 논문 최종
변수 집합의 pooled MLR만 논문 MLR 기준값과 비교한다.

### 4. backward stepwise 미구현

기존의 `use_stepwise=True`는 backward stepwise 선택을 실행하지 않고 이미 알려진
최종 변수 세 개를 하드코딩한다. 이는 최종 논문 회귀식을 실행하는 데는 충분하지만,
변수선택 절차의 재현은 아니다. 새 파일도 이 실험 목적에 맞추어 최종 변수 집합을
명시적으로 사용한다.

## 새 실행 파일

`run_baselines_corrected.py`는 기존 `run_simulation.py`를 수정하지 않는다.

- AR: 기존 프로젝트의 “같은 시간대, 과거 12일” 정의를 유지하고 rolling-history
  인덱스만 바로잡는다.
- MLR: `SSRD`, `TSR`, `Hour`를 사용한 pooled 선형회귀로 구현한다.
- 평가: 전체 테스트 표본 nRMSE 및 penalty rate 50%의 optimality gap을 저장한다.
- 검증: 날짜마다 12개 시간대가 모두 있고 필요한 기상변수에 결측이 없는지 실행 전
  검사한다.

## 재현 범위의 한계

현재 프로젝트의 가격 병합 데이터는 2014년 1~4월뿐이므로 학습 90일·테스트 30일이다.
논문은 300일 학습·100일 테스트와 자체 데이터 정렬을 사용했다. 따라서 새 결과는
구현 교정의 검증용이며, 논문의 절대 성능 수치를 그대로 재현한다고 주장할 수 없다.
## 실행 결과 (현재 프로젝트 데이터)

2026-08-26에 `run_baselines_corrected.py`를 실행했다. 데이터 검증 결과 120일 모두
하루 12개 관측치를 가지며, AR/MLR에 쓰는 기상변수 결측은 없었다. 아래 표는
전체 테스트 표본의 nRMSE 및 penalty rate 50%의 optimality gap을 기준으로, 논문,
기존 `run_simulation.py`의 저장 결과, 수정 구현을 한 표에 둔 것이다.

| 계열 | 지표 | 논문 (300일/100일) | 기존 `run_simulation.py` (90일/30일) | 수정 구현 (90일/30일) |
|---|---|---:|---:|---:|
| AR | nRMSE (%) | 34.76 | 86.5501 | 64.5435 |
| AR | Optimality gap (%) | 15.04 | 28.1617 | 19.8000 |
| MLR | nRMSE (%) | 21.76 | 45.0663¹ | 67.7498² |
| MLR | Optimality gap (%) | 12.59 | 15.0248¹ | 32.9124² |

¹ 기존 실행은 시간대별로 분리한 `MLR(sw-3vars)`이다. `SSRD`, `TSR`, `hour`라는
세 입력을 사용하지만 각 모델 안에서 hour가 상수이므로, 논문 MLR과 동일한 식이
아니다.

² 수정 실행은 모든 날짜·시간대 1,080개 학습 행을 함께 사용한 pooled
`MLR(SSRD, TSR, Hour)`이다. 논문 식의 변수 구조에 맞추었지만, 현 데이터 기간과
기상정보 정렬은 논문과 다르다.

참고로 기존 실행의 12변수 MLR은 nRMSE 50.4931%, optimality gap 16.0550%였다.
이는 논문의 최종 3변수 MLR과 다른 비교 대상이므로 주 표에는 넣지 않았다.

수정 AR의 큰 변화는 테스트 이력 인덱스 오류가 실제 성능 산출을 왜곡했음을 확인한다.
수정 pooled MLR이 현 데이터에서 더 낮은 성능을 보인다고 해서 기존 시간대별 MLR이
논문에 더 충실하다는 뜻은 아니다. 두 모델은 서로 다른 회귀식이다. 이 차이는 현재
90/30 표본, 논문과 다른 데이터 기간 및 기상정보 정렬을 포함해 별도로 원인을
검증해야 한다. 논문의 300/100 실험 수치와 현재 수치를 동등하게 비교하거나
“재현 성공/실패”로 결론 내릴 수 없다.

CSV 결과는 `results/simulation_output/baseline_corrected_results.csv`, 각 시간대
예측값은 `results/simulation_output/baseline_corrected_predictions.csv`에 저장했다.
