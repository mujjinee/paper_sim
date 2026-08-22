# 논문 실험 기간 상세 분석: MISO 가격 데이터가 필요한 실제 기간



## 작업 날짜



2026-08-22



## 논문 Section 5.1 (Experiment design)의 원문



> "The sizes of training and testing data used in numerical experiments are **300 days and 100 days**, respectively. For this numerical experiment, we set the maximum capacity of the panels as 30 MW. We divide the solar power, day-ahead, and real-time price data into two sets, **three months as the training and one month as the testing data**."



→ 논문 내부에 **서술적 표현(300일/100일)**과 **구체적 설명(3개월 학습/1개월 테스트)**이 혼재되어 있음. 둘은 모순이 아님 — 두 가지 실험 설정을 모두 언급한 것임.



## GEFCom2014 Solar 데이터 기간



- 전체: 2012-04-01 ~ 2014-07-01 UTC (약 28개월, 853일)

- Zone 1~3, 시간별, POWER (0~1 정규화) + 12개 기상 변수



## MISO 가격 데이터 — 논문이 밝힌 정보



| 항목 | 논문 내용 |

|------|-----------|

| 출처 | MISO (Midcontinent Independent System Operator) [20] |

| Fig. 1 | "average hourly day-ahead and real-time price for the Midcontinent Independent System Operator (**MISO**) **for 4 months**" |

| 노드/허브 | 밝히지 않음. "Data will be made available on request" |

| 정확한 기간 | 밝히지 않음 |

| Day-ahead + Real-time | 둘 다 사용 |



**결론: 논문은 MISO 가격 데이터의 정확한 시작/끝 날짜를 밝히지 않음.**



## Fig. 1에서 읽을 수 있는 정보



Fig. 1은 4개월간의 일평균 DA/RT 가격의 시간대별 분포를 보여줌:

- DA 가격: 평균 $40~50/MWh (낮 시간대에 높음)

- RT 가격: 평균 $30~40/MWh (DA보다 낮음)

- 24시간 중 일부 시간대(특히 낮)에 RT 가격이 DA보다 높은 경우 있음



## 300일 학습 + 100일 테스트의 의미



- 300일 × 12시간 = 3,600시간의 학습 데이터 (AR lag=12이므로 최소 3,588일치 행 필요)

- 100일 × 12시간 = 1,200시간의 테스트 데이터

- **핵심: 300일 학습하려면 MISO 가격 데이터도 최소 300일 이상 필요**



## 현재 우리의 상황



| | 논문 | 우리 |

|--|------|------|

| 학습 데이터 | 300일 | 90일 (1~3월) |

| 테스트 데이터 | 100일 | 30일 (4월) |

| MISO 가격 | 기간 불명, 최소 300일+ | 2014.01~2014.04 (약 120일) |

| Solar+Weather | GEFCom2014 전체 가능 | 2014.01~2014.04로 제한됨 |



**GEFCom2014 Solar 데이터는 2012.04~2014.07까지 존재하므로, Solar/Weather 측면에서는 300일 학습이 충분함. 문제가 되는 것은 MISO 가격 데이터만 4개월로 제한되어 있다는 점.**



## MISO 가격 데이터 확장이 필요한지 판단



### Scenario A: 논문이 실제로 300일 학습했을 경우

- MISO 가격 데이터가 **최소 10개월 이상** 필요 (300일 학습 + 100일 테스트 = 약 400일)

- → MISO 가격 데이터를 대폭 확장해야 함



### Scenario B: 논문이 3개월/1개월 설정으로 실험했을 경우

- 학습 90일 + 테스트 30일 = 4개월

- → 현재 우리 설정과 일치. 그러나 nRMSE가 논문 결과와 2~2.5배 차이남

- → 이는 **같은 기간이라도 다른 시기의 데이터**를 썼을 가능성



### 가능성이 높은 시나리오

논문의 서술("300 days and 100 days")을 숫자로 명시한 점을 고려하면:

1. **GEFCom2014 Solar 데이터 전체 기간(2012.04~2014.07) 중 300일을 학습, 100일을 테스트로 사용**

2. **MISO 가격 데이터도 동일 기간에 대응되도록 맞춰 사용**

3. Solar/Weather는 GEFCom2014에 전체 기간 존재하므로 문제 없음

4. MISO 가격도 2012~2014년 데이터를 별도로 확보했을 것



## MISO 가격 데이터 확장 방안



### 확장해야 할 기간

- **최소 2012년 4월 ~ 2014년 7월** (GEFCom2014 Solar 전체 기간과 일치)

- 학습 300일 + 테스트 100일 = 400일 이상 필요



### 확장 방법

1. **MISO Settlement Point Prices 아카이브**에서 2012~2014년 월별 데이터 다운로드

   - URL: https://www.misoenergy.org/markets-and-scheduling/practice-and-policy-proposals/settlement-point-prices/

   - 월별 zip 파일 (da_pr_xls.zip, rt_pr_xls.zip) formato

2. 이미 다운로드한 2014.01~2014.04에 추가해서 2012년 전체 데이터 수집

3. 기존 `process_data.py` 방식으로 Excel 파싱 → CSV 통합



### 대안: MISO 가격이 없더라도 solar-only 학습

- AR 모델은 solar 데이터만 필요 → MISO 없이 300일 학습 가능

- MLR 모델은 weather+solar만 필요 → MISO 없이 300일 학습 가능

- **하지만 optimality gap 계산과 최적화 모델은 DA/RT 가격이 필수**

- → nRMSE 개선만 원한다면 MISO 없이 학습 기간 확장 가능

- → 전체 논문 재현을 위해서는 MISO 가격 데이터 확장 필수



## 요약



| 질문 | 답변 |

|------|------|

| 논문이 어느 기간을 사용했는지? | 정확히 밝히지 않음. "Data will be made available on request" |

| 300일 학습/100일 테스트? | Section 5.1에서 명시적 언급. 3개월/1개월도 함께 서술 |

| MISO 가격 기간이 필요한 분량? | 최소 400일 (300일 학습 + 100일 테스트) |

| GEFCom2014 Solar로 가능한가? | 예, 2012.04~2014.07 전체 기간 데이터 보유 |

| 현재 MISO 데이터로 가능한가? | 아님. 4개월(120일)만 보유 |

| MISO 확장이 필수인가? | AR/MLR의 nRMSE 개선만: 선택사항. 최적화/gap 재현: **필수** |

| MISO 기간 확장 범위? | 2012.04 ~ 2014.07 (GEFCom Solar 기간과 일치) |