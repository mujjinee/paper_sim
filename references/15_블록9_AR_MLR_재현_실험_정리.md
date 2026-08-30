# 블록9 AR/MLR 재현 실험 정리

## 작업 날짜

2026-08-29 ~ 2026-08-30

## 배경

`merged_for_simulation.csv`를 새로 만드는 작업에서 출발해, AR·MLR 단독 성능을
논문(Karimi & Kwon, 2022, Applied Energy 326: 119929)과 비교하고, 발견된 버그를
하나씩 고쳐나가며 최종적으로 논문이 제안한 4개 모형(기본 AR, 제안 AR, 기본 MLR,
제안 MLR)을 재현한 세션의 기록이다.

---

## 1. 데이터 파이프라인 (`build_merged_for_simulation.py`)

`extract_prices_data.py` + `extract_solar_weather.py` + `merge_for_simulation.py`를
하나로 합쳐, `data/` 폴더의 원본만으로 zone별 `merged_for_simulation_z01/02/03.csv`를
바로 만드는 스크립트로 재작성했다. 세션 동안 다음이 차례로 추가됐다.

1. **전체 구간 확장**: `da_lmp_prices.csv`/`rt_lmp_prices.csv` 존재 구간(2013-03-26~
   2014-04-30)에 맞춰 시작일을 늘림. 이후 확장된 가격 파일(`*_extended.csv`,
   2012-11-01~2014-04-30, 갭·중복 없음 확인됨)로 교체하면서 `START_DATE`도
   `2012-11-01`로 재확장.
2. **가격 타임존 보정 (EST → UTC)**: MISO 원본 엑셀 헤더에 `Peak Hour: HE 21 (EST)`가
   명시되어 있음을 확인 — MISO는 **고정 EST(UTC-5, 서머타임 미적용)**로 발표한다.
   GEFCom2014 태양광/기상은 UTC라서, 가격 timestamp에 `+5시간`을 더해 UTC로 맞춘 뒤
   병합하도록 수정. (보정 전에는 가격과 태양광/기상이 5시간 어긋난 채로 merge되고
   있었음 — AR nRMSE엔 영향 없지만 optimality gap 계산에는 영향을 줌)
3. **SSRD/TSR 차분(deaccumulation)**: `D:/03_JiWon/APEN/data/readme.md`,
   `operational_corrected`를 참고해 VAR169(SSRD)/VAR178(TSR)이 "01,02,...,23,00"
   24시간 예보 묶음 안에서 누적되는 값임을 확인. zone·구간 필터링 전에 원본 전체를
   묶음 단위로 차분해 `dssrd`/`dtsr` 열을 추가(원래 누적값 `ssrd`/`tsr`은 하위호환용으로
   유지).
4. **Sydney 현지시간 열 추가**: `timestamp`(UTC)는 유지하고, `local_timestamp`/
   `local_date`/`local_hour`를 `Australia/Sydney` 기준으로 추가 계산해 저장.
   (12시간 낮 시간대 필터링 자체는 하지 않고 각 분석 스크립트가 `local_hour`로 고름)

최종 출력: `merged_for_simulation_z01/02/03.csv`, 13,099행,
2012-11-01 05:00 ~ 2014-04-30 23:00.

### 중간에 확인된 구버전 버그 (참고용)

- `data_org/prices/rt_lmp_prices.csv`(구버전 추출본)는 2014-04-29 23시에서 끊겨
  있어서, 그걸로 만든 `data_org/merged_for_simulation.csv`는 4/30 RT 가격 24개가
  전부 4/29 23시 값으로 forward-fill돼 있었음. 지금 쓰는 `data/rt_lmp_prices.csv`
  (및 `*_extended.csv`)는 이 문제 없음 — 재검증 완료.

---

## 2. AR 단독 실험

### 2.1 기본 구현

`ar_only.py`: 논문 Eq.(3) direct multi-step AR. 손실함수는 논문 4.3.1절 서술대로
절대오차합(LAD) — `sklearn.linear_model.QuantileRegressor(quantile=0.5, alpha=0)`
사용. 처음엔 "과거 12일의 같은 시간대" 값을 lag로 썼으나(원래 `run_simulation.py`
방식), 이후 논문 Eq.(3)의 `S_{t-h-l}` 인덱싱을 다시 읽고 **"직전 하루 12시간"을
입력으로 쓰는 게 더 논문에 가깝다**고 결론 내림 (§4 참고).

### 2.2 데이터 구간별 nRMSE (UTC 0~11시 기준, 차분 전)

| 구간 | 테스트 기간 | z01 | z02 | z03 |
|---|---|---:|---:|---:|
| 가격 데이터 제약 401일 | 2014-01-21~04-30 | 54.91% | 52.38% | 53.36% |
| 전체 구간 마지막 400일 | 2014-03-23~06-30 | 75.51% | 71.54% | 69.03% |
| 전체 구간 300/100일 블록1 | 2013-01-27~05-06 | 48.00% | 45.70% | 43.54% |
| 전체 구간 300/100일 블록2 | 2014-03-03~06-10 | 62.52% | 60.78% | 61.17% |
| **논문** | (미공개) | **34.76%** | | |

`ar_only_120day_blocks.py`(90일 학습/30일 테스트, 120일 블록 6개)와
`ar_only_400day_sliding.py`(300/100일, 30일씩 겹치게 슬라이딩 15블록)로 계절성을
확인:

- **평균 발전량이 높고 변동계수(CV)가 낮은 계절(9~1월, 남반구 봄~여름)일수록 nRMSE가
  낮다** — 1월 평균 0.339·CV 0.90(최고), 6월 평균 0.198·CV 1.19(최악).
- 슬라이딩 실험에서 **블록9~10(테스트 2013-09-24~2014-01-31)이 전 구간 중 최저
  nRMSE**: z01 38.84%(블록10), z02 38.05%(블록9), z03 38.74%(블록10) — 논문과 차이
  3~4%p까지 좁혀짐. 이후 이 위치를 "블록9"로 고정해 MLR·제안모형 실험에 계속 사용.

### 2.3 타임존 재검증 (Sydney 현지시간)

논문 5.1절이 "낮 12시간 = 호주 현지시간 09:00~21:00"이라고 명시함에도 지금까지는
UTC 0~11시를 썼음. `sydney_local_block_experiment.py`로 `Australia/Sydney` 변환 +
현지시간 9~20시 선택 + dSSRD/dTSR 차분을 모두 적용해 블록9 위치를 재실행:

| Zone | AR (UTC) | AR (Sydney) | 개선 |
|---|---:|---:|---:|
| z01 | 39.83% | 37.80% | -2.03%p |
| z02 | 38.05% | **35.67%** | -2.38%p |
| z03 | 39.42% | 36.73% | -2.69%p |

z02(35.67%)가 이 세션에서 논문(34.76%)에 가장 근접한 AR 결과.

---

## 3. MLR 단독 실험 (블록9, Zone1)

`mlr_only_block9.py`: 논문 Eq.(6) pooled MLR — 하루 모든 시간대에 **동일한 계수
집합**을 쓰는 모델(AR처럼 시간대별 12개 모델이 아님). 최종 변수는 논문이 backward
stepwise로 고른 **SSRD, TSR, Hour 3개** (Hour는 더미가 아니라 스칼라 하나 — 논문
369행: "월(Month)과 시(Hour)도 추가 독립변수로 고려... 14개 변수 중... 최종적으로
SSRD, TSR, Hour 세 변수가 선택").

| 단계 | z01 | z02 | z03 | 논문 |
|---|---:|---:|---:|---:|
| 원시 SSRD/TSR (차분 전) | 41.78% | 51.86% | 43.83% | 21.76% |
| **dSSRD/dTSR 차분 적용** | **32.46%** | **32.78%** | **29.71%** | 21.76% |

차분 전에는 MLR이 같은 블록의 AR보다도 나빴는데(예: z02 51.86% vs AR 38.05%),
VAR169/178이 "예보 시작부터의 누적값"이라는 걸 놓친 게 원인이었음. 차분 한 번으로
10~19%p 개선.

---

## 4. `operational_corrected`와의 비교에서 찾은 것들

`D:\03_JiWon\APEN\operational_corrected`(별도 프로젝트)와 결과가 크게 다른 이유를
추적하며 다음을 확인:

1. **AR lag 설계 차이**: 그쪽은 "직전 하루 12시간"을 입력으로 쓰는데(Eq.3의
   `S_{t-h-l}` 인덱싱과 일치), 우리는 "과거 12일 같은 시간대"를 썼음 → **논문에
   더 가까운 쪽은 operational_corrected**였고, 최종 스크립트에 이 방식을 채용.
2. **이익함수(Eq. 1a) 재확인** — 논문 원문(PDF 7페이지, PyMuPDF로 직접 렌더링해
   대조):
   ```
   max Σ [ DP_t·x_t + RP_t·y'_t − PC_t·y''_t ]    (1a, 3항)
   ```
   우리가 그동안 써온 `unit_commitment_profit`(예: `run_simulation.py`,
   `run_experiment_comparison.py`)은 `- rt_price*shortage` 항이 추가로 들어간
   **4항짜리 잘못된 버전**이었음. `operational_corrected`의 3항 버전이 논문과 일치.
3. **오라클 정의 차이**: 논문 Eq.(13)과 5.1절 서술("일간전 가격이 실시간 가격보다
   높으면 약정량은 생성된 태양광 에너지만큼 최대한 많아야 하고, 그렇지 않으면
   0으로")은 오라클을 **`{0, 실제발전량}` 중 선택**으로 명시. `operational_corrected`는
   여기에 **설비최대(1.0)까지 포함한 3후보**를 써서(3항 이익함수 하에서 penalty_rate
   <100%일 때 항상 설비최대가 더 유리하다는 수학적 성질 때문), 그쪽의 Baseline AR
   optimality gap이 51%까지 치솟는 원인이 됨. 논문 서술에 더 충실한 쪽은
   `{0,S}`라고 판단해 채택.
4. **Eq.(13) 원문 자체의 오타**: PDF를 직접 봐도 부족분 항이 `max{0, S_t−Ŝ_t}`로
   잉여항과 동일하게 인쇄돼 있음(경제적으로 말이 안 됨, `RP_k`도 아래첨자 오타).
   Eq.(1a)/y',y'' 정의와 5.1절 서술로 미루어 `max{0, Ŝ_t−S_t}`(부족)의 오타로
   판단하고 그렇게 구현.

---

## 5. 논문 제안 모형(Eq.9/10) 재현 — 최종 결과 (블록9, Zone1)

`paper_proposed_block9.py`: 위 1~4의 수정사항을 모두 반영해 기본/제안 AR·MLR
4개 모형을 학습·평가. 제안모형(MILP)은 `operational_corrected`의 검증된 정식화를
재사용하되, **학습 목적함수의 정규화 분모(오라클 합)는 `operational_corrected`
원래 방식({0,S,capacity})을, 최종 평가 지표는 논문식 오라클({0,S})을 쓰도록
분리**했다 — 평가용 오라클을 학습에도 그대로 쓰면 W1=1,W2=20의 의도된 균형이
깨져 계수가 발산(AR nRMSE 103%까지 치솟음)하는 걸 실제로 확인했기 때문.

학습: 2012-11-28~2013-09-23(300일), 테스트: 2013-09-24~2014-01-01(100일),
penalty_rate=50%, W1=1, W2=20 (논문 Table 3/4 비교 지점).

| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |
|---|---:|---:|---:|---:|---:|---:|
| 기본 모형 AR | 34.76% | 38.783485% | 15.04% | 3.696468% | +4.023485%p | -11.343532%p |
| 논문 제안 모형 AR | 34.89% | 38.577529% | 13.91% | 3.631549% | +3.687529%p | -10.278451%p |
| 기본 모형 MLR | 21.76% | 32.183847% | 12.59% | 2.745470% | +10.423847%p | -9.844530%p |
| 논문 제안 모형 MLR | 21.92% | 32.174187% | 11.91% | 2.586135% | +10.254187%p | -9.323865%p |

같은 예측식끼리 비교(제안 − 기본):

| 비교 | Δ nRMSE | Δ optimality gap |
|---|---:|---:|
| 제안 AR − 기본 AR | -0.205956%p | -0.064919%p |
| 제안 MLR − 기본 MLR | -0.009660%p | -0.159335%p |

두 경우 다 optimality gap이 줄어드는 방향은 논문과 일치. nRMSE는 논문에서는
살짝 늘어나는데 여기선 거의 그대로거나 살짝 줄어듦(0.01~0.2%p, 잡음 수준).

---

## 6. 남은 한계

- **논문이 정확한 300/100일 날짜, MISO 노드를 공개하지 않음** ("Data will be made
  available on request") — 지금 구간이 논문과 동일한 표본이라는 보장이 없음.
- **optimality gap 절대값이 논문보다 훨씬 작음**(2.6~3.7% vs 논문 12~15%) — 이익함수/
  오라클 정의를 논문 서술에 최대한 맞췄는데도 이 정도 차이가 남아있어, gap 계산
  방식에 대한 이해가 논문 저자와 여전히 다를 가능성이 있음.
- **제안모형 MILP의 학습용/평가용 오라클 분리는 논문에 없는 임의 조건** —
  `D:\03_JiWon\APEN\docs\02_RESULTS_AND_LIMITATIONS.md` 6.4절과 같은 성격의,
  계산을 가능하게 하려고 임의로 정한 부분.
- **태양광 발전량(GEFCom2014 Zone1)과 가격(MISO System LMP)이 같은 시장 데이터가
  아님** — timestamp로만 연결한 모형 내 지표.
- `run_experiment_comparison.py`(구버전, `run_simulation.py` 계열)는 아직 4항
  이익함수 그대로 남아있음 — 필요시 별도로 고쳐야 함.

---

## 7. 이 세션에서 만든/수정한 스크립트

| 파일 | 역할 |
|---|---|
| `build_merged_for_simulation.py` | zone별 병합 데이터셋 생성 (EST→UTC, dssrd/dtsr, Sydney local time, 확장 가격) |
| `ar_only.py` | AR 단독(LAD), zone별 merged 파일 사용 |
| `ar_only_full_range.py` | AR 단독, predictors15.csv 전체 구간을 400일 블록(비중첩)으로 |
| `ar_only_120day_blocks.py` | AR 단독, 120일 블록(90학습/30테스트) 6개 |
| `ar_only_400day_sliding.py` | AR 단독, 400일 블록을 30일씩 겹치며 슬라이딩(15개) |
| `sydney_local_block_experiment.py` | AR+MLR, Sydney 현지시간 09~20시로 블록9 위치 재검증 |
| `check_price_timezone_impact.py` | 가격 EST→UTC 보정 전/후 optimality gap 비교 |
| `mlr_only_block9.py` | MLR 단독(pooled, LAD), dSSRD/dTSR 차분 적용 |
| `paper_proposed_block9.py` | 최종: 기본/제안 AR·MLR 4개 모형, 논문 Eq.1a/13 반영 |
