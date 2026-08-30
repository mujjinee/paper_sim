# -*- coding: utf-8 -*-
"""
MLR(다중선형회귀) 모델만 떼어내서, AR 슬라이딩 실험의 "블록9"와
정확히 같은 학습/테스트 구간에 대해 nRMSE만 확인하는 스크립트.

블록9 구간 (ar_only_400day_sliding.py 에서 그대로 가져옴):
    학습: 2012-11-28 ~ 2013-09-23 (300일)
    테스트: 2013-09-24 ~ 2014-01-01 (100일)

논문 Eq.(6): Ŝ_t = α + Σ_k β_k V_{k,t}
  - "하루의 모든 시간대에 대해 동일한 계수 집합"을 쓰는 pooled 모델 (AR처럼 시간대별로
    12개 모델을 따로 만드는 게 아님).
  - 최종 변수(backward stepwise로 논문이 고른 3개): SSRD(VAR169), TSR(VAR178), Hour.

손실함수: 논문 Eq.(8) 절대오차합(LAD) -> AR과 동일하게 QuantileRegressor(quantile=0.5, alpha=0).

SSRD/TSR 차분(deaccumulation) - D:/03_JiWon/APEN/data/readme.md, operational_corrected 참고:
  VAR169(SSRD), VAR178(TSR)은 각 시간의 독립된 값이 아니라, "01,02,...,23,00" 순서로
  묶인 24시간 예보 묶음 안에서 "누적"되는 값이다. 그래서 이번 시간 값에서 바로 직전
  시간 값을 빼야 진짜 "이번 한 시간 동안의 증분"(dSSRD, dTSR)이 나온다.
  예: 한 묶음의 누적값이 10, 25, 40 이면 시간별 증분은 10, 15, 15 다.
  묶음의 첫 시간은 뺄 대상(직전 누적값)이 없으므로 원래 누적값을 그대로 증분으로 쓴다.
  이 차분은 zone/낮시간 선택보다 먼저, 전체 원본에 대해 수행해야 한다.

MLR은 AR과 달리 "내일 날씨 예보"를 입력으로 쓰는 모델이라, AR처럼 하루씩 굴리면서
예측할 필요가 없다. 테스트 기간의 SSRD/TSR 값(=예보값)을 그대로 모델에 넣어
한 번에 전체 테스트 기간을 예측한다.

실행 방법:
    python mlr_only_block9.py
"""

import os                                            # 파일 경로를 다루기 위한 표준 라이브러리
import numpy as np                                   # 숫자 배열 계산을 위한 라이브러리
import pandas as pd                                  # 표 데이터를 다루기 위한 라이브러리
from sklearn.linear_model import QuantileRegressor   # 절대오차(LAD) 회귀를 위한 모델

BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 이 파일이 있는 폴더 경로
PREDICTORS_FILE = os.path.join(BASE_DIR, "data", "predictors15.csv")   # 원본 태양광+기상 데이터 파일

ZONE_ID_LIST = [1, 2, 3]     # 태양광 발전소 3곳
HOURS_PER_DAY = 12           # 하루 중 태양광이 나오는 시간대 개수 (UTC 0~11시)

# 블록9와 정확히 같은 구간 (ar_only_400day_sliding.py 결과에서 그대로 가져옴)
TRAIN_START = pd.Timestamp("2012-11-28").date()   # 학습 시작일
TRAIN_END = pd.Timestamp("2013-09-23").date()      # 학습 마지막일 (300일째)
TEST_START = pd.Timestamp("2013-09-24").date()     # 테스트 시작일
TEST_END = pd.Timestamp("2014-01-01").date()        # 테스트 마지막일 (100일째)

PAPER_MLR_NRMSE = 21.76      # 논문 Table 3에 실려있는 MLR(3변수)의 nRMSE(%). 비교 기준값


# ============================================================
# 1. 데이터 준비: predictors15.csv -> 누적값을 증분으로 차분 -> 필요한 열만 남긴 표
# ============================================================
def deaccumulate(raw):
    """
    VAR169(SSRD), VAR178(TSR)은 "01,02,...,23,00" 24시간 예보 묶음 안에서
    누적되는 값이다. zone별로, 묶음별로 차분해서 시간별 증분(dSSRD, dTSR)을 만든다.

    묶음을 찾는 방법: 이 행의 시각에서 1시간을 빼고 날짜만 남기면(정오 이하 버림),
    그 묶음이 "시작한 날짜"가 나온다. 예를 들어
      - D일 01:00 행 -> (D일 01:00 - 1시간) = D일 00:00 -> 묶음 날짜 = D
      - D+1일 00:00 행(=묶음의 마지막 시각, 24번째) -> (D+1일 00:00 - 1시간)
        = D일 23:00 -> 묶음 날짜 = D
    즉 D일 01:00 ~ D+1일 00:00 까지 총 24개 행이 전부 "묶음 날짜 = D" 로 같이 묶인다.
    """
    raw = raw.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)  # zone별, 시간순으로 확실히 정렬

    bundle_start = raw["TIMESTAMP"] - pd.Timedelta(hours=1)   # 이 행의 시각에서 1시간을 뺀 임시 시각
    raw["bundle_date"] = bundle_start.dt.normalize()            # 그 임시 시각의 "날짜"만 남김 (묶음 이름표)

    groups = raw.groupby(["ZONEID", "bundle_date"], sort=False)  # zone + 묶음 단위로 그룹을 만듦
    step_in_bundle = groups.cumcount() + 1                        # 묶음 안에서 몇 번째 시간인지 (1~24)
    is_first_step = step_in_bundle.eq(1)                           # 묶음의 "첫 번째" 시간인지 여부 (뺄 대상이 없음)

    for accumulated_col, increment_col in [("VAR169", "dSSRD"), ("VAR178", "dTSR")]:
        increment = groups[accumulated_col].diff()                  # 묶음 안에서 "이번 값 - 직전 값" (첫 시간은 NaN이 됨)
        increment.loc[is_first_step] = raw.loc[is_first_step, accumulated_col]  # 첫 시간은 누적값 자체를 증분으로 씀
        raw[increment_col] = increment                               # 계산한 증분을 새 열로 저장

    return raw                                                       # 증분 열(dSSRD, dTSR)이 추가된 전체 표를 돌려줌


def load_one_zone(zone_id, raw_with_increments):
    """
    차분이 이미 끝난 전체 표(raw_with_increments)에서 zone_id에 해당하는 행만 골라,
    MLR에 필요한 열(날짜, 시간대, 발전량, dSSRD, dTSR)만 남긴 표를 돌려준다.
    """
    zone_rows = raw_with_increments[raw_with_increments["ZONEID"] == zone_id].copy()   # 원하는 zone만 남김
    zone_rows = zone_rows[zone_rows["TIMESTAMP"].dt.hour < HOURS_PER_DAY]                # 낮 시간대(0~11시)만 남김
    zone_rows["date"] = zone_rows["TIMESTAMP"].dt.date          # 날짜만 뽑아 새 열로 저장
    zone_rows["hour_idx"] = zone_rows["TIMESTAMP"].dt.hour       # 시간대(0~11)만 뽑아 새 열로 저장

    small_table = zone_rows[["date", "hour_idx", "POWER", "dSSRD", "dTSR"]].copy()   # 필요한 열만 남김
    small_table = small_table.rename(columns={"dSSRD": "ssrd", "dTSR": "tsr"})        # 이후 코드에서 쓰기 편하게 이름 통일
    small_table = small_table.dropna()                          # 값이 비어있는 행은 제거

    return small_table                                           # 정리된 표를 돌려줌


# ============================================================
# 2. 평가지표: nRMSE
# ============================================================
def calculate_nrmse(actual_values, predicted_values):
    error = actual_values - predicted_values          # 실제값과 예측값의 차이
    rmse = np.sqrt(np.mean(error ** 2))                 # 오차 제곱의 평균 -> 제곱근 = RMSE
    average_actual = np.mean(actual_values)              # 실제 발전량들의 평균값
    return 100 * rmse / average_actual                   # RMSE를 평균으로 나눠 %로 표현


# ============================================================
# 3. 한 zone에 대해 MLR 학습 + 테스트 + 채점
# ============================================================
def run_mlr_for_one_zone(zone_id, raw_with_increments):
    zone_label = f"z{zone_id:02d}"
    table = load_one_zone(zone_id, raw_with_increments)         # 이 zone의 (날짜,시간대,발전량,ssrd,tsr) 표

    train_table = table[(table["date"] >= TRAIN_START) & (table["date"] <= TRAIN_END)]  # 학습 구간만 남김
    test_table = table[(table["date"] >= TEST_START) & (table["date"] <= TEST_END)]      # 테스트 구간만 남김

    print(f"[{zone_label}] 학습 행 수: {len(train_table)} (300일 x 12시간 = 3600행이어야 정상)")
    print(f"[{zone_label}] 테스트 행 수: {len(test_table)} (100일 x 12시간 = 1200행이어야 정상)")

    # 입력변수 X = [ssrd, tsr, hour_idx] 3개, 정답 y = POWER (논문 Eq.6, pooled 모델)
    X_train = train_table[["ssrd", "tsr", "hour_idx"]].to_numpy()   # 학습 입력값 표를 숫자 배열로
    y_train = train_table["POWER"].to_numpy()                        # 학습 정답값(발전량)을 숫자 배열로

    X_test = test_table[["ssrd", "tsr", "hour_idx"]].to_numpy()      # 테스트 입력값 표를 숫자 배열로
    y_test = test_table["POWER"].to_numpy()                           # 테스트 정답값(발전량)을 숫자 배열로

    # 절대오차합(LAD)을 최소화하는 회귀모델 (논문 Eq.8과 동일한 손실함수, AR과도 동일)
    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")   # 모델 생성
    model.fit(X_train, y_train)                                          # 학습 실행 (한 세트의 계수만 만들어짐)

    # MLR은 "내일 날씨 예보"가 이미 주어졌다고 가정하는 모델이라, AR처럼 하루씩
    # 굴리면서 예측할 필요 없이, 테스트 구간 전체를 한 번에 예측한다.
    y_pred = model.predict(X_test)                                       # 테스트 입력값으로 바로 예측
    y_pred = np.clip(y_pred, 0, 1)                                        # 발전량은 0~1 사이여야 하므로 범위를 잘라냄

    nrmse_score = calculate_nrmse(y_test, y_pred)                         # nRMSE(%) 계산
    print(f"[{zone_label}] MLR nRMSE = {nrmse_score:.2f}%")

    return nrmse_score


# ============================================================
# 4. 전체 실행: zone마다 반복
# ============================================================
def main():
    print(f"학습기간: {TRAIN_START} ~ {TRAIN_END} (300일)")
    print(f"테스트기간: {TEST_START} ~ {TEST_END} (100일)  <- AR 블록9와 동일 구간\n")

    raw = pd.read_csv(PREDICTORS_FILE)                        # csv 전체를 한 번만 읽어옴
    raw["TIMESTAMP"] = pd.to_datetime(                        # TIMESTAMP 문자열을 날짜+시간 타입으로 변환
        raw["TIMESTAMP"], format="%Y%m%d %H:%M"
    )
    raw_with_increments = deaccumulate(raw)                    # zone/낮시간 거르기 전에 먼저 누적값을 증분으로 차분

    results = {}
    for zone_id in ZONE_ID_LIST:                    # zone 1, 2, 3 순서대로
        results[f"z{zone_id:02d}"] = run_mlr_for_one_zone(zone_id, raw_with_increments)
        print()

    print("=== MLR(pooled, dSSRD+dTSR+Hour, LAD 손실, 누적값 차분 적용) - 블록9 구간 결과 ===")
    print(f"{'Zone':<8}{'nRMSE':>10}{'논문 대비':>12}")
    for label, val in results.items():
        print(f"{label:<8}{val:>9.2f}%{val - PAPER_MLR_NRMSE:>+11.2f}%p")
    print(f"{'논문':<8}{PAPER_MLR_NRMSE:>9.2f}%")


if __name__ == "__main__":
    main()
