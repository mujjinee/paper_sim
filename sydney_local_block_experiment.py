# -*- coding: utf-8 -*-
"""
타임존을 제대로 반영한 AR / MLR 재실험 스크립트.

지금까지 썼던 UTC 00:00~11:00 낮 시간대 대신, 논문이 실제로 말하는
"호주 현지시간(Sydney) 09:00~20:00"으로 낮 시간대를 다시 정의해서
AR / MLR nRMSE가 얼마나 달라지는지 확인한다.

처리 순서 (D:\\03_JiWon\\APEN\\data\\readme.md, operational_corrected 참고):
  1. VAR169/VAR178 누적값을 시간별 증분(dSSRD, dTSR)으로 차분 (타임존 변환보다 먼저)
  2. TIMESTAMP를 UTC로 간주 -> Australia/Sydney 현지시간으로 변환
  3. 현지시간 09:00~20:00(12시간)만 낮 시간대로 사용, hour_idx = 현지시(local_hour) - 9
  4. 이전에 쓰던 "블록9"와 같은 위치(인덱스 240 = 9번째 블록, 300일 학습+100일 테스트)를
     새로 만든 현지시간 기준 날짜 목록에서 다시 뽑아 AR/MLR을 각각 돌림

비교 대상 (UTC 00:00~11:00 기준, 이전 실행 결과):
  AR  블록9  : z01=39.83%, z02=38.05%, z03=39.42%
  MLR 블록9  : z01=32.46%, z02=32.78%, z03=29.71% (dSSRD/dTSR 차분 적용)

실행 방법:
    python sydney_local_block_experiment.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTORS_FILE = os.path.join(BASE_DIR, "data", "predictors15.csv")

ZONE_ID_LIST = [1, 2, 3]
HOURS_PER_DAY = 12            # 낮 시간대 개수 (Sydney 현지 09:00~20:00)
LOCAL_HOUR_START = 9           # Sydney 현지시간 기준 낮 시간 시작 (9시)
LOCAL_HOUR_END = 21            # 낮 시간 끝 (21시, 즉 20:00까지 포함 -> 9~20시 총 12개)
LOCAL_TIMEZONE = "Australia/Sydney"

LAG_DAYS = 12                  # AR lag 일수
TRAIN_DAYS = 300               # 학습 일수
TEST_DAYS = 100                # 테스트 일수
WINDOW_DAYS = TRAIN_DAYS + TEST_DAYS   # 400일
STEP_DAYS = 30                 # 슬라이딩 간격 (이전 실험과 동일)
BLOCK_NUMBER = 9                # 이전에 봤던 "블록9"와 같은 위치를 재현
WINDOW_START_INDEX = (BLOCK_NUMBER - 1) * STEP_DAYS   # = 240

PAPER_AR_NRMSE = 34.76
PAPER_MLR_NRMSE = 21.76


# ============================================================
# 1. 원본 로딩 + 차분(deaccumulation) - zone/타임존 처리 전에 먼저 수행
# ============================================================
def load_and_deaccumulate():
    raw = pd.read_csv(PREDICTORS_FILE)
    raw["TIMESTAMP"] = pd.to_datetime(raw["TIMESTAMP"], format="%Y%m%d %H:%M")   # naive datetime (아직 UTC 표시 없음)
    raw = raw.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)         # zone별, 시간순 정렬

    bundle_start = raw["TIMESTAMP"] - pd.Timedelta(hours=1)     # 예보 묶음을 찾기 위한 임시 시각
    raw["bundle_date"] = bundle_start.dt.normalize()              # 묶음 이름표 (날짜만)

    groups = raw.groupby(["ZONEID", "bundle_date"], sort=False)
    step_in_bundle = groups.cumcount() + 1
    is_first_step = step_in_bundle.eq(1)

    for accumulated_col, increment_col in [("VAR169", "dSSRD"), ("VAR178", "dTSR")]:
        increment = groups[accumulated_col].diff()
        increment.loc[is_first_step] = raw.loc[is_first_step, accumulated_col]
        raw[increment_col] = increment

    return raw


# ============================================================
# 2. UTC -> Sydney 현지시간 변환 + 낮 시간대(9~20시) 선택
# ============================================================
def to_sydney_daylight_rows(raw_with_increments, zone_id):
    """
    zone_id에 해당하는 행만 골라서, TIMESTAMP를 UTC로 간주 -> Sydney 현지시간으로 바꾸고,
    현지시간 09:00~20:00만 남긴 표를 돌려준다.
    반환 열: local_date, hour_idx(0~11), POWER, dSSRD, dTSR
    """
    zone_rows = raw_with_increments[raw_with_increments["ZONEID"] == zone_id].copy()

    utc_ts = zone_rows["TIMESTAMP"].dt.tz_localize("UTC")             # 원본을 UTC로 간주
    local_ts = utc_ts.dt.tz_convert(LOCAL_TIMEZONE)                    # Sydney 현지시간으로 변환 (DST 자동 반영)

    zone_rows["local_date"] = local_ts.dt.date                          # 현지 날짜
    zone_rows["local_hour"] = local_ts.dt.hour                          # 현지 시(0~23)

    daylight = zone_rows[
        (zone_rows["local_hour"] >= LOCAL_HOUR_START) & (zone_rows["local_hour"] < LOCAL_HOUR_END)
    ].copy()
    daylight["hour_idx"] = daylight["local_hour"] - LOCAL_HOUR_START    # 0~11 로 다시 번호 매김

    table = daylight[["local_date", "hour_idx", "POWER", "dSSRD", "dTSR"]].dropna()
    return table


def to_daily_arrays(table):
    """(date,hour_idx,POWER,...) 행 표를 (날짜 수, 12) solar 배열 + 완전한 날짜 리스트로 변환."""
    hour_counts = table.groupby("local_date")["hour_idx"].nunique()
    complete_dates = sorted(hour_counts[hour_counts == HOURS_PER_DAY].index)
    table = table[table["local_date"].isin(complete_dates)]

    solar = np.zeros((len(complete_dates), HOURS_PER_DAY))
    for i, d in enumerate(complete_dates):
        day = table[table["local_date"] == d].sort_values("hour_idx")
        solar[i] = day["POWER"].values

    return complete_dates, solar


# ============================================================
# 3. AR 모델 (기존과 동일한 구조)
# ============================================================
class ARModel:
    def __init__(self):
        self.hour_models = {}

    def train(self, train_power_table):
        total_train_days = train_power_table.shape[0]
        for hour in range(HOURS_PER_DAY):
            X, y = [], []
            for day in range(LAG_DAYS, total_train_days):
                X.append(train_power_table[day - LAG_DAYS:day, hour])
                y.append(train_power_table[day, hour])
            model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
            model.fit(np.array(X), np.array(y))
            self.hour_models[hour] = model

    def predict_one_day(self, last_12_days):
        forecast = np.zeros(HOURS_PER_DAY)
        for hour in range(HOURS_PER_DAY):
            x = last_12_days[:, hour].reshape(1, -1)
            forecast[hour] = np.clip(self.hour_models[hour].predict(x)[0], 0, 1)
        return forecast


def calculate_nrmse(actual, predicted):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return 100 * rmse / np.mean(actual)


# ============================================================
# 4. 실행: zone마다 AR + MLR을 블록9 위치(인덱스 240)에서 실행
# ============================================================
def run_ar(dates, solar):
    window_dates = dates[WINDOW_START_INDEX:WINDOW_START_INDEX + WINDOW_DAYS]
    window_solar = solar[WINDOW_START_INDEX:WINDOW_START_INDEX + WINDOW_DAYS]

    train_solar = window_solar[:TRAIN_DAYS]
    test_solar = window_solar[TRAIN_DAYS:]
    print(f"  학습: {window_dates[0]} ~ {window_dates[TRAIN_DAYS - 1]}  "
          f"테스트: {window_dates[TRAIN_DAYS]} ~ {window_dates[-1]}")

    ar = ARModel()
    ar.train(train_solar)

    history = train_solar.copy()
    forecasts = np.zeros_like(test_solar)
    for day in range(TEST_DAYS):
        forecasts[day] = ar.predict_one_day(history[-LAG_DAYS:])
        history = np.vstack([history, test_solar[day]])

    return calculate_nrmse(test_solar.flatten(), forecasts.flatten()), window_dates[TRAIN_DAYS - 1], window_dates[TRAIN_DAYS], window_dates[-1]


def run_mlr(table, train_dates, test_dates):
    train_rows = table[table["local_date"].isin(train_dates)]
    test_rows = table[table["local_date"].isin(test_dates)]

    X_train = train_rows[["dSSRD", "dTSR", "hour_idx"]].to_numpy()
    y_train = train_rows["POWER"].to_numpy()
    X_test = test_rows[["dSSRD", "dTSR", "hour_idx"]].to_numpy()
    y_test = test_rows["POWER"].to_numpy()

    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0, 1)

    return calculate_nrmse(y_test, y_pred)


def main():
    print(f"블록{BLOCK_NUMBER} 재현 (Sydney 현지시간 09:00~20:00 기준), window_start_index={WINDOW_START_INDEX}\n")

    raw_with_increments = load_and_deaccumulate()

    ar_results = {}
    mlr_results = {}

    for zone_id in ZONE_ID_LIST:
        label = f"z{zone_id:02d}"
        table = to_sydney_daylight_rows(raw_with_increments, zone_id)
        dates, solar = to_daily_arrays(table)

        print(f"[{label}] 전체 사용 가능 날짜: {len(dates)}일 ({dates[0]} ~ {dates[-1]})")

        ar_nrmse, train_end, test_start, test_end = run_ar(dates, solar)
        ar_results[label] = ar_nrmse
        print(f"[{label}] AR nRMSE  = {ar_nrmse:.2f}%")

        train_dates = set(dates[WINDOW_START_INDEX:WINDOW_START_INDEX + TRAIN_DAYS])
        test_dates = set(dates[WINDOW_START_INDEX + TRAIN_DAYS:WINDOW_START_INDEX + WINDOW_DAYS])
        mlr_nrmse = run_mlr(table, train_dates, test_dates)
        mlr_results[label] = mlr_nrmse
        print(f"[{label}] MLR nRMSE = {mlr_nrmse:.2f}%\n")

    print("=== Sydney 현지시간 기준 결과 요약 (블록9 위치) ===")
    print(f"{'Zone':<8}{'AR':>10}{'MLR':>10}")
    for zone_id in ZONE_ID_LIST:
        label = f"z{zone_id:02d}"
        print(f"{label:<8}{ar_results[label]:>9.2f}%{mlr_results[label]:>9.2f}%")
    print(f"{'논문':<8}{PAPER_AR_NRMSE:>9.2f}%{PAPER_MLR_NRMSE:>9.2f}%")

    print("\n=== 이전(UTC 0~11시 기준) 결과와 비교 ===")
    prev_ar = {"z01": 39.83, "z02": 38.05, "z03": 39.42}
    prev_mlr = {"z01": 32.46, "z02": 32.78, "z03": 29.71}
    print(f"{'Zone':<8}{'AR(이전)':>10}{'AR(신규)':>10}{'차이':>8}   {'MLR(이전)':>10}{'MLR(신규)':>10}{'차이':>8}")
    for zone_id in ZONE_ID_LIST:
        label = f"z{zone_id:02d}"
        ar_diff = ar_results[label] - prev_ar[label]
        mlr_diff = mlr_results[label] - prev_mlr[label]
        print(f"{label:<8}{prev_ar[label]:>9.2f}%{ar_results[label]:>9.2f}%{ar_diff:>+7.2f}%p   "
              f"{prev_mlr[label]:>9.2f}%{mlr_results[label]:>9.2f}%{mlr_diff:>+7.2f}%p")


if __name__ == "__main__":
    main()
