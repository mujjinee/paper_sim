"""
MISO 가격(EST) <-> GEFCom2014 태양광/기상(UTC) 타임존 정렬 보정이
optimality gap 계산에 실제로 얼마나 영향을 주는지 확인하는 스크립트.

같은 AR 예측(forecast)에 대해:
  - OLD: merged_for_simulation.csv (가격 timestamp 보정 전, EST를 UTC인 것처럼 취급)
  - NEW: merged_for_simulation_z01.csv (가격을 EST->UTC로 +5시간 보정한 뒤 병합)
두 가격 정렬로 optimality gap(penalty_rate=50%)을 각각 계산해 비교한다.

이익함수는 논문 Section 4.2 Eq.(1a)를 그대로 따른다: DP*x + RP*y' - PC*y''
(부족분 y''에 real-time price를 추가로 물리는 항은 논문에 없음).
오라클은 논문 5.1절 서술 그대로 "DA>RT면 실제발전량만큼, 아니면 0"
(Eq. 13의 max{DP,RP}*S_t 와 동일 — capacity까지 미는 옵션은 논문에 없음).

Usage:
    python check_price_timezone_impact.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
OLD_PATH = os.path.join(BASE, "merged_for_simulation.csv")        # 보정 전 (zone1, EST 취급 안 함)
NEW_PATH = os.path.join(BASE, "merged_for_simulation_z01.csv")    # 보정 후 (zone1, EST->UTC)

N_HOURS = 12
N_LAGS = 12
N_TRAIN_DAYS = 300
N_TEST_DAYS = 100
PENALTY_RATE = 0.5


def load_daily_arrays(data_path):
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].dt.hour.between(0, N_HOURS - 1)].copy()
    df["date"] = df["timestamp"].dt.date
    df["hour_idx"] = df["timestamp"].dt.hour
    df = df.dropna(subset=["solar_power", "da_price", "rt_price"])

    hour_counts = df.groupby("date")["hour_idx"].nunique()
    complete_dates = set(hour_counts[hour_counts == N_HOURS].index)
    df = df[df["date"].isin(complete_dates)]

    dates = sorted(df["date"].unique())
    n_days = len(dates)
    solar = np.zeros((n_days, N_HOURS))
    da = np.zeros((n_days, N_HOURS))
    rt = np.zeros((n_days, N_HOURS))
    for i, d in enumerate(dates):
        day = df[df["date"] == d].sort_values("hour_idx")
        solar[i] = day["solar_power"].values
        da[i] = day["da_price"].values
        rt[i] = day["rt_price"].values
    return dates, solar, da, rt


class ARModel:
    def __init__(self, n_lags=N_LAGS):
        self.n_lags = n_lags
        self.models = {}

    def fit(self, daily_solar):
        n_days, n_hours = daily_solar.shape
        for h in range(n_hours):
            X, y = [], []
            for day in range(self.n_lags, n_days):
                X.append(daily_solar[day - self.n_lags:day, h])
                y.append(daily_solar[day, h])
            model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
            model.fit(np.array(X), np.array(y))
            self.models[h] = model

    def predict_next_day(self, past_n_days):
        n_hours = past_n_days.shape[1]
        forecast = np.zeros(n_hours)
        for h in range(n_hours):
            x = past_n_days[:, h].reshape(1, -1)
            forecast[h] = np.clip(self.models[h].predict(x)[0], 0, 1)
        return forecast


def unit_commitment_profit(commitment, actual, da_price, rt_price, penalty_cost):
    mismatch = actual - commitment
    surplus = np.maximum(mismatch, 0)
    shortage = np.maximum(-mismatch, 0)
    # 논문 Eq. (1a): DP*x + RP*y' - PC*y''  (부족분에 RT가격을 추가로 물리는 항은 없음)
    return np.sum(
        da_price * commitment + rt_price * surplus - penalty_cost * shortage
    )


def oracle_profit(actual, da_price, rt_price, penalty_cost):
    optimal_commit = np.where(da_price > rt_price, actual, 0)
    return unit_commitment_profit(optimal_commit, actual, da_price, rt_price, penalty_cost)


def optimality_gap_pct(commitment, actual, da_price, rt_price, penalty_rate):
    pc = penalty_rate * da_price
    o_p = oracle_profit(actual, da_price, rt_price, pc)
    a_p = unit_commitment_profit(commitment, actual, da_price, rt_price, pc)
    return 100 * (o_p - a_p) / o_p


def nrmse(actual, predicted):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return 100 * rmse / np.mean(actual)


def main():
    dates_old, solar_old, da_old, rt_old = load_daily_arrays(OLD_PATH)
    dates_new, solar_new, da_new, rt_new = load_daily_arrays(NEW_PATH)

    need = N_TRAIN_DAYS + N_TEST_DAYS
    solar_old, da_old, rt_old, dates_old = (
        solar_old[-need:], da_old[-need:], rt_old[-need:], dates_old[-need:]
    )
    solar_new, da_new, rt_new, dates_new = (
        solar_new[-need:], da_new[-need:], rt_new[-need:], dates_new[-need:]
    )

    print(f"OLD 테스트 구간: {dates_old[N_TRAIN_DAYS]} ~ {dates_old[-1]}")
    print(f"NEW 테스트 구간: {dates_new[N_TRAIN_DAYS]} ~ {dates_new[-1]}")
    assert dates_old[N_TRAIN_DAYS:] == dates_new[N_TRAIN_DAYS:], "테스트 날짜가 다름!"
    assert np.allclose(solar_old, solar_new), "solar 값이 다름! (원래 같아야 함)"

    train_solar = solar_new[:N_TRAIN_DAYS]
    test_solar = solar_new[N_TRAIN_DAYS:]

    # AR 학습 (solar만 쓰므로 old/new 동일 -> 한 번만)
    ar = ARModel(n_lags=N_LAGS)
    ar.fit(train_solar)

    history = train_solar.copy()
    forecasts = np.zeros_like(test_solar)
    for day in range(N_TEST_DAYS):
        past = history[-N_LAGS:]
        forecasts[day] = ar.predict_next_day(past)
        history = np.vstack([history, test_solar[day]])

    ar_nrmse = nrmse(test_solar.flatten(), forecasts.flatten())

    test_da_old, test_rt_old = da_old[N_TRAIN_DAYS:], rt_old[N_TRAIN_DAYS:]
    test_da_new, test_rt_new = da_new[N_TRAIN_DAYS:], rt_new[N_TRAIN_DAYS:]

    gap_old = optimality_gap_pct(
        forecasts.flatten(), test_solar.flatten(),
        test_da_old.flatten(), test_rt_old.flatten(), PENALTY_RATE,
    )
    gap_new = optimality_gap_pct(
        forecasts.flatten(), test_solar.flatten(),
        test_da_new.flatten(), test_rt_new.flatten(), PENALTY_RATE,
    )

    print(f"\nAR nRMSE (old/new 동일): {ar_nrmse:.2f}%")
    print("\n=== Optimality Gap (penalty_rate=50%) : 가격 타임존 보정 전/후 비교 ===")
    print(f"OLD (가격 timestamp 보정 없음, EST를 UTC처럼 취급) : {gap_old:.2f}%")
    print(f"NEW (가격 EST -> UTC +5시간 보정)                  : {gap_new:.2f}%")
    print(f"차이                                              : {gap_new - gap_old:+.2f}%p")
    print(f"\n(참고) 논문 Table 3 AR: nRMSE=34.76%, Optimality Gap=15.04%")


if __name__ == "__main__":
    main()
