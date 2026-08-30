"""
AR (Auto-Regressive) 모델만 독립적으로 구현/실행하는 스크립트.
AR 자체의 예측 성능(nRMSE)만 확인하는 것이 목적이며, optimality gap은 다루지 않는다.

논문(Karimi & Kwon, 2022, Applied Energy 326: 119929)
Section 4.3.1, Eq. (3): direct multi-step AR.
  하루 중 태양광 발전이 있는 12개 시간대(h=0..11) 각각에 대해
  별도의 회귀 모델을 학습 (과거 N_LAGS일의 같은 시간대 값을 입력으로 사용).

손실함수: 논문 4.3.1절 "본 연구는 AR의 손실함수로 절대오차합을 사용한다" (Eq. 5, LAD)
  에 맞춰 sklearn QuantileRegressor(quantile=0.5, alpha=0)로 학습.
  (quantile=0.5, L1 정규화 없음 => 잔차 절대값 합을 최소화하는 median regression = LAD)

데이터: merged_for_simulation_z01/z02/z03.csv (GEFCom2014 Solar Zone 1/2/3,
2013-03-26 ~ 2014-04-30, 401일) 각각에 대해 동일한 AR 모델을 학습/평가한다.

학습/테스트 분할: 논문 5.1절 "수치 실험에 사용된 훈련·테스트 데이터 크기는
각각 300일, 100일" 에 맞춰 마지막 400일 중 300일 학습 / 100일 테스트로 사용.

평가: nRMSE (Eq. 11-12) 만 확인. 논문 Table 3의 AR 결과 nRMSE=34.76% 와 비교.

Usage:
    python ar_only.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
ZONE_FILES = {
    "z01": os.path.join(BASE, "merged_for_simulation_z01.csv"),
    "z02": os.path.join(BASE, "merged_for_simulation_z02.csv"),
    "z03": os.path.join(BASE, "merged_for_simulation_z03.csv"),
}

N_HOURS = 12         # 태양광 발전이 있는 낮 시간대 개수 (UTC 00:00~11:00)
N_LAGS = 12           # AR lag: 과거 며칠의 같은 시간대 값을 사용할지 (논문: 12)
N_TRAIN_DAYS = 300    # 논문 5.1절: 훈련 300일
N_TEST_DAYS = 100     # 논문 5.1절: 테스트 100일

PAPER_AR_NRMSE = 34.76   # 논문 Table 3


# ============================================================
# 1. 데이터 로딩 -> (n_days, N_HOURS) 배열
# ============================================================
def load_daily_arrays(data_path):
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 태양광 발전이 있는 낮 시간대만 사용
    df = df[df["timestamp"].dt.hour.between(0, N_HOURS - 1)].copy()
    df["date"] = df["timestamp"].dt.date
    df["hour_idx"] = df["timestamp"].dt.hour
    df = df.dropna(subset=["solar_power"])

    # 12시간이 모두 갖춰진 날짜만 사용 (가격 커버리지 보정 등으로 일부 시간이
    # 빠진 첫날/마지막날 같은 부분일은 제외)
    hour_counts = df.groupby("date")["hour_idx"].nunique()
    complete_dates = set(hour_counts[hour_counts == N_HOURS].index)
    dropped = sorted(set(df["date"].unique()) - complete_dates)
    if dropped:
        print(f"  (불완전한 날짜 {len(dropped)}개 제외: {dropped})")
    df = df[df["date"].isin(complete_dates)]

    dates = sorted(df["date"].unique())
    n_days = len(dates)
    solar = np.zeros((n_days, N_HOURS))

    for i, d in enumerate(dates):
        day = df[df["date"] == d].sort_values("hour_idx")
        solar[i] = day["solar_power"].values

    return dates, solar


# ============================================================
# 2. AR 모델 - direct multi-step (논문 Section 4.3.1, Eq. 3)
#    시간대별(0~11) 독립 회귀, 손실함수는 절대오차합(LAD, 논문 Eq. 5)
#    Ŝ_t^h = α^h + Σ_l β_l^h · S_{t-l}^h   (l = 1..N_LAGS)
# ============================================================
class ARModel:
    def __init__(self, n_lags=N_LAGS):
        self.n_lags = n_lags
        self.models = {}  # hour_idx -> QuantileRegressor(quantile=0.5) = LAD

    def fit(self, daily_solar):
        """daily_solar: (n_days, N_HOURS) 학습용 배열."""
        n_days, n_hours = daily_solar.shape
        for h in range(n_hours):
            X, y = [], []
            for day in range(self.n_lags, n_days):
                # 과거 n_lags일의 같은 시간대 h 값 (오래된 -> 최근 순)
                X.append(daily_solar[day - self.n_lags:day, h])
                y.append(daily_solar[day, h])
            # quantile=0.5, alpha=0(정규화 없음) => 절대오차합(LAD) 최소화
            model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
            model.fit(np.array(X), np.array(y))
            self.models[h] = model

    def predict_next_day(self, past_n_days):
        """past_n_days: (n_lags, N_HOURS) 가장 최근 n_lags일 실제값.

        반환: (N_HOURS,) 다음 날 예측값 (0~1 클립).
        """
        n_hours = past_n_days.shape[1]
        forecast = np.zeros(n_hours)
        for h in range(n_hours):
            x = past_n_days[:, h].reshape(1, -1)
            pred = self.models[h].predict(x)[0]
            forecast[h] = np.clip(pred, 0, 1)
        return forecast


# ============================================================
# 3. 평가 함수 - nRMSE 만 (논문 Eq. 11-12)
# ============================================================
def nrmse(actual, predicted):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return 100 * rmse / np.mean(actual)


# ============================================================
# 4. 실행: 300일 학습 / 100일 테스트, rolling one-day-ahead 예측
# ============================================================
def run_ar(data_path, label):
    dates, solar = load_daily_arrays(data_path)
    n_days = len(dates)
    print(f"\n[{label}] 전체 데이터: {n_days}일 ({dates[0]} ~ {dates[-1]})")

    need = N_TRAIN_DAYS + N_TEST_DAYS
    if n_days < need:
        raise ValueError(f"데이터가 {need}일보다 적습니다 (현재 {n_days}일)")

    # 마지막 400일(=300+100)만 사용
    start = n_days - need
    solar, dates = solar[start:], dates[start:]

    train_solar = solar[:N_TRAIN_DAYS]
    test_solar = solar[N_TRAIN_DAYS:]
    print(f"[{label}] 학습: {dates[0]} ~ {dates[N_TRAIN_DAYS - 1]} ({N_TRAIN_DAYS}일)")
    print(f"[{label}] 테스트: {dates[N_TRAIN_DAYS]} ~ {dates[-1]} ({N_TEST_DAYS}일)")

    # --- AR 모델 학습 (LAD 손실) ---
    ar = ARModel(n_lags=N_LAGS)
    ar.fit(train_solar)

    # --- Rolling one-day-ahead 예측: 매 테스트일마다 직전 실제 N_LAGS일 사용 ---
    history = train_solar.copy()
    forecasts = np.zeros_like(test_solar)
    for day in range(N_TEST_DAYS):
        past = history[-N_LAGS:]
        forecasts[day] = ar.predict_next_day(past)
        history = np.vstack([history, test_solar[day]])  # 실제값으로 이력 업데이트

    # --- 평가: nRMSE 만 ---
    ar_nrmse = nrmse(test_solar.flatten(), forecasts.flatten())
    print(f"[{label}] nRMSE = {ar_nrmse:.2f}%")
    return ar_nrmse


def main():
    results = {}
    for label, path in ZONE_FILES.items():
        results[label] = run_ar(path, label)

    print("\n=== AR 모델 (Eq. 3, direct multi-step, LAD 손실) 결과 - Zone별 비교 ===")
    print(f"{'Zone':<8}{'nRMSE':>10}{'논문 대비':>12}")
    for label, val in results.items():
        print(f"{label:<8}{val:>9.2f}%{val - PAPER_AR_NRMSE:>+11.2f}%p")
    print(f"{'논문':<8}{PAPER_AR_NRMSE:>9.2f}%")


if __name__ == "__main__":
    main()
