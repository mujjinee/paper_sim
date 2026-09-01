"""Corrected AR and MLR baseline experiment for the APEN paper.

This script deliberately leaves ``run_simulation.py`` unchanged.  It corrects
two independently identified implementation defects in the baseline models:

* AR reads the *latest* n_lags daily observations at every forecast origin.
* MLR is one pooled regression over all day/hour observations, so ``hour`` is
  a varying explanatory variable rather than a constant in a per-hour model.

The script uses the project's currently available Jan--Apr 2014 merged data:
Jan--Mar train and Apr test.  It is therefore a diagnostic replication, not a
numerical reproduction of the paper's 300-day / 100-day experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "merged_for_simulation.csv"
OUTPUT_DIR = BASE / "results" / "simulation_output"
N_HOURS = 12
AR_LAGS_DAYS = 12
PENALTY_RATE = 0.5
MLR_FEATURES = ["ssrd", "tsr", "hour"]


def load_data() -> pd.DataFrame:
    """Load only the twelve project-defined daytime UTC hours, with checks."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [column.strip() for column in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.loc[df["timestamp"].dt.hour.between(0, N_HOURS - 1)].copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df = df.rename(columns={"solar_power": "solar_norm"})

    required = ["solar_norm", "da_price", "rt_price", *MLR_FEATURES]
    df = df.dropna(subset=required).sort_values(["date", "hour"])

    counts = df.groupby("date").size()
    incomplete = counts[counts != N_HOURS]
    if not incomplete.empty:
        raise ValueError(
            "Every retained day must have exactly 12 hourly rows; "
            f"invalid dates: {incomplete.to_dict()}"
        )
    if df.duplicated(["date", "hour"]).any():
        raise ValueError("A date/hour observation is duplicated.")
    return df.reset_index(drop=True)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = df["timestamp"].dt.month
    train = df.loc[months.isin([1, 2, 3])].copy()
    test = df.loc[months.eq(4)].copy()
    if train.empty or test.empty:
        raise ValueError("Configured Jan--Mar train / Apr test split is empty.")
    return train, test


def daily_matrix(df: pd.DataFrame, value_column: str) -> tuple[list, np.ndarray]:
    pivot = df.pivot(index="date", columns="hour", values=value_column)
    pivot = pivot.reindex(columns=range(N_HOURS))
    if pivot.isna().any().any():
        raise ValueError(f"Missing values while forming daily {value_column} matrix.")
    return list(pivot.index), pivot.to_numpy(dtype=float)


class SeasonalDailyAR:
    """One direct regression for each hour, using previous daily same-hour lags."""

    def __init__(self, n_lags: int = AR_LAGS_DAYS) -> None:
        self.n_lags = n_lags
        self.models: dict[int, LinearRegression] = {}

    def fit(self, train_daily: np.ndarray) -> "SeasonalDailyAR":
        if len(train_daily) <= self.n_lags:
            raise ValueError("Not enough training days for configured AR lags.")
        for hour in range(N_HOURS):
            x_rows = [
                train_daily[day - self.n_lags : day, hour][::-1]
                for day in range(self.n_lags, len(train_daily))
            ]
            y = train_daily[self.n_lags :, hour]
            self.models[hour] = LinearRegression().fit(np.asarray(x_rows), y)
        return self

    def predict_next_day(self, history_daily: np.ndarray) -> np.ndarray:
        if len(history_daily) < self.n_lags:
            raise ValueError("AR history is shorter than n_lags.")
        # Crucial correction: take the LAST n_lags days, not rows 0..n_lags-1.
        recent = history_daily[-self.n_lags :]
        forecast = np.empty(N_HOURS, dtype=float)
        for hour, model in self.models.items():
            forecast[hour] = model.predict(recent[:, hour][::-1].reshape(1, -1))[0]
        return np.clip(forecast, 0.0, 1.0)


class PooledMLR:
    """Paper-style pooled MLR: SSRD, TSR, and Hour vary across all rows."""

    def __init__(self, features: list[str] = MLR_FEATURES) -> None:
        self.features = features
        self.model = LinearRegression()

    def fit(self, train: pd.DataFrame) -> "PooledMLR":
        self.model.fit(train[self.features], train["solar_norm"])
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(rows[self.features]), 0.0, 1.0)


def nrmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    mean_actual = float(np.mean(actual))
    if mean_actual <= 0:
        raise ValueError("nRMSE denominator must be positive.")
    return 100 * float(np.sqrt(np.mean((actual - forecast) ** 2))) / mean_actual


def profit(commitment, actual, da_price, rt_price, penalty_cost) -> float:
    mismatch = actual - commitment
    surplus = np.maximum(mismatch, 0.0)
    shortage = np.maximum(-mismatch, 0.0)
    return float(np.sum(da_price * commitment + rt_price * surplus - rt_price * shortage - penalty_cost * shortage))


def optimality_gap(commitment, actual, da_price, rt_price, penalty_rate: float) -> float:
    penalty_cost = penalty_rate * da_price
    oracle_commitment = np.where(da_price > rt_price, actual, 0.0)
    oracle = profit(oracle_commitment, actual, da_price, rt_price, penalty_cost)
    achieved = profit(commitment, actual, da_price, rt_price, penalty_cost)
    if abs(oracle) < 1e-10:
        return 0.0
    return 100 * (oracle - achieved) / oracle


def run() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    train, test = split_train_test(df)
    _, train_daily = daily_matrix(train, "solar_norm")
    test_dates, test_daily = daily_matrix(test, "solar_norm")

    ar = SeasonalDailyAR().fit(train_daily)
    ar_forecast_daily = np.empty_like(test_daily)
    history = train_daily.copy()
    for day in range(len(test_daily)):
        ar_forecast_daily[day] = ar.predict_next_day(history)
        # At the next day-ahead forecast origin, the previous day's actual output
        # is observed and may be used as AR history; coefficients stay fixed.
        history = np.vstack([history, test_daily[day]])

    mlr = PooledMLR().fit(train)
    test_rows = test.sort_values(["date", "hour"]).reset_index(drop=True)
    mlr_forecast_flat = mlr.predict(test_rows)
    mlr_forecast_daily = mlr_forecast_flat.reshape(len(test_dates), N_HOURS)

    actual_flat = test_daily.ravel()
    da_flat = test_rows["da_price"].to_numpy(dtype=float)
    rt_flat = test_rows["rt_price"].to_numpy(dtype=float)
    rows = []
    for name, forecast in (("AR_corrected", ar_forecast_daily), ("MLR_pooled_ssrd_tsr_hour", mlr_forecast_daily)):
        flat = forecast.ravel()
        rows.append(
            {
                "model": name,
                "train_days": len(train_daily),
                "test_days": len(test_daily),
                "nRMSE_overall_pct": nrmse(actual_flat, flat),
                "optimality_gap_pct_penalty_50": optimality_gap(
                    flat, actual_flat, da_flat, rt_flat, PENALTY_RATE
                ),
            }
        )

    result = pd.DataFrame(rows)
    prediction_rows = []
    for day_index, date in enumerate(test_dates):
        for hour in range(N_HOURS):
            prediction_rows.append(
                {
                    "date": date,
                    "hour": hour,
                    "actual_solar_norm": test_daily[day_index, hour],
                    "ar_corrected": ar_forecast_daily[day_index, hour],
                    "mlr_pooled_ssrd_tsr_hour": mlr_forecast_daily[day_index, hour],
                }
            )
    result.to_csv(OUTPUT_DIR / "baseline_corrected_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "baseline_corrected_predictions.csv", index=False)

    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results: {OUTPUT_DIR / 'baseline_corrected_results.csv'}")
    print(f"Predictions: {OUTPUT_DIR / 'baseline_corrected_predictions.csv'}")
    return result


if __name__ == "__main__":
    run()
