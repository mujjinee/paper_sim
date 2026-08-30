# -*- coding: utf-8 -*-
"""
논문(Karimi & Kwon, 2022)이 제안한 4가지 모형(기본 AR, 제안 AR, 기본 MLR, 제안 MLR)을
블록9 구간(학습 2012-11-28~2013-09-23, 테스트 2013-09-24~2014-01-01, Zone 1)에 대해
실제로 학습·평가하는 스크립트.

이번에 확정/수정한 부분을 모두 반영:
  - merged_for_simulation_z01.csv 사용 (build_merged_for_simulation.py 에서 이미:
    * MISO 가격 EST -> UTC 보정, 확장된 가격 구간(2012-11-01~2014-04-30) 반영
    * VAR169/VAR178 -> dssrd/dtsr 차분(deaccumulation) 반영
    * Sydney 현지시간 local_date/local_hour 열 반영)
  - 낮 시간대 = Sydney 현지시간 09:00~20:00 (local_hour 9~20)
  - AR lag 설계 = 논문 Eq.(3) "S_{t-h-l}" 그대로: 시간대(h)와 무관하게 "직전 하루" 12시간을
    입력으로 쓰는 direct multi-step (operational_corrected 의 해석을 따름).
  - MLR = pooled 모델, 입력변수는 dSSRD, dTSR, Hour(스칼라 하나, 더미 아님) - 논문이
    backward stepwise로 고른 최종 3변수 그대로.
  - 손실함수 = 절대오차합(LAD), scipy.linprog 으로 직접 풂 (operational_corrected 재사용).
  - 이익함수 = 논문 Eq.(1a) 그대로 3항: DP*commitment + RP*surplus - PC*shortage
    (부족분에 RT가격을 추가로 물리는 항 없음).
  - 오라클 = 논문 Eq.(13)/5.1절 서술 그대로 {0, 실제발전량} 중 선택 (capacity 후보 없음).
    operational_corrected 는 여기에 {0,S,capacity} 3후보를 썼는데, 논문 서술과 다르다고
    판단해 이 스크립트에서는 {0,S} 만 쓰도록 오라클 함수만 수정했다. (다른 MILP 구조,
    surplus/shortage 계수, 이진변수 축소 로직은 전부 operational_corrected 그대로 - 오라클은
    목적함수의 정규화 상수로만 쓰이므로 이 변경이 제약식/이진변수 로직에 영향을 주지 않음)
  - 제안 모형은 논문 Table 3/4 비교 지점과 동일하게 W1=1, W2=20, penalty_rate=0.5 사용.

실행 방법:
    python paper_proposed_block9.py
"""

import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z01.csv")

HOURS_PER_DAY = 12
LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21          # local_hour < 21 -> 9,10,...,20 (12개)
LAG_DAYS = 1                  # AR: "직전 하루"의 12시간을 입력으로 씀 (Eq.3)

TRAIN_START = pd.Timestamp("2012-11-28")
TRAIN_END = pd.Timestamp("2013-09-23")
TEST_START = pd.Timestamp("2013-09-24")
TEST_END = pd.Timestamp("2014-01-01")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
PENALTY_RATE = 0.5
W1, W2 = 1.0, 20.0            # 논문 Table 3/4 비교 지점

PAPER = {
    "기본 모형 AR":     (34.76, 15.04),
    "논문 제안 모형 AR": (34.89, 13.91),
    "기본 모형 MLR":     (21.76, 12.59),
    "논문 제안 모형 MLR": (21.92, 11.91),
}


# ============================================================
# 1. 데이터 준비
# ============================================================
def load_block9():
    df = pd.read_csv(MERGED_FILE)
    df["local_date"] = pd.to_datetime(df["local_date"])
    df = df[(df["local_hour"] >= LOCAL_HOUR_START) & (df["local_hour"] < LOCAL_HOUR_END)].copy()
    df["hour_idx"] = df["local_hour"] - LOCAL_HOUR_START

    history_date = TRAIN_START - pd.Timedelta(days=1)     # AR 학습 첫날의 "직전 하루" (Eq.3 lag=1 용, target에는 포함 안 됨)
    history = df[df["local_date"] == history_date].sort_values("hour_idx").reset_index(drop=True)
    assert len(history) == HOURS_PER_DAY, f"이력 날짜({history_date.date()})에 12시간이 다 없음"

    train = df[(df["local_date"] >= TRAIN_START) & (df["local_date"] <= TRAIN_END)].sort_values(
        ["local_date", "hour_idx"]).reset_index(drop=True)
    test = df[(df["local_date"] >= TEST_START) & (df["local_date"] <= TEST_END)].sort_values(
        ["local_date", "hour_idx"]).reset_index(drop=True)

    assert train["local_date"].nunique() == 300 and len(train) == 3600, "학습 데이터 크기가 300일x12시간이 아님"
    assert test["local_date"].nunique() == 100 and len(test) == 1200, "테스트 데이터 크기가 100일x12시간이 아님"

    return history, train, test


def to_daily_matrix(frame, column):
    ordered = frame.sort_values(["local_date", "hour_idx"])
    matrix = ordered[column].to_numpy(dtype=float).reshape(-1, HOURS_PER_DAY)
    return matrix


# ============================================================
# 2. 절대오차합(LAD) 회귀 - scipy.linprog 직접 풂
# ============================================================
def solve_bounded_lad(X, actual):
    X = np.asarray(X, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n, p = X.shape

    X_sparse = sparse.csr_matrix(X)
    identity = sparse.eye(n, format="csr")
    zeros = sparse.csr_matrix((n, n))
    constraints = sparse.vstack([
        sparse.hstack([X_sparse, -identity]),
        sparse.hstack([-X_sparse, -identity]),
        sparse.hstack([X_sparse, zeros]),
        sparse.hstack([-X_sparse, zeros]),
    ], format="csr")
    limits = np.concatenate([actual, -actual, np.ones(n), np.zeros(n)])
    objective = np.concatenate([np.zeros(p), np.ones(n) / n])
    bounds = [(None, None)] * p + [(0.0, None)] * n

    result = linprog(objective, A_ub=constraints, b_ub=limits, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LAD 회귀 실패: {result.message}")

    coefficients = result.x[:p]
    return coefficients, X @ coefficients


# ============================================================
# 3. AR lag 설계 (논문 Eq.3: 직전 하루의 12시간을 입력으로)
# ============================================================
def build_ar_lag_design(history_daily, train_daily):
    history = np.asarray(history_daily, dtype=float)
    train = np.asarray(train_daily, dtype=float)
    previous_day = np.vstack([history[-1], train[:-1]])          # 각 날짜의 "바로 전날" 12시간
    X = np.column_stack([np.ones(len(train)), previous_day[:, ::-1]])   # intercept + 전날 12시간(역순)
    return X, train.copy()


def fit_baseline_ar(history_daily, train_daily):
    X, targets = build_ar_lag_design(history_daily, train_daily)
    coeffs_by_hour = []
    for hour in range(HOURS_PER_DAY):
        coeffs, _ = solve_bounded_lad(X, targets[:, hour])
        coeffs_by_hour.append(coeffs)
    return np.vstack(coeffs_by_hour)


def predict_ar_rolling(coefficients_by_hour, previous_day_actual, test_actual):
    coefficients = np.asarray(coefficients_by_hour, dtype=float)
    previous_day = np.asarray(previous_day_actual, dtype=float)
    test_actual = np.asarray(test_actual, dtype=float)
    predictions = np.empty_like(test_actual)

    for day in range(len(test_actual)):
        features = np.r_[1.0, previous_day[::-1]]
        predictions[day] = coefficients @ features
        previous_day = test_actual[day]           # 다음 날 예측을 위해, 알고 있는 "실제" 전날 값으로 갱신
    return predictions


# ============================================================
# 4. MLR 설계 (pooled, dSSRD+dTSR+Hour, Hour는 스칼라 하나)
# ============================================================
def build_mlr_design(frame):
    X = np.column_stack([
        np.ones(len(frame)),
        frame["dssrd"].to_numpy(dtype=float),
        frame["dtsr"].to_numpy(dtype=float),
        frame["hour_idx"].to_numpy(dtype=float),
    ])
    return X


def fit_baseline_mlr(X, actual):
    coeffs, fitted = solve_bounded_lad(X, actual)
    return coeffs, fitted


# ============================================================
# 5. 이익/오라클 함수 - 논문 Eq.(1a), Eq.(13) 그대로 (3항, {0,S} 오라클)
# ============================================================
def imbalance_components(actual, commitment):
    actual = np.asarray(actual, dtype=float)
    commitment = np.asarray(commitment, dtype=float)
    surplus = np.maximum(actual - commitment, 0.0)
    shortage = np.maximum(commitment - actual, 0.0)
    return surplus, shortage


def profit_per_observation(actual, commitment, day_ahead_price, real_time_price, penalty_rate=PENALTY_RATE):
    surplus, shortage = imbalance_components(actual, commitment)
    penalty = penalty_rate * np.asarray(day_ahead_price, dtype=float)
    return CAPACITY_MW * DURATION_HOURS * (
        np.asarray(day_ahead_price, dtype=float) * np.asarray(commitment, dtype=float)
        + np.asarray(real_time_price, dtype=float) * surplus
        - penalty * shortage
    )


def oracle_profit_per_observation(actual, day_ahead_price, real_time_price):
    """논문 Eq.(13) / 5.1절: 오라클은 {0, 실제발전량} 중 더 이익나는 쪽을 고른다.
    최종 평가(nRMSE/gap 보고)에서만 쓴다."""
    actual = np.asarray(actual, dtype=float)
    candidates = np.stack([np.zeros_like(actual), actual])       # capacity(1.0) 후보 없음 - 논문 서술 그대로
    profits = np.stack([
        profit_per_observation(actual, x, day_ahead_price, real_time_price) for x in candidates
    ])
    best = profits.argmax(axis=0)
    oracle_profit = np.take_along_axis(profits, best[None, :], axis=0)[0]
    oracle_commitment = np.take_along_axis(candidates, best[None, :], axis=0)[0]
    return oracle_profit, oracle_commitment


def oracle_profit_for_training(actual, day_ahead_price, real_time_price):
    """제안모형 MILP 내부의 정규화 상수(denominator) 계산 전용.

    operational_corrected 원래 방식대로 {0, 실제발전량, 설비최대(1.0)} 3후보를 쓴다.
    이 값은 W1=1,W2=20 목적함수의 스케일을 맞추는 상수일 뿐이라, 평가 지표(오라클
    {0,S})와는 분리해서 써야 한다 - {0,S}로 바꾸면 분모가 작아져 W1이 의도보다
    과도하게 강해지고, 계수가 불안정하게(발산) 학습되는 문제가 실제로 확인됐다.
    """
    actual = np.asarray(actual, dtype=float)
    candidates = np.stack([np.zeros_like(actual), actual, np.ones_like(actual)])
    profits = np.stack([
        profit_per_observation(actual, x, day_ahead_price, real_time_price) for x in candidates
    ])
    best = profits.argmax(axis=0)
    oracle_profit = np.take_along_axis(profits, best[None, :], axis=0)[0]
    return oracle_profit


def normalized_economic_loss(actual, commitment, day_ahead_price, real_time_price):
    oracle, _ = oracle_profit_per_observation(actual, day_ahead_price, real_time_price)
    realized = profit_per_observation(actual, commitment, day_ahead_price, real_time_price)
    return float((oracle - realized).sum() / oracle.sum())


def nrmse_percent(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    rmse = np.sqrt(np.mean((np.asarray(prediction, dtype=float) - actual) ** 2))
    return 100.0 * rmse / actual.mean()


def evaluate(actual, raw_prediction, day_ahead_price, real_time_price):
    projected = np.clip(np.asarray(raw_prediction, dtype=float), 0.0, 1.0)
    realized = profit_per_observation(actual, projected, day_ahead_price, real_time_price)
    oracle, _ = oracle_profit_per_observation(actual, day_ahead_price, real_time_price)
    gap_pct = 100.0 * float((oracle - realized).sum() / oracle.sum())
    return nrmse_percent(actual, projected), gap_pct


# ============================================================
# 6. 제안 모형 MILP (논문 Eq.9/10) - operational_corrected 재사용, 오라클만 교체
# ============================================================
def fit_paper_proposed_milp(X, actual, day_ahead_price, real_time_price, w1=W1, w2=W2):
    X = np.asarray(X, dtype=float)
    actual = np.asarray(actual, dtype=float)
    day_ahead_price = np.asarray(day_ahead_price, dtype=float)
    real_time_price = np.asarray(real_time_price, dtype=float)

    n, p = X.shape
    oracle = oracle_profit_for_training(actual, day_ahead_price, real_time_price)  # 학습 정규화 전용 오라클 ({0,S,capacity})
    denominator = oracle.sum()
    scale = CAPACITY_MW * DURATION_HOURS
    penalty = PENALTY_RATE * day_ahead_price

    surplus_cost = -w1 * scale * real_time_price / denominator + w2 / n
    shortage_cost = w1 * scale * penalty / denominator + w2 / n
    binary_rows = np.flatnonzero(surplus_cost + shortage_cost < 0)
    binary_count = len(binary_rows)

    beta = slice(0, p)
    x_slice = slice(p, p + n)
    y_plus = slice(p + n, p + 2 * n)
    y_minus = slice(p + 2 * n, p + 3 * n)
    z = slice(p + 3 * n, p + 3 * n + binary_count)
    variable_count = p + 3 * n + binary_count

    objective = np.zeros(variable_count)
    objective[x_slice] = -w1 * scale * day_ahead_price / denominator
    objective[y_plus] = surplus_cost
    objective[y_minus] = shortage_cost

    equations = sparse.lil_matrix((2 * n, variable_count))
    equations[:n, beta] = -X
    equations[:n, x_slice] = sparse.eye(n)
    equations[n:, x_slice] = sparse.eye(n)
    equations[n:, y_plus] = sparse.eye(n)
    equations[n:, y_minus] = -sparse.eye(n)
    rhs = np.concatenate([np.zeros(n), actual])
    constraints = [LinearConstraint(equations.tocsr(), rhs, rhs)]

    if binary_count:
        complementarity = sparse.lil_matrix((2 * binary_count, variable_count))
        complementarity[np.arange(binary_count), y_plus.start + binary_rows] = 1
        complementarity[np.arange(binary_count), z.start + np.arange(binary_count)] = 1
        complementarity[binary_count + np.arange(binary_count), y_minus.start + binary_rows] = 1
        complementarity[binary_count + np.arange(binary_count), z.start + np.arange(binary_count)] = -1
        constraints.append(LinearConstraint(
            complementarity.tocsr(),
            np.full(2 * binary_count, -np.inf),
            np.r_[np.ones(binary_count), np.zeros(binary_count)],
        ))

    lower = np.r_[np.full(p, -np.inf), np.zeros(3 * n + binary_count)]
    upper = np.r_[np.full(p, np.inf), np.ones(3 * n + binary_count)]
    integrality = np.zeros(variable_count, dtype=int)
    integrality[z] = 1

    result = milp(c=objective, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=constraints, options={"mip_rel_gap": 1e-9})
    if not result.success:
        raise RuntimeError(f"제안모형 MILP 실패: {result.message}")

    coefficients = result.x[beta]
    return coefficients, X @ coefficients


def fit_proposed_ar(history_daily, train_daily, train_da_daily, train_rt_daily):
    X, targets = build_ar_lag_design(history_daily, train_daily)
    coeffs_by_hour = []
    for hour in range(HOURS_PER_DAY):
        coeffs, _ = fit_paper_proposed_milp(X, targets[:, hour], train_da_daily[:, hour], train_rt_daily[:, hour])
        coeffs_by_hour.append(coeffs)
    return np.vstack(coeffs_by_hour)


def fit_proposed_mlr(X, actual, day_ahead_price, real_time_price):
    coeffs, fitted = fit_paper_proposed_milp(X, actual, day_ahead_price, real_time_price)
    return coeffs, fitted


# ============================================================
# 7. 실행
# ============================================================
def main():
    history, train, test = load_block9()
    print(f"이력(AR용, target 아님): {(TRAIN_START - pd.Timedelta(days=1)).date()}")
    print(f"학습: {TRAIN_START.date()} ~ {TRAIN_END.date()} (300일 x 12시간 = {len(train)}행)")
    print(f"테스트: {TEST_START.date()} ~ {TEST_END.date()} (100일 x 12시간 = {len(test)}행)\n")

    history_daily = to_daily_matrix(history, "solar_power")   # (1, 12) - AR lag=1일용 이력
    train_solar = to_daily_matrix(train, "solar_power")
    test_solar = to_daily_matrix(test, "solar_power")
    train_da = to_daily_matrix(train, "da_price")
    train_rt = to_daily_matrix(train, "rt_price")
    test_da_flat = test.sort_values(["local_date", "hour_idx"])["da_price"].to_numpy()
    test_rt_flat = test.sort_values(["local_date", "hour_idx"])["rt_price"].to_numpy()
    test_actual_flat = test.sort_values(["local_date", "hour_idx"])["solar_power"].to_numpy()

    results = {}

    # --- 기본 모형 AR ---
    print("[1/4] 기본 모형 AR 학습 중...")
    ar_coeffs = fit_baseline_ar(history_daily, train_solar)
    ar_pred = predict_ar_rolling(ar_coeffs, train_solar[-1], test_solar)
    ar_pred_flat = np.clip(ar_pred, 0, 1).flatten()
    results["기본 모형 AR"] = evaluate(test_actual_flat, ar_pred_flat, test_da_flat, test_rt_flat)

    # --- 논문 제안 모형 AR ---
    print("[2/4] 논문 제안 모형 AR 학습 중 (MILP)...")
    ar_prop_coeffs = fit_proposed_ar(history_daily, train_solar, train_da, train_rt)
    ar_prop_pred = predict_ar_rolling(ar_prop_coeffs, train_solar[-1], test_solar)
    ar_prop_pred_flat = np.clip(ar_prop_pred, 0, 1).flatten()
    results["논문 제안 모형 AR"] = evaluate(test_actual_flat, ar_prop_pred_flat, test_da_flat, test_rt_flat)

    # --- 기본 모형 MLR ---
    print("[3/4] 기본 모형 MLR 학습 중...")
    X_train = build_mlr_design(train)
    X_test = build_mlr_design(test)
    y_train = train["solar_power"].to_numpy()
    mlr_coeffs, _ = fit_baseline_mlr(X_train, y_train)
    mlr_pred = np.clip(X_test @ mlr_coeffs, 0, 1)
    results["기본 모형 MLR"] = evaluate(test_actual_flat, mlr_pred, test_da_flat, test_rt_flat)

    # --- 논문 제안 모형 MLR ---
    print("[4/4] 논문 제안 모형 MLR 학습 중 (MILP)...")
    train_da_flat = train["da_price"].to_numpy()
    train_rt_flat = train["rt_price"].to_numpy()
    mlr_prop_coeffs, _ = fit_proposed_mlr(X_train, y_train, train_da_flat, train_rt_flat)
    mlr_prop_pred = np.clip(X_test @ mlr_prop_coeffs, 0, 1)
    results["논문 제안 모형 MLR"] = evaluate(test_actual_flat, mlr_prop_pred, test_da_flat, test_rt_flat)

    # --- 결과표 ---
    print("\n| 모델 | 논문 nRMSE | 이번 구현 nRMSE | 논문 optimality gap | 이번 구현 optimality gap | Δ nRMSE | Δ optimality gap |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in ["기본 모형 AR", "논문 제안 모형 AR", "기본 모형 MLR", "논문 제안 모형 MLR"]:
        our_nrmse, our_gap = results[name]
        paper_nrmse, paper_gap = PAPER[name]
        print(f"| {name} | {paper_nrmse:.2f}% | {our_nrmse:.6f}% | {paper_gap:.2f}% | {our_gap:.6f}% "
              f"| {our_nrmse - paper_nrmse:+.6f}%p | {our_gap - paper_gap:+.6f}%p |")


if __name__ == "__main__":
    main()
