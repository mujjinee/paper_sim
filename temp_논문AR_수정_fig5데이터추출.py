
# -*- coding: utf-8 -*-

# =====================================================================
# temp_논문AR_수정_fig5데이터추출.py
#
# 목적: 논문AR_수정.py 와 완전히 같은 이익함수/오라클 정의
#       (shortage cost = RP + PC, Eq.1a 수정판)를 그대로 쓰되,
#       W1/W2 스윕(Fig.3용) 대신 W1=W2=1 고정하고 PENALTY_RATE를
#       0~100% 스윕(Fig.5용)해서 nRMSE·gap 데이터를 뽑는다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
# =====================================================================

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
import csv


# =====================================================================
# 0. 설정값
# =====================================================================
BASE_DIR = os.getcwd()
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")

HOURS_PER_DAY = 12
LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21

TRAIN_START = pd.Timestamp("2013-08-25")
TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23")
TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0

W1_FIXED = 1.0                    # Fig.5 는 W1=W2=1 고정
W2_FIXED = 1.0

PENALTY_RATES = [round(0.1 * k, 1) for k in range(11)]   # 0.0, 0.1, ..., 1.0


# =====================================================================
# 1. 데이터 읽기 + 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START


# =====================================================================
# 2. 이력/학습/테스트 구간으로 자르기
# =====================================================================
is_history = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history].copy().sort_values("hour_idx")

is_train = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])

is_test = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])

print(f"이력: {len(history_rows)}행, 학습: {len(train_rows)}행, 테스트: {len(test_rows)}행")


# =====================================================================
# 3. (날짜 x 12시간) 배열로 만들기
# =====================================================================
history_solar = np.zeros((1, HOURS_PER_DAY))
for _, row in history_rows.iterrows():
    history_solar[0, row["hour_idx"]] = row["solar_power"]

train_dates_sorted = sorted(train_rows["local_date"].unique())
n_train_days = len(train_dates_sorted)

train_solar = np.zeros((n_train_days, HOURS_PER_DAY))
train_da_price = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt_price = np.zeros((n_train_days, HOURS_PER_DAY))
for _, row in train_rows.iterrows():
    d = (row["local_date"] - TRAIN_START).days
    h = row["hour_idx"]
    train_solar[d, h] = row["solar_power"]
    train_da_price[d, h] = row["da_price"]
    train_rt_price[d, h] = row["rt_price"]

test_dates_sorted = sorted(test_rows["local_date"].unique())
n_test_days = len(test_dates_sorted)

test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))
for _, row in test_rows.iterrows():
    d = (row["local_date"] - TEST_START).days
    h = row["hour_idx"]
    test_solar[d, h] = row["solar_power"]
    test_da_price[d, h] = row["da_price"]
    test_rt_price[d, h] = row["rt_price"]


# =====================================================================
# 4. AR 학습용 입력(X) - 논문 Eq.(3): 직전 하루의 12시간
# =====================================================================
history_and_train_solar = np.vstack([history_solar, train_solar])

n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1))
ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train_solar[d][::-1]

ar_design_matrix = np.hstack([ar_intercept, ar_lag])
n_features = ar_design_matrix.shape[1]

scale = CAPACITY_MW * DURATION_HOURS
actual_flat = test_solar.flatten()
da_flat = test_da_price.flatten()
rt_flat = test_rt_price.flatten()
n_test_obs = len(actual_flat)


# =====================================================================
# 5. 기본 AR(baseline) - 벌금율과 무관하게 계수는 한 번만 구하면 됨
# =====================================================================
ar_coefficients = np.zeros((HOURS_PER_DAY, n_features))
for h in range(HOURS_PER_DAY):
    y_h = train_solar[:, h]
    X_sp = sparse.csr_matrix(ar_design_matrix)
    I = sparse.eye(n_ar_rows, format="csr")
    A = sparse.vstack([sparse.hstack([X_sp, -I]), sparse.hstack([-X_sp, -I])], format="csr")
    b_lim = np.concatenate([y_h, -y_h])
    c_obj = np.concatenate([np.zeros(n_features), np.ones(n_ar_rows) / n_ar_rows])
    vb = [(None, None)] * n_features + [(0.0, None)] * n_ar_rows
    res = linprog(c_obj, A_ub=A, b_ub=b_lim, bounds=vb, method="highs")
    ar_coefficients[h] = res.x[:n_features]

test_forecast_ar = np.zeros((n_test_days, HOURS_PER_DAY))
prev_day = train_solar[-1]
for d in range(n_test_days):
    feat = np.concatenate([[1.0], prev_day[::-1]])
    for h in range(HOURS_PER_DAY):
        test_forecast_ar[d, h] = np.clip(np.dot(ar_coefficients[h], feat), 0, 1)
    prev_day = test_solar[d]
ar_pred_flat = test_forecast_ar.flatten()

ar_rmse = np.sqrt(np.mean((actual_flat - ar_pred_flat) ** 2))
ar_nrmse = 100 * ar_rmse / np.mean(actual_flat)


# =====================================================================
# 6. AR: 벌금율별로 gap만 재계산 (예측 자체는 고정)
#    - 이익함수(수정판): DA*x + RP*y+ - RP*y- - PC*y-  (shortage cost=RP+PC)
# =====================================================================
fig5_rows = []   # (model, penalty_rate, nRMSE, gap) 담을 리스트

for pr in PENALTY_RATES:
    sum_realized = 0.0; sum_oracle = 0.0
    for i in range(n_test_obs):
        a = actual_flat[i]; x = ar_pred_flat[i]; dp = da_flat[i]; rp = rt_flat[i]
        pc = pr * dp
        mismatch = a - x; surplus = max(mismatch, 0); shortage = max(-mismatch, 0)
        realized = scale * (dp * x + rp * surplus - rp * shortage - pc * shortage)   # 수정판 이익함수
        sum_realized += realized
        oracle = max(scale * rp * a, scale * dp * a)
        sum_oracle += oracle
    gap = 100 * (sum_oracle - sum_realized) / sum_oracle
    fig5_rows.append(("AR", pr, ar_nrmse, gap))
    print(f"[AR] penalty={pr:.0%}  nRMSE={ar_nrmse:.3f}%  gap={gap:.3f}%")


# =====================================================================
# 7. 제안모형(W1=W2=1): 벌금율별로 재학습 (MILP)
#    - 학습용 오라클도 수정판 이익함수(shortage cost=RP+PC)로 계산
# =====================================================================
for pr in PENALTY_RATES:

    coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))

    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]
        da_h = train_da_price[:, hour]
        rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows

        # ---- 학습용 오라클: {0, 실제발전량, 설비최대(1.0)} 3후보 (수정판 이익함수) ----
        oracle_profit_train = np.zeros(n_obs)
        for i in range(n_obs):
            a = y_h[i]; dp = da_h[i]; rp = rt_h[i]
            pc = pr * dp
            p0 = scale * rp * a
            pa = scale * dp * a
            surplus_f = max(a - 1.0, 0); shortage_f = max(1.0 - a, 0)
            p1 = scale * (dp * 1.0 + rp * surplus_f - (rp + pc) * shortage_f)   # 수정판: shortage cost=RP+PC
            oracle_profit_train[i] = max(p0, pa, p1)
        training_denom = oracle_profit_train.sum()

        surplus_cost = np.zeros(n_obs)
        shortage_cost = np.zeros(n_obs)
        for i in range(n_obs):
            pc = pr * da_h[i]
            surplus_cost[i] = (-W1_FIXED * scale * rt_h[i] / training_denom) + (W2_FIXED / n_obs)
            shortage_cost[i] = (W1_FIXED * scale * (rt_h[i] + pc) / training_denom) + (W2_FIXED / n_obs)   # 수정판

        binary_row_list = [i for i in range(n_obs) if surplus_cost[i] + shortage_cost[i] < 0]
        binary_rows_arr = np.array(binary_row_list, dtype=int)
        n_binary = len(binary_rows_arr)

        beta_s = 0; x_s = n_features; yp_s = n_features + n_obs
        ym_s = n_features + 2 * n_obs; z_s = n_features + 3 * n_obs
        n_variables = n_features + 3 * n_obs + n_binary

        objective = np.zeros(n_variables)
        for i in range(n_obs):
            objective[x_s + i] = -W1_FIXED * scale * da_h[i] / training_denom
            objective[yp_s + i] = surplus_cost[i]
            objective[ym_s + i] = shortage_cost[i]

        X_sparse = sparse.csr_matrix(ar_design_matrix)
        identity_n = sparse.eye(n_obs, format="csr")

        eq_a = sparse.lil_matrix((n_obs, n_variables))
        eq_a[:, beta_s:beta_s + n_features] = -X_sparse
        eq_a[:, x_s:x_s + n_obs] = identity_n

        eq_b = sparse.lil_matrix((n_obs, n_variables))
        eq_b[:, x_s:x_s + n_obs] = identity_n
        eq_b[:, yp_s:yp_s + n_obs] = identity_n
        eq_b[:, ym_s:ym_s + n_obs] = -identity_n

        all_eq = sparse.vstack([eq_a, eq_b], format="csr")
        eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
        eq_con = LinearConstraint(all_eq, eq_rhs, eq_rhs)
        all_con = [eq_con]

        if n_binary > 0:
            comp_mat = sparse.lil_matrix((2 * n_binary, n_variables))
            for k in range(n_binary):
                row = binary_rows_arr[k]
                comp_mat[k, yp_s + row] = 1.0; comp_mat[k, z_s + k] = 1.0
                comp_mat[n_binary + k, ym_s + row] = 1.0; comp_mat[n_binary + k, z_s + k] = -1.0
            comp_upper = np.concatenate([np.ones(n_binary), np.zeros(n_binary)])
            comp_lower = np.full(2 * n_binary, -np.inf)
            all_con.append(LinearConstraint(comp_mat.tocsr(), comp_lower, comp_upper))

        lb = np.concatenate([np.full(n_features, -np.inf), np.zeros(3 * n_obs + n_binary)])
        ub = np.concatenate([np.full(n_features, np.inf), np.ones(3 * n_obs + n_binary)])
        bounds = Bounds(lb, ub)

        integrality = np.zeros(n_variables, dtype=int)
        for k in range(n_binary):
            integrality[z_s + k] = 1

        milp_result = milp(c=objective, integrality=integrality, bounds=bounds,
                            constraints=all_con, options={"mip_rel_gap": 1e-9})
        coefficients_by_hour[hour] = milp_result.x[beta_s:beta_s + n_features]

    # ---- 테스트 예측 ----
    test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))
    prev_day = train_solar[-1]
    for d in range(n_test_days):
        feat = np.concatenate([[1.0], prev_day[::-1]])
        for h in range(HOURS_PER_DAY):
            raw = np.dot(coefficients_by_hour[h], feat)
            test_forecast[d, h] = np.clip(raw, 0, 1)
        prev_day = test_solar[d]
    pred_flat = test_forecast.flatten()

    rmse = np.sqrt(np.mean((actual_flat - pred_flat) ** 2))
    nrmse = 100 * rmse / np.mean(actual_flat)

    # ---- gap 계산 (수정판 이익함수, 평가는 항상 {0,S} 오라클) ----
    sum_realized = 0.0; sum_oracle = 0.0
    for i in range(n_test_obs):
        a = actual_flat[i]; x = pred_flat[i]; dp = da_flat[i]; rp = rt_flat[i]
        pc = pr * dp
        mismatch = a - x; surplus = max(mismatch, 0); shortage = max(-mismatch, 0)
        realized = scale * (dp * x + rp * surplus - rp * shortage - pc * shortage)
        sum_realized += realized
        oracle = max(scale * rp * a, scale * dp * a)
        sum_oracle += oracle
    gap = 100 * (sum_oracle - sum_realized) / sum_oracle

    fig5_rows.append(("Proposed", pr, nrmse, gap))
    print(f"[Proposed] penalty={pr:.0%}  nRMSE={nrmse:.3f}%  gap={gap:.3f}%")


# =====================================================================
# 8. CSV 저장 + 표 출력
# =====================================================================
print()
print("=" * 60)
print(f"{'Model':>10} {'Penalty':>8} {'nRMSE':>10} {'Gap':>10}")
print("-" * 60)
for model, pr, nrmse, gap in fig5_rows:
    print(f"{model:>10} {pr:>7.0%} {nrmse:>9.2f}% {gap:>9.2f}%")
print("=" * 60)

csv_path = os.path.join(BASE_DIR, "results", "temp_fig5_논문AR수정_결과.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Penalty_Rate", "nRMSE", "Gap"])
    for model, pr, nrmse, gap in fig5_rows:
        writer.writerow([model, pr, round(nrmse, 4), round(gap, 4)])
print(f"\n저장: {csv_path}")
