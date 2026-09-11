# -*- coding: utf-8 -*-
# temp_zhang_c_finetune.py
# 방법3(Zhang 비대칭)의 c값을 1.0~2.0 구간에서 0.1 단위로 촘촘히 스윕해서,
# KPI 지점(W1/W2=1/20)에서 논문과 가장 가까운 c를 AR/MLR 각각 따로 찾는다.
# (방법3_zhang_asym_AR_MLR.py 의 데이터 로딩 + solve_ar_proposed/solve_mlr_proposed 로직 재사용)

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, linprog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")

LOCAL_HOUR_START, LOCAL_HOUR_END, HOURS_PER_DAY = 9, 21, 12
TRAIN_START = pd.Timestamp("2013-08-25")
TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23")
TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")
CAPACITY_MW, DURATION_HOURS = 30.0, 1.0
scale = CAPACITY_MW * DURATION_HOURS

KPI_W1, KPI_W2 = 1, 20
PAPER_AR_PROP_NRMSE_KPI = 34.89
PAPER_AR_PROP_GAP_KPI = 13.91
PAPER_MLR_PROP_NRMSE_KPI = 21.92
PAPER_MLR_PROP_GAP_KPI = 11.91

raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])
is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START

is_history = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history].copy().sort_values("hour_idx")
history_solar = np.zeros((1, HOURS_PER_DAY))
for _, row in history_rows.iterrows():
    history_solar[0, row["hour_idx"]] = row["solar_power"]

is_train = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])
is_test = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])

train_dates = sorted(train_rows["local_date"].unique())
n_train_days = len(train_dates)
test_dates = sorted(test_rows["local_date"].unique())
n_test_days = len(test_dates)

train_solar = np.zeros((n_train_days, HOURS_PER_DAY))
train_da_price = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt_price = np.zeros((n_train_days, HOURS_PER_DAY))
for _, row in train_rows.iterrows():
    d = (row["local_date"] - TRAIN_START).days; h = row["hour_idx"]
    train_solar[d, h] = row["solar_power"]; train_da_price[d, h] = row["da_price"]; train_rt_price[d, h] = row["rt_price"]

test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))
for _, row in test_rows.iterrows():
    d = (row["local_date"] - TEST_START).days; h = row["hour_idx"]
    test_solar[d, h] = row["solar_power"]; test_da_price[d, h] = row["da_price"]; test_rt_price[d, h] = row["rt_price"]

n_train_obs = len(train_rows)
mlr_train_solar = train_rows["solar_power"].to_numpy()
mlr_train_dssrd = train_rows["dssrd"].to_numpy()
mlr_train_dtsr = train_rows["dtsr"].to_numpy()
mlr_train_hour = train_rows["hour_idx"].to_numpy(dtype=float)
mlr_train_da = train_rows["da_price"].to_numpy()
mlr_train_rt = train_rows["rt_price"].to_numpy()

n_test_obs = len(test_rows)
actual_flat = test_rows["solar_power"].to_numpy()
da_flat = test_rows["da_price"].to_numpy()
rt_flat = test_rows["rt_price"].to_numpy()

n_features_mlr = 4
X_mlr_train = np.column_stack([np.ones(n_train_obs), mlr_train_dssrd, mlr_train_dtsr, mlr_train_hour])
X_mlr_test = np.column_stack([np.ones(n_test_obs), test_rows["dssrd"].to_numpy(),
                               test_rows["dtsr"].to_numpy(), test_rows["hour_idx"].to_numpy(dtype=float)])

n_features_ar = 13
history_and_train = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1))
ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])

actual_flat_all = test_solar.flatten()
da_flat_all = test_da_price.flatten()
rt_flat_all = test_rt_price.flatten()

print(f"train {n_train_days}일, test {n_test_days}일")


def compute_zhang_gap(pred_flat, c_rate, actual, da, rt):
    sum_realized = 0.0; sum_oracle = 0.0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]; dp = da[i]; rp = rt[i]
        rho_plus = c_rate * rp; rho_minus = rp
        mismatch = a - x
        surplus = max(mismatch, 0); shortage = max(-mismatch, 0)
        realized = scale * (dp * x + rho_minus * surplus - rho_plus * shortage)
        sum_realized += realized
        p0 = scale * rho_minus * a; pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rho_minus * s1 - rho_plus * y1)
        sum_oracle += max(p0, pa, p1)
    return 100.0 * (sum_oracle - sum_realized) / sum_oracle if sum_oracle > 1e-10 else 0.0


def calc_nrmse(pred_flat, actual):
    rmse = np.sqrt(np.mean((actual - pred_flat) ** 2))
    return 100.0 * rmse / np.mean(actual)


def solve_ar_proposed(c_rate, W1, W2):
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]; da_h = train_da_price[:, hour]; rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows
        rho_plus_h = c_rate * rt_h; rho_minus_h = rt_h

        oracle_train = np.zeros(n_obs)
        for i in range(n_obs):
            a = y_h[i]; dp = da_h[i]; r_p = rho_plus_h[i]; r_m = rho_minus_h[i]
            p0 = scale * r_m * a; pa = scale * dp * a
            s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
            p1 = scale * (dp * 1.0 + r_m * s1 - r_p * y1)
            oracle_train[i] = max(p0, pa, p1)
        denom = oracle_train.sum()

        sc = np.zeros(n_obs); yc = np.zeros(n_obs)
        for i in range(n_obs):
            sc[i] = (-W1 * scale * rho_minus_h[i] / denom) + (W2 / n_obs)
            yc[i] = (W1 * scale * rho_plus_h[i] / denom) + (W2 / n_obs)

        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
        bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)
        b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
        ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs
        n_var = n_features_ar + 3 * n_obs + n_bin

        obj = np.zeros(n_var)
        for i in range(n_obs):
            obj[x_s + i] = -W1 * scale * da_h[i] / denom
            obj[yp_s + i] = sc[i]; obj[ym_s + i] = yc[i]

        X_sp = sparse.csr_matrix(ar_design); I_n = sparse.eye(n_obs, format="csr")
        eq_a = sparse.lil_matrix((n_obs, n_var))
        eq_a[:, b_s:b_s+n_features_ar] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
        eq_b = sparse.lil_matrix((n_obs, n_var))
        eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n; eq_b[:, ym_s:ym_s+n_obs] = -I_n
        all_eq = sparse.vstack([eq_a, eq_b], format="csr")
        eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
        all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

        if n_bin > 0:
            comp = sparse.lil_matrix((2 * n_bin, n_var))
            for k in range(n_bin):
                r = bin_arr[k]
                comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
                comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
            all_con.append(LinearConstraint(comp.tocsr(), np.full(2*n_bin, -np.inf),
                          np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin)])
        integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1

        r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con,
                 options={"mip_rel_gap": 1e-9})
        coeffs[hour] = r.x[b_s:b_s+n_features_ar] if r.success else 0.0

    fc = np.zeros((n_test_days, HOURS_PER_DAY)); prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return fc.flatten()


def solve_mlr_proposed(c_rate, W1, W2):
    n_obs = n_train_obs
    rho_plus = c_rate * mlr_train_rt; rho_minus = mlr_train_rt
    oracle_train = np.zeros(n_obs)
    for i in range(n_obs):
        a = mlr_train_solar[i]; dp = mlr_train_da[i]; r_p = rho_plus[i]; r_m = rho_minus[i]
        p0 = scale * r_m * a; pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + r_m * s1 - r_p * y1)
        oracle_train[i] = max(p0, pa, p1)
    denom = oracle_train.sum()

    sc = np.zeros(n_obs); yc = np.zeros(n_obs)
    for i in range(n_obs):
        sc[i] = (-W1 * scale * rho_minus[i] / denom) + (W2 / n_obs)
        yc[i] = (W1 * scale * rho_plus[i] / denom) + (W2 / n_obs)

    bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)
    b_s = 0; x_s = n_features_mlr; yp_s = n_features_mlr + n_obs
    ym_s = n_features_mlr + 2 * n_obs; z_s = n_features_mlr + 3 * n_obs
    n_var = n_features_mlr + 3 * n_obs + n_bin

    obj = np.zeros(n_var)
    for i in range(n_obs):
        obj[x_s + i] = -W1 * scale * mlr_train_da[i] / denom
        obj[yp_s + i] = sc[i]; obj[ym_s + i] = yc[i]

    X_sp = sparse.csr_matrix(X_mlr_train); I_n = sparse.eye(n_obs, format="csr")
    eq_a = sparse.lil_matrix((n_obs, n_var))
    eq_a[:, b_s:b_s+n_features_mlr] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
    eq_b = sparse.lil_matrix((n_obs, n_var))
    eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n; eq_b[:, ym_s:ym_s+n_obs] = -I_n
    all_eq = sparse.vstack([eq_a, eq_b], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_obs), mlr_train_solar])
    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

    if n_bin > 0:
        comp = sparse.lil_matrix((2 * n_bin, n_var))
        for k in range(n_bin):
            r = bin_arr[k]
            comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
            comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
        all_con.append(LinearConstraint(comp.tocsr(), np.full(2*n_bin, -np.inf),
                      np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

    lb = np.concatenate([np.full(n_features_mlr, -np.inf), np.zeros(3*n_obs+n_bin)])
    ub = np.concatenate([np.full(n_features_mlr, np.inf), np.ones(3*n_obs+n_bin)])
    integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1

    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con,
             options={"mip_rel_gap": 1e-9})
    coeffs = r.x[b_s:b_s+n_features_mlr] if r.success else np.zeros(n_features_mlr)
    return np.clip(X_mlr_test @ coeffs, 0, 1)


# ── c 1.0~2.0, 0.1 단위 촘촘히 스윕 (KPI 지점 W1/W2=1/20 고정) ──
C_FINE = [round(1.0 + 0.1 * i, 1) for i in range(11)]  # 1.0,1.1,...,2.0

print("\n" + "=" * 90)
print(f"{'c':>5} | {'AR nRMSE':>9} {'AR Gap':>8} {'ΔnRMSE':>8} {'ΔGap':>7} | "
      f"{'MLR nRMSE':>9} {'MLR Gap':>8} {'ΔnRMSE':>8} {'ΔGap':>7}")
print("-" * 90)

results = []
for c in C_FINE:
    ar_pred = solve_ar_proposed(c, KPI_W1, KPI_W2)
    ar_n = calc_nrmse(ar_pred, actual_flat_all)
    ar_g = compute_zhang_gap(ar_pred, c, actual_flat_all, da_flat_all, rt_flat_all)
    mlr_pred = solve_mlr_proposed(c, KPI_W1, KPI_W2)
    mlr_n = calc_nrmse(mlr_pred, actual_flat)
    mlr_g = compute_zhang_gap(mlr_pred, c, actual_flat, da_flat, rt_flat)

    d_ar_n = ar_n - PAPER_AR_PROP_NRMSE_KPI
    d_ar_g = ar_g - PAPER_AR_PROP_GAP_KPI
    d_mlr_n = mlr_n - PAPER_MLR_PROP_NRMSE_KPI
    d_mlr_g = mlr_g - PAPER_MLR_PROP_GAP_KPI

    results.append((c, ar_n, ar_g, d_ar_n, d_ar_g, mlr_n, mlr_g, d_mlr_n, d_mlr_g))
    print(f"{c:>5} | {ar_n:>8.2f}% {ar_g:>7.2f}% {d_ar_n:>+7.2f}p {d_ar_g:>+6.2f}p | "
          f"{mlr_n:>8.2f}% {mlr_g:>7.2f}% {d_mlr_n:>+7.2f}p {d_mlr_g:>+6.2f}p")

print("\n" + "=" * 90)
best_ar = min(results, key=lambda r: abs(r[3]) + abs(r[4]))
best_mlr = min(results, key=lambda r: abs(r[7]) + abs(r[8]))
print(f"AR  최적 c = {best_ar[0]}  (|ΔnRMSE|+|ΔGap| = {abs(best_ar[3])+abs(best_ar[4]):.2f})")
print(f"MLR 최적 c = {best_mlr[0]}  (|ΔnRMSE|+|ΔGap| = {abs(best_mlr[7])+abs(best_mlr[8]):.2f})")

print("\n(참고) c=1.5 현재 사용값:")
c15 = [r for r in results if r[0] == 1.5][0]
print(f"AR : |Δ|={abs(c15[3])+abs(c15[4]):.2f}   MLR: |Δ|={abs(c15[7])+abs(c15[8]):.2f}")
