# -*- coding: utf-8 -*-
# temp_debug_fig5_weight_sweep.py
# Fig.5(AR, rate 스윕)이 논문 모양(제안모형 nRMSE가 rate에 따라 34~67%로 크게 움직이고,
# Gap은 11% 근처로 평평)을 재현하는지, W2 weight 값을 여러 개 바꿔가며 비교한다.
# 방법2_4term_ar_mlr_z03_block18.py 의 데이터 로딩/AR MILP 로직을 그대로 재사용(축소판).

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, linprog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")

LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21
HOURS_PER_DAY = 12

TRAIN_START = pd.Timestamp("2013-08-25")
TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23")
TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS

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
    d = (row["local_date"] - TRAIN_START).days
    h = row["hour_idx"]
    train_solar[d, h] = row["solar_power"]
    train_da_price[d, h] = row["da_price"]
    train_rt_price[d, h] = row["rt_price"]

test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))
for _, row in test_rows.iterrows():
    d = (row["local_date"] - TEST_START).days
    h = row["hour_idx"]
    test_solar[d, h] = row["solar_power"]
    test_da_price[d, h] = row["da_price"]
    test_rt_price[d, h] = row["rt_price"]

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


def compute_gap_4term(pred_flat, penalty_rate, actual, da, rt):
    sum_realized = 0.0
    sum_oracle = 0.0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]
        dp = da[i]; rp = rt[i]
        pc = penalty_rate * dp
        mismatch = a - x
        surplus = max(mismatch, 0)
        shortage = max(-mismatch, 0)
        realized = scale * (dp * x + rp * surplus - rp * shortage - pc * shortage)
        sum_realized += realized
        p0 = scale * rp * a
        pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rp * s1 - rp * y1 - pc * y1)
        oracle = max(p0, pa, p1)
        sum_oracle += oracle
    return 100.0 * (sum_oracle - sum_realized) / sum_oracle if sum_oracle > 1e-10 else 0.0


def nrmse(pred_flat, actual):
    rmse = np.sqrt(np.mean((actual - pred_flat) ** 2))
    return 100.0 * rmse / np.mean(actual)


# AR baseline (한 번만)
ar_coefficients = np.zeros((HOURS_PER_DAY, n_features_ar))
for h in range(HOURS_PER_DAY):
    y_h = train_solar[:, h]
    X_sp = sparse.csr_matrix(ar_design)
    I_n = sparse.eye(n_ar_rows, format="csr")
    A = sparse.vstack([sparse.hstack([X_sp, -I_n]), sparse.hstack([-X_sp, -I_n])], format="csr")
    b = np.concatenate([y_h, -y_h])
    c = np.concatenate([np.zeros(n_features_ar), np.ones(n_ar_rows) / n_ar_rows])
    vb = [(None, None)] * n_features_ar + [(0.0, None)] * n_ar_rows
    res = linprog(c, A_ub=A, b_ub=b, bounds=vb, method="highs")
    ar_coefficients[h] = res.x[:n_features_ar]

ar_test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))
prev_day = train_solar[-1]
for d in range(n_test_days):
    feat = np.concatenate([[1.0], prev_day[::-1]])
    for h in range(HOURS_PER_DAY):
        ar_test_forecast[d, h] = np.clip(np.dot(ar_coefficients[h], feat), 0, 1)
    prev_day = test_solar[d]
ar_pred_flat = ar_test_forecast.flatten()
ar_nrmse = nrmse(ar_pred_flat, actual_flat_all)
print(f"AR baseline nRMSE = {ar_nrmse:.2f}%")


def solve_ar_proposed(penalty_rate, W1, W2):
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]
        da_h = train_da_price[:, hour]
        rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows

        oracle_train = np.zeros(n_obs)
        for i in range(n_obs):
            a = y_h[i]; dp = da_h[i]; rp = rt_h[i]
            pc = penalty_rate * dp
            p0 = scale * rp * a; pa = scale * dp * a
            s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
            p1 = scale * (dp * 1.0 + rp * s1 - rp * y1 - pc * y1)
            oracle_train[i] = max(p0, pa, p1)
        denom = oracle_train.sum()

        sc = np.zeros(n_obs)
        yc = np.zeros(n_obs)
        for i in range(n_obs):
            pc = penalty_rate * da_h[i]
            sc[i] = (-W1 * scale * rt_h[i] / denom) + (W2 / n_obs)
            yc[i] = (W1 * scale * (rt_h[i] + pc) / denom) + (W2 / n_obs)

        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
        bin_arr = np.array(bin_list, dtype=int)
        n_bin = len(bin_arr)

        b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
        ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs
        n_var = n_features_ar + 3 * n_obs + n_bin

        obj = np.zeros(n_var)
        for i in range(n_obs):
            obj[x_s + i] = -W1 * scale * da_h[i] / denom
            obj[yp_s + i] = sc[i]; obj[ym_s + i] = yc[i]

        X_sp = sparse.csr_matrix(ar_design)
        I_n = sparse.eye(n_obs, format="csr")
        eq_a = sparse.lil_matrix((n_obs, n_var))
        eq_a[:, b_s:b_s + n_features_ar] = -X_sp; eq_a[:, x_s:x_s + n_obs] = I_n
        eq_b = sparse.lil_matrix((n_obs, n_var))
        eq_b[:, x_s:x_s + n_obs] = I_n; eq_b[:, yp_s:yp_s + n_obs] = I_n
        eq_b[:, ym_s:ym_s + n_obs] = -I_n
        all_eq = sparse.vstack([eq_a, eq_b], format="csr")
        eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
        all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

        if n_bin > 0:
            comp = sparse.lil_matrix((2 * n_bin, n_var))
            for k in range(n_bin):
                r = bin_arr[k]
                comp[k, yp_s + r] = 1.0; comp[k, z_s + k] = 1.0
                comp[n_bin + k, ym_s + r] = 1.0; comp[n_bin + k, z_s + k] = -1.0
            all_con.append(LinearConstraint(comp.tocsr(),
                          np.full(2 * n_bin, -np.inf),
                          np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3 * n_obs + n_bin)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3 * n_obs + n_bin)])
        integ = np.zeros(n_var, dtype=int)
        integ[z_s:z_s + n_bin] = 1

        r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),
                 constraints=all_con, options={"mip_rel_gap": 1e-9})
        if not r.success:
            raise RuntimeError(f"AR MILP failed h={hour} rate={penalty_rate} W1={W1} W2={W2}: {r.message}")
        coeffs[hour] = r.x[b_s:b_s + n_features_ar]

    fc = np.zeros((n_test_days, HOURS_PER_DAY))
    prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return fc.flatten()


# ── 논문 Fig.5(AR) 판독값 (참고용) ──
RATES_PAPER = [round(0.1 * i, 1) for i in range(11)]  # 0~100%
PAPER_PROP_NRMSE = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
PAPER_PROP_GAP   = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]

TEST_RATES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
W2_CANDIDATES = [1, 2, 3, 5, 8, 15]
W1 = 1

print("\n" + "=" * 100)
print(f"{'W2':>4} | " + " | ".join(f"rate={r:.1f}" for r in TEST_RATES))
print("-" * 100)

results = {}  # W2 -> {rate: (nrmse, gap)}
for W2 in W2_CANDIDATES:
    row_n = []
    row_g = []
    for rate in TEST_RATES:
        pred = solve_ar_proposed(rate, W1, W2)
        n = nrmse(pred, actual_flat_all)
        g = compute_gap_4term(pred, rate, actual_flat_all, da_flat_all, rt_flat_all)
        row_n.append(n); row_g.append(g)
    results[W2] = (row_n, row_g)
    print(f"W2={W2:>3} nRMSE: " + " | ".join(f"{v:6.2f}" for v in row_n))
    print(f"      Gap  : " + " | ".join(f"{v:6.2f}" for v in row_g))

print("\n논문 판독값 (0,10,30,50,70,100%만 추출):")
idx_map = [RATES_PAPER.index(r) for r in TEST_RATES if r in RATES_PAPER]
paper_n = [PAPER_PROP_NRMSE[RATES_PAPER.index(r)] for r in TEST_RATES]
paper_g = [PAPER_PROP_GAP[RATES_PAPER.index(r)] for r in TEST_RATES]
print("nRMSE: " + " | ".join(f"{v:6.2f}" for v in paper_n))
print("Gap  : " + " | ".join(f"{v:6.2f}" for v in paper_g))

print(f"\nAR baseline nRMSE (rate무관, 고정): {ar_nrmse:.2f}%")

print("\n" + "=" * 100)
print("W2별 논문과의 오차(SSE, nRMSE+Gap 합산 기준)")
print("=" * 100)
for W2 in W2_CANDIDATES:
    row_n, row_g = results[W2]
    sse = sum((row_n[i] - paper_n[i]) ** 2 + (row_g[i] - paper_g[i]) ** 2 for i in range(len(TEST_RATES)))
    print(f"W2={W2:>3}: SSE={sse:8.2f}")
