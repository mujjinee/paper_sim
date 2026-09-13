# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_spo_denom_ablation.py
#
# 사용자 질문 검증: 옛 SPO+ 스크립트(spo_plus_ar_mlr_z01_block21.py,
# 삭제된 integrated_spo_plus_4term_fig3568_AR_MLR.py)의 denom(오라클 합)
# 나누기가, 보고서 2.5절에서 "최종적으로 뺐다"고 한 W1 정규화와 같은
# 것인지, 그리고 그걸 빼면(=raw로 바꾸면) 결과가 방법6.1처럼 좋아지는지
# 확인한다.
#
# "PC=rate*DA + I[DA>RT] indicator" profit 형식을 고정한 채, 정규화
# 방식만 세 가지로 바꿔서 같은 KPI 지점(W1=1,W2=20,rate=0.5)에서 비교:
#   (A) denom 정규화 (옛 spo_plus_ar_mlr_z01_block21.py 그대로)
#   (B) raw, W2 가중치 없음 (denom도 없고 weight도 없음)
#   (C) raw + W2_BALANCE_WEIGHT=4 (현재 방법6_SPO_base_AR_MLR.py 그대로)
#
# 데이터는 z03/블록18로 통일(현재 방법6.1과 같은 조건)해서 비교한다.
# =====================================================================
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")
LOCAL_HOUR_START, LOCAL_HOUR_END, HOURS_PER_DAY = 9, 21, 12
TRAIN_START = pd.Timestamp("2013-08-25"); TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23"); TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")
CAPACITY_MW, DURATION_HOURS = 30.0, 1.0
scale = CAPACITY_MW * DURATION_HOURS
KPI_W1, KPI_W2, KPI_RATE = 1, 20, 0.5
W2_BALANCE_WEIGHT = 4

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
train_dates = sorted(train_rows["local_date"].unique()); n_train_days = len(train_dates)
train_solar = np.zeros((n_train_days, HOURS_PER_DAY))
train_da = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt = np.zeros((n_train_days, HOURS_PER_DAY))
for _, row in train_rows.iterrows():
    d = (row["local_date"] - TRAIN_START).days; h = row["hour_idx"]
    train_solar[d, h] = row["solar_power"]; train_da[d, h] = row["da_price"]; train_rt[d, h] = row["rt_price"]

is_test = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])
test_dates = sorted(test_rows["local_date"].unique()); n_test_days = len(test_dates)
test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt = np.zeros((n_test_days, HOURS_PER_DAY))
for _, row in test_rows.iterrows():
    d = (row["local_date"] - TEST_START).days; h = row["hour_idx"]
    test_solar[d, h] = row["solar_power"]; test_da[d, h] = row["da_price"]; test_rt[d, h] = row["rt_price"]

n_features_ar = 13
history_and_train = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1)); ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])

actual_flat_all = test_solar.flatten(); da_flat_all = test_da.flatten(); rt_flat_all = test_rt.flatten()


def nrmse(pred, actual):
    return 100.0 * np.sqrt(np.mean((actual - pred) ** 2)) / np.mean(actual)


def compute_gap_4term_3cand(pred_flat, penalty_rate, actual, da, rt):
    """SPO+ base 방식 그대로: PC=rate*DA(indicator 없는 평가용 원식), oracle 3후보"""
    sum_r = 0.0; sum_o = 0.0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]; dp = da[i]; rp = rt[i]
        pc = penalty_rate * dp
        m = a - x
        yp = max(m, 0); ym = max(-m, 0)
        sum_r += scale * (dp * x + rp * yp - rp * ym - pc * ym)
        p0 = scale * rp * a; pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rp * s1 - rp * y1 - pc * y1)
        sum_o += max(p0, pa, p1)
    return 100.0 * (sum_o - sum_r) / sum_o if sum_o > 1e-10 else 0.0


def solve_ar_spo_variant(penalty_rate, W1, W2, mode):
    """mode: 'denom'(옛 z01 스크립트 그대로), 'raw'(정규화 없음), 'raw_weight'(현재 방법6.1)"""
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
        n_obs = n_ar_rows
        pc_h = penalty_rate * da_h
        indicator = (da_h > rt_h).astype(float)

        if mode == "denom":
            oracle = np.zeros(n_obs)
            for i in range(n_obs):
                a = y_h[i]; dp = da_h[i]; rp = rt_h[i]
                p0 = scale*rp*a; pa = scale*dp*a
                s1 = max(a-1.0,0); y1 = max(1.0-a,0)
                p1 = scale*(dp*1.0 + rp*s1 - rp*y1 - (penalty_rate*dp)*y1)
                oracle[i] = max(p0, pa, p1)
            denom = oracle.sum()
            sc = (-W1*scale*rt_h/denom) + (W2/n_obs)
            yc = (W1*scale*rt_h/denom) + (W1*scale*pc_h*indicator/denom) + (W2/n_obs)
            obj_x = -W1*scale*da_h/denom
        elif mode == "raw":
            sc = -W1*rt_h + W2
            yc = W1*rt_h + W1*pc_h*indicator + W2
            obj_x = -W1*da_h
        elif mode == "raw_weight":
            W2_eff = W2 * W2_BALANCE_WEIGHT
            sc = -W1*rt_h + W2_eff
            yc = W1*rt_h + W1*pc_h*indicator + W2_eff
            obj_x = -W1*da_h
        else:
            raise ValueError(mode)

        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
        bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)
        b_s=0; x_s=n_features_ar; yp_s=n_features_ar+n_obs
        ym_s=n_features_ar+2*n_obs; z_s=n_features_ar+3*n_obs
        n_var = n_features_ar+3*n_obs+n_bin

        obj = np.zeros(n_var)
        obj[x_s:x_s+n_obs] = obj_x
        obj[yp_s:yp_s+n_obs] = sc
        obj[ym_s:ym_s+n_obs] = yc

        X_sp = sparse.csr_matrix(ar_design); I_n = sparse.eye(n_obs, format="csr")
        eq_a = sparse.lil_matrix((n_obs, n_var))
        eq_a[:, b_s:b_s+n_features_ar] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
        eq_b = sparse.lil_matrix((n_obs, n_var))
        eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n; eq_b[:, ym_s:ym_s+n_obs] = -I_n
        all_eq = sparse.vstack([eq_a, eq_b], format="csr")
        eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
        all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]
        if n_bin > 0:
            comp = sparse.lil_matrix((2*n_bin, n_var))
            for k in range(n_bin):
                r = bin_arr[k]
                comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
                comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
            all_con.append(LinearConstraint(comp.tocsr(), np.full(2*n_bin,-np.inf),
                          np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))
        lb = np.concatenate([np.full(n_features_ar,-np.inf), np.zeros(3*n_obs+n_bin)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin)])
        integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1
        r = milp(c=obj, integrality=integ, bounds=Bounds(lb,ub), constraints=all_con,
                 options={"mip_rel_gap": 1e-9})
        if not r.success:
            raise RuntimeError(f"MILP 실패 mode={mode} h={hour}: {r.message}")
        coeffs[hour] = r.x[b_s:b_s+n_features_ar]

    fc = np.zeros((n_test_days, HOURS_PER_DAY)); prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d,h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return fc.flatten()


print("=" * 78)
print(f"SPO+ base(PC=rate*DA, I[DA>RT] 조건부) 정규화 방식 3종 비교 — KPI 지점")
print(f"(W1={KPI_W1}, W2={KPI_W2}, rate={KPI_RATE}, z03/블록18)")
print("=" * 78)
print(f"논문 참고값: AR nRMSE=34.89%, Gap=13.91%")
print()

for mode, label in [("denom", "(A) denom 정규화 (옛 spo_plus_ar_mlr_z01_block21.py 방식)"),
                     ("raw", "(B) raw, W2 가중치 없음"),
                     ("raw_weight", "(C) raw + W2_BALANCE_WEIGHT=4 (현재 방법6.1)")]:
    pred = solve_ar_spo_variant(KPI_RATE, KPI_W1, KPI_W2, mode)
    n = nrmse(pred, actual_flat_all)
    g = compute_gap_4term_3cand(pred, KPI_RATE, actual_flat_all, da_flat_all, rt_flat_all)
    print(f"{label}")
    print(f"   nRMSE={n:.2f}%  Gap={g:.2f}%   (Δ nRMSE={n-34.89:+.2f}%p, Δ Gap={g-13.91:+.2f}%p)")
    print()

print("=" * 78)
print("W1/W2를 방법1이 심하게 붕괴하는 지점(5/1, 1/0)까지 넓혀서 같은 3종 비교")
print("=" * 78)
for W1, W2, wlabel in [(5, 1, "5/1"), (1, 0, "1/0")]:
    print(f"\n--- W1/W2={wlabel} (방법1은 이 지점에서 nRMSE 116%대로 붕괴함) ---")
    for mode, mlabel in [("denom", "denom"), ("raw", "raw"), ("raw_weight", "raw+weight4")]:
        pred = solve_ar_spo_variant(KPI_RATE, W1, W2, mode)
        n = nrmse(pred, actual_flat_all)
        g = compute_gap_4term_3cand(pred, KPI_RATE, actual_flat_all, da_flat_all, rt_flat_all)
        print(f"  {mlabel:12s}: nRMSE={n:.2f}%  Gap={g:.2f}%")
