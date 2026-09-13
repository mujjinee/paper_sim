# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_regret_vs_profit_exact_equivalence.py
#
# 사용자 결론 검증: "이익식을 완전히 고정한 채, 손실만 regret(ζ=oracle-profit)
# vs raw profit-max(-profit)로 바꿔도 학습된 β는 100% 동일하다."
#
# 방법6.1(SPO+ base)의 4항 이익식(PC=rate*DA, RT 무조건 포함, PC엔 I[DA>RT])을
# 그대로 쓰되, 목적함수를 두 가지로 명시적으로 다르게 구성해서 비교한다:
#
#   (I)  "regret" 버전: obj = Σ[W1·(oracle_i − profit_i(β))] + W2·MAE
#        → oracle_i를 실제로 목적함수에 상수항(더미 변수 k=1 고정)으로 포함시켜
#          "정말로 regret을 최소화"하는 형태를 코드로 구현한다.
#   (II) "profit-max" 버전: obj = Σ[W1·(−profit_i(β))] + W2·MAE
#        → oracle 없이 순수 -profit 최소화.
#
# 두 버전의 β(및 목적함수값 차이)를 직접 비교해서, 등식이 "근사"가 아니라
# "정확히" 성립하는지 확인한다. (scipy milp는 목적함수에 상수항을 직접 못 받으므로
# 더미 변수 k(하한=상한=1)를 하나 추가해 상수*k 형태로 상수항을 흉내낸다.)
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

n_features_ar = 13
history_and_train = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1)); ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])


def oracle_3cand_raw(y, da, rt, pc):
    p0 = rt * y; pa = da * y
    surplus_full = np.maximum(y - 1.0, 0.0); shortage_full = np.maximum(1.0 - y, 0.0)
    p1 = da * 1.0 + rt * surplus_full - pc * shortage_full
    return np.maximum.reduce([p0, pa, p1])


def solve_hour(hour, W1, W2, rate, formulation):
    """formulation: 'profit_max' (오라클 없음) 또는 'regret' (오라클을 실제로 목적함수에 상수항으로 포함)"""
    y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
    n_obs = n_ar_rows
    W2_eff = W2 * W2_BALANCE_WEIGHT
    pc_h = rate * da_h
    indicator = (da_h > rt_h).astype(float)

    # 방법6.1과 완전히 동일한 4항 이익식의 경제항 계수
    sc = -W1 * rt_h + W2_eff
    yc = W1 * rt_h + W1 * pc_h * indicator + W2_eff
    obj_x = -W1 * da_h

    # regret용 oracle (4항식, PC=rate*DA*indicator 버전에 맞춘 3후보 오라클)
    pc_for_oracle = rate * da_h  # 오라클도 같은 PC 정의 사용(indicator 없이, 방법6.1 Gap 정의와 동일)
    oracle_i = oracle_3cand_raw(y_h, da_h, rt_h, pc_for_oracle)  # (raw, scale 없음 — W1과 같은 단위로 맞추기 위해 아래서 scale 곱함)
    oracle_sum_scaled = float(np.sum(W1 * scale * oracle_i))  # Σ W1*oracle_i (경제항과 같은 스케일)

    bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)
    b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
    ym_s = n_features_ar + 2*n_obs; z_s = n_features_ar + 3*n_obs
    k_s = z_s + n_bin  # 더미 변수(상수항 흉내용), regret 버전에서만 사용
    n_var = k_s + 1 if formulation == "regret" else z_s + n_bin

    obj = np.zeros(n_var)
    obj[x_s:x_s+n_obs] = obj_x
    obj[yp_s:yp_s+n_obs] = sc
    obj[ym_s:ym_s+n_obs] = yc
    if formulation == "regret":
        obj[k_s] = oracle_sum_scaled   # ← "Σoracle_i"를 실제로 목적함수에 더하는 상수항(더미 변수로 구현)

    X_sp = sparse.csr_matrix(ar_design); I_n = sparse.eye(n_obs, format="csr")
    eq_a = sparse.lil_matrix((n_obs, n_var))
    eq_a[:, b_s:b_s+n_features_ar] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
    eq_b = sparse.lil_matrix((n_obs, n_var))
    eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n; eq_b[:, ym_s:ym_s+n_obs] = -I_n
    all_eq = sparse.vstack([eq_a, eq_b], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
    if formulation == "regret":
        # 더미 변수 k=1 고정: k - 1 = 0
        k_row = sparse.lil_matrix((1, n_var)); k_row[0, k_s] = 1.0
        all_eq = sparse.vstack([all_eq, k_row], format="csr")
        eq_rhs = np.concatenate([eq_rhs, [1.0]])
    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

    if n_bin > 0:
        comp = sparse.lil_matrix((2*n_bin, n_var))
        for k in range(n_bin):
            r = bin_arr[k]
            comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
            comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
        all_con.append(LinearConstraint(comp.tocsr(), np.full(2*n_bin, -np.inf),
                      np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

    lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin)])
    ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin)])
    if formulation == "regret":
        lb = np.concatenate([lb, [1.0]]); ub = np.concatenate([ub, [1.0]])  # 더미 k ∈ [1,1]
    integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1

    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con,
             options={"mip_rel_gap": 1e-9})
    if not r.success:
        raise RuntimeError(f"MILP 실패 formulation={formulation} h={hour}: {r.message}")
    beta = r.x[b_s:b_s+n_features_ar]
    return beta, r.fun, oracle_sum_scaled


print("=" * 78)
print("이익식 완전 고정, 손실만 [regret(오라클 포함)] vs [순수 profit-max(오라클 없음)]")
print(f"KPI 지점: W1={KPI_W1}, W2={KPI_W2}, rate={KPI_RATE}, z03/블록18, 방법6.1과 동일한 4항식")
print("=" * 78)

max_beta_diff_overall = 0.0
for hour in range(HOURS_PER_DAY):
    beta_pm, fun_pm, _ = solve_hour(hour, KPI_W1, KPI_W2, KPI_RATE, "profit_max")
    beta_rg, fun_rg, oracle_sum = solve_hour(hour, KPI_W1, KPI_W2, KPI_RATE, "regret")
    diff = np.max(np.abs(beta_pm - beta_rg))
    max_beta_diff_overall = max(max_beta_diff_overall, diff)
    fun_diff = fun_rg - fun_pm
    print(f"h={hour:2d}: max|β_profit-β_regret|={diff:.2e}   "
          f"목적함수값 차이(regret-profit)={fun_diff:+.4f}   Σoracle(예상 차이)={oracle_sum:+.4f}   "
          f"{'✓ 상수항과 정확히 일치' if abs(fun_diff-oracle_sum)<1e-4 else '✗ 불일치!!'}")

print()
print(f"전체 12시간대 중 최대 |β_profit-max − β_regret| = {max_beta_diff_overall:.2e}")
if max_beta_diff_overall < 1e-6:
    print(">>> 결론: β가 수치오차 수준까지 완전히 동일하다 — regret 최소화와 profit-max는")
    print("    (같은 이익식일 때) 정확히 같은 최적화 문제라는 사용자의 결론이 실측으로도 확인됨.")
else:
    print(">>> 결론: β가 유의미하게 다르다 — 등식이 깨지는 지점이 있다는 뜻, 추가 조사 필요.")
