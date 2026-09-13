# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_cvar_lambda_resweep_fig58.py
#
# temp_debug_cvar_lambda_resweep.py는 KPI 지점(W1=1,W2=20)에서 λ=0.2를 찾았다.
# 그런데 Fig.5/8은 W1=W2=1(W2_eff=W2*4=4, KPI의 80보다 20배 작음)을 쓰기 때문에,
# 같은 λ=0.2가 20배 가까이 강하게 작동해 nRMSE를 크게 무너뜨린다(방법7 재실행 결과
# 확인, 시도방법7.md §24 후속 문제). 이 스크립트는 W1=W2=1, rate=0.5(Fig.5/8의
# 기준점)에서 λ를 훨씬 작은 범위로 다시 스윕해 "Fig.5/8 전용 λ"를 찾는다.
#
# 실행 로그: temp_debug_cvar_lambda_resweep_fig58.log
# =====================================================================
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_debug_cvar_lambda_resweep_fig58.log")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except UnicodeEncodeError:
                enc = getattr(st, "encoding", "utf-8") or "utf-8"
                st.write(s.encode(enc, errors="replace").decode(enc, errors="replace"))
    def flush(self):
        for st in self.streams:
            st.flush()

logfile = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, logfile)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")
LOCAL_HOUR_START, LOCAL_HOUR_END, HOURS_PER_DAY = 9, 21, 12
TRAIN_START = pd.Timestamp("2013-08-25"); TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23"); TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")
CAPACITY_MW, DURATION_HOURS = 30.0, 1.0
scale = CAPACITY_MW * DURATION_HOURS
CVAR_ALPHA = 0.90
FIG58_W1, FIG58_W2, FIG58_RATE = 1, 1, 0.5  # Fig.5/8 기준점(rate=KPI와 동일 50%)
W2_BALANCE_WEIGHT = 4

print("=" * 70); print("  데이터 로딩"); print("=" * 70)
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

train_dates = sorted(train_rows["local_date"].unique()); n_train_days = len(train_dates)
test_dates = sorted(test_rows["local_date"].unique()); n_test_days = len(test_dates)

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

n_features_ar = 13
history_and_train = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1)); ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])

actual_flat_all = test_solar.flatten(); da_flat_all = test_da_price.flatten(); rt_flat_all = test_rt_price.flatten()
print(f"  train days={n_train_days}, test days={n_test_days}")


def oracle_3cand_raw(y, da, rt, pc):
    p0 = rt * y; pa = da * y
    surplus_full = np.maximum(y - 1.0, 0.0); shortage_full = np.maximum(1.0 - y, 0.0)
    p1 = da * 1.0 + rt * surplus_full - pc * shortage_full
    return np.maximum.reduce([p0, pa, p1])

def compute_gap_4term(pred_flat, penalty_rate, actual, da, rt):
    sum_realized = 0.0; sum_oracle = 0.0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]; dp = da[i]; rp = rt[i]
        pc = rp + penalty_rate * dp
        mismatch = a - x
        surplus = max(mismatch, 0); shortage = max(-mismatch, 0)
        realized = scale * (dp * x + rp * surplus - pc * shortage)
        sum_realized += realized
        sum_oracle += max(scale * rp * a, scale * dp * a)
    return 100.0 * (sum_oracle - sum_realized) / sum_oracle if sum_oracle > 1e-10 else 0.0

def nrmse(pred_flat, actual):
    return 100.0 * np.sqrt(np.mean((actual - pred_flat) ** 2)) / np.mean(actual)

def compute_regret_diagnostics(pred_flat, penalty_rate, actual, da, rt):
    pc = rt + penalty_rate * da
    mismatch = actual - pred_flat
    surplus = np.maximum(mismatch, 0.0); shortage = np.maximum(-mismatch, 0.0)
    realized = scale * (da * pred_flat + rt * surplus - pc * shortage)
    oracle = scale * oracle_3cand_raw(actual, da, rt, pc)
    regret = oracle - realized
    regret = np.where(np.abs(regret) < 1e-6, 0.0, regret)
    n = len(regret)
    tail_count = max(1, int(np.ceil((1.0 - CVAR_ALPHA) * n)))
    sorted_r = np.sort(regret)
    return {"total_regret": float(np.sum(regret)), "cvar90_regret": float(np.mean(sorted_r[-tail_count:])),
            "max_regret": float(np.max(regret))}


def solve_ar_cvar(penalty_rate, W1, W2, lam):
    W2_eff = W2 * W2_BALANCE_WEIGHT
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]; da_h = train_da_price[:, hour]; rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows
        pc_h = rt_h + penalty_rate * da_h
        sc = -W1 * rt_h + W2_eff; yc = W1 * pc_h + W2_eff
        oracle_h = oracle_3cand_raw(y_h, da_h, rt_h, pc_h)
        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
        bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)
        b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
        ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs
        zeta_s = z_s + n_bin; s_s = zeta_s + 1; n_var = s_s + n_obs
        obj = np.zeros(n_var)
        obj[x_s:x_s + n_obs] = -W1 * da_h
        obj[yp_s:yp_s + n_obs] = sc
        obj[ym_s:ym_s + n_obs] = yc
        obj[zeta_s] = W1 * lam * n_obs
        obj[s_s:s_s + n_obs] = W1 * lam / (1.0 - CVAR_ALPHA)
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
        if lam > 0.0:
            cvar_a = sparse.lil_matrix((n_obs, n_var))
            for i in range(n_obs):
                cvar_a[i, s_s + i] = 1.0; cvar_a[i, zeta_s] = 1.0
                cvar_a[i, x_s + i] = da_h[i]; cvar_a[i, yp_s + i] = rt_h[i]; cvar_a[i, ym_s + i] = -pc_h[i]
            all_con.append(LinearConstraint(cvar_a.tocsr(), oracle_h, np.full(n_obs, np.inf)))
        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin), [0.0], np.zeros(n_obs)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin), [np.inf], np.full(n_obs, np.inf)])
        integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1
        r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con, options={"mip_rel_gap": 1e-9})
        if not r.success:
            raise RuntimeError(f"AR CVaR MILP failed h={hour} lam={lam}: {r.message}")
        coeffs[hour] = r.x[b_s:b_s+n_features_ar]
    fc = np.zeros((n_test_days, HOURS_PER_DAY)); prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return fc.flatten()


LAMBDA_GRID = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]

print("\n" + "=" * 70)
print(f"  Fig.5/8 기준점(W1={FIG58_W1},W2={FIG58_W2},rate={FIG58_RATE})에서 λ 재스윕")
print("=" * 70)

rows = []
base = None
for lam in LAMBDA_GRID:
    pred = solve_ar_cvar(FIG58_RATE, FIG58_W1, FIG58_W2, lam)
    n = nrmse(pred, actual_flat_all)
    g = compute_gap_4term(pred, FIG58_RATE, actual_flat_all, da_flat_all, rt_flat_all)
    d = compute_regret_diagnostics(pred, FIG58_RATE, actual_flat_all, da_flat_all, rt_flat_all)
    if lam == 0.0:
        base = (n, g, d)
    d_n = n - base[0]; d_g = g - base[1]
    d_tr = 100*(d["total_regret"]-base[2]["total_regret"])/abs(base[2]["total_regret"])
    d_cv = 100*(d["cvar90_regret"]-base[2]["cvar90_regret"])/abs(base[2]["cvar90_regret"])
    d_mr = 100*(d["max_regret"]-base[2]["max_regret"])/abs(base[2]["max_regret"])
    rows.append(dict(lam=lam, n=n, g=g, d_n=d_n, d_g=d_g, d_tr=d_tr, d_cv=d_cv, d_mr=d_mr))
    print(f"  lam={lam:<7} nRMSE={n:.3f}%(Δ{d_n:+.3f}) Gap={g:.3f}%(Δ{d_g:+.3f})  "
          f"ΔTotalRegret={d_tr:+.2f}% ΔCVaR90={d_cv:+.2f}% ΔMaxRegret={d_mr:+.2f}%")

df = pd.DataFrame(rows)
csv_path = os.path.join(BASE_DIR, "results", "simulation_output", "temp_cvar_lambda_resweep_fig58.csv")
df.to_csv(csv_path, index=False)
print(f"\n  saved: {csv_path}")

candidates = df[df["d_n"] <= 1.0]
if len(candidates) > 0:
    best = candidates.loc[candidates["d_tr"].idxmin()]
    print(f"\n  [추천] ΔnRMSE <= +1.0%p 제약 하에서 ΔTotalRegret이 가장 크게 줄어드는 λ = {best['lam']}")
    print(f"         nRMSE {best['n']:.3f}%(Δ{best['d_n']:+.3f}) Gap {best['g']:.3f}%(Δ{best['d_g']:+.3f}) "
          f"ΔTotalRegret={best['d_tr']:+.2f}% ΔCVaR90={best['d_cv']:+.2f}% ΔMaxRegret={best['d_mr']:+.2f}%")
else:
    print("\n  [추천] 조건을 만족하는 λ가 없음")

logfile.close()
