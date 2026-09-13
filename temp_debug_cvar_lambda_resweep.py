# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_cvar_lambda_resweep.py
#
# temp_debug_cvar_weight_dilution.py의 진단 결과(W2_BALANCE_WEIGHT=4가
# CVaR 효과를 희석시킨다)를 바탕으로, 가중치 보정이 있는 상태(방법7 그대로,
# W2_eff=W2*4)에서 nRMSE를 너무 망가뜨리지 않으면서 Total regret/CVaR90을
# 의미 있게 줄이는 새 λ를 찾는다 (방법2가 W2_BALANCE_WEIGHT=4를 찾을 때 쓴
# SSE 스윕과 같은 정신 — 다만 여기서는 "논문 판독값과의 SSE"가 아니라
# "ΔnRMSE 허용범위 안에서 ΔTotalRegret/ΔCVaR90 최대화"가 기준이다).
#
# KPI 지점(W1=1,W2=20,rate=50%)에서 λ를 0.05~1.5까지 촘촘히 스윕하며
# AR·MLR 둘 다 nRMSE/Gap/TotalRegret/CVaR90/MaxRegret을 계산한다.
#
# 실행 로그는 temp_debug_cvar_lambda_resweep.log 에 저장한다.
# =====================================================================
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, linprog

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_debug_cvar_lambda_resweep.log")

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
KPI_W1, KPI_W2, KPI_RATE = 1, 20, 0.5
W2_BALANCE_WEIGHT = 4  # 방법2/방법7과 동일 (가중치 보정 유지)

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

n_train_obs = len(train_rows)
mlr_train_solar = train_rows["solar_power"].to_numpy()
mlr_train_dssrd = train_rows["dssrd"].to_numpy(); mlr_train_dtsr = train_rows["dtsr"].to_numpy()
mlr_train_hour = train_rows["hour_idx"].to_numpy(dtype=float)
mlr_train_da = train_rows["da_price"].to_numpy(); mlr_train_rt = train_rows["rt_price"].to_numpy()
n_test_obs = len(test_rows)
actual_flat = test_rows["solar_power"].to_numpy(); da_flat = test_rows["da_price"].to_numpy(); rt_flat = test_rows["rt_price"].to_numpy()

n_features_mlr = 4
X_mlr_train = np.column_stack([np.ones(n_train_obs), mlr_train_dssrd, mlr_train_dtsr, mlr_train_hour])
X_mlr_test = np.column_stack([np.ones(n_test_obs), test_rows["dssrd"].to_numpy(),
                               test_rows["dtsr"].to_numpy(), test_rows["hour_idx"].to_numpy(dtype=float)])

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


def solve_mlr_cvar(penalty_rate, W1, W2, lam):
    W2_eff = W2 * W2_BALANCE_WEIGHT
    n_obs = n_train_obs
    pc_all = mlr_train_rt + penalty_rate * mlr_train_da
    sc = -W1 * mlr_train_rt + W2_eff; yc = W1 * pc_all + W2_eff
    oracle_all = oracle_3cand_raw(mlr_train_solar, mlr_train_da, mlr_train_rt, pc_all)
    bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)
    b_s = 0; x_s = n_features_mlr; yp_s = n_features_mlr + n_obs
    ym_s = n_features_mlr + 2 * n_obs; z_s = n_features_mlr + 3 * n_obs
    zeta_s = z_s + n_bin; s_s = zeta_s + 1; n_var = s_s + n_obs
    obj = np.zeros(n_var)
    obj[x_s:x_s + n_obs] = -W1 * mlr_train_da
    obj[yp_s:yp_s + n_obs] = sc
    obj[ym_s:ym_s + n_obs] = yc
    obj[zeta_s] = W1 * lam * n_obs
    obj[s_s:s_s + n_obs] = W1 * lam / (1.0 - CVAR_ALPHA)
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
    if lam > 0.0:
        cvar_a = sparse.lil_matrix((n_obs, n_var))
        for i in range(n_obs):
            cvar_a[i, s_s + i] = 1.0; cvar_a[i, zeta_s] = 1.0
            cvar_a[i, x_s + i] = mlr_train_da[i]; cvar_a[i, yp_s + i] = mlr_train_rt[i]; cvar_a[i, ym_s + i] = -pc_all[i]
        all_con.append(LinearConstraint(cvar_a.tocsr(), oracle_all, np.full(n_obs, np.inf)))
    lb = np.concatenate([np.full(n_features_mlr, -np.inf), np.zeros(3*n_obs+n_bin), [0.0], np.zeros(n_obs)])
    ub = np.concatenate([np.full(n_features_mlr, np.inf), np.ones(3*n_obs+n_bin), [np.inf], np.full(n_obs, np.inf)])
    integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1
    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con, options={"mip_rel_gap": 1e-9})
    if not r.success:
        raise RuntimeError(f"MLR CVaR MILP failed lam={lam}: {r.message}")
    coeffs = r.x[b_s:b_s+n_features_mlr]
    return np.clip(X_mlr_test @ coeffs, 0, 1)


LAMBDA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]

print("\n" + "=" * 70)
print(f"  가중치 보정 있음(W2_eff=W2*{W2_BALANCE_WEIGHT}) 상태에서 λ 재스윕 — KPI 지점(W1=1,W2=20,rate=50%)")
print("=" * 70)

rows = []
base_ar = None; base_mlr = None
for lam in LAMBDA_GRID:
    ar_pred = solve_ar_cvar(KPI_RATE, KPI_W1, KPI_W2, lam)
    mlr_pred = solve_mlr_cvar(KPI_RATE, KPI_W1, KPI_W2, lam)
    a_n = nrmse(ar_pred, actual_flat_all)
    a_g = compute_gap_4term(ar_pred, KPI_RATE, actual_flat_all, da_flat_all, rt_flat_all)
    a_d = compute_regret_diagnostics(ar_pred, KPI_RATE, actual_flat_all, da_flat_all, rt_flat_all)
    m_n = nrmse(mlr_pred, actual_flat)
    m_g = compute_gap_4term(mlr_pred, KPI_RATE, actual_flat, da_flat, rt_flat)
    m_d = compute_regret_diagnostics(mlr_pred, KPI_RATE, actual_flat, da_flat, rt_flat)
    if lam == 0.0:
        base_ar = (a_n, a_g, a_d); base_mlr = (m_n, m_g, m_d)
    d_an = a_n - base_ar[0]; d_ag = a_g - base_ar[1]
    d_atr = 100*(a_d["total_regret"]-base_ar[2]["total_regret"])/abs(base_ar[2]["total_regret"])
    d_acv = 100*(a_d["cvar90_regret"]-base_ar[2]["cvar90_regret"])/abs(base_ar[2]["cvar90_regret"])
    d_amr = 100*(a_d["max_regret"]-base_ar[2]["max_regret"])/abs(base_ar[2]["max_regret"])
    d_mn = m_n - base_mlr[0]; d_mg = m_g - base_mlr[1]
    d_mtr = 100*(m_d["total_regret"]-base_mlr[2]["total_regret"])/abs(base_mlr[2]["total_regret"])
    d_mcv = 100*(m_d["cvar90_regret"]-base_mlr[2]["cvar90_regret"])/abs(base_mlr[2]["cvar90_regret"])
    rows.append(dict(lam=lam, a_n=a_n, a_g=a_g, d_an=d_an, d_ag=d_ag, d_atr=d_atr, d_acv=d_acv, d_amr=d_amr,
                      m_n=m_n, m_g=m_g, d_mn=d_mn, d_mg=d_mg, d_mtr=d_mtr, d_mcv=d_mcv))
    print(f"  lam={lam:<5} AR nRMSE={a_n:.3f}%(Δ{d_an:+.3f}) Gap={a_g:.3f}%(Δ{d_ag:+.3f})  "
          f"ΔTotalRegret={d_atr:+.2f}% ΔCVaR90={d_acv:+.2f}% ΔMaxRegret={d_amr:+.2f}%   |   "
          f"MLR nRMSE={m_n:.3f}%(Δ{d_mn:+.3f}) Gap={m_g:.3f}%(Δ{d_mg:+.3f}) "
          f"ΔTotalRegret={d_mtr:+.2f}% ΔCVaR90={d_mcv:+.2f}%")

df = pd.DataFrame(rows)
csv_path = os.path.join(BASE_DIR, "results", "simulation_output", "temp_cvar_lambda_resweep_weighted.csv")
df.to_csv(csv_path, index=False)
print(f"\n  saved: {csv_path}")

# 추천 λ: AR ΔnRMSE <= +1.0%p 이면서 ΔTotalRegret이 가장 큰 폭으로 감소하는 지점
candidates = df[df["d_an"] <= 1.0]
if len(candidates) > 0:
    best = candidates.loc[candidates["d_atr"].idxmin()]
    print(f"\n  [추천] ΔnRMSE(AR) <= +1.0%p 제약 하에서 ΔTotalRegret이 가장 크게 줄어드는 λ = {best['lam']}")
    print(f"         AR: nRMSE {best['a_n']:.3f}%(Δ{best['d_an']:+.3f}) Gap {best['a_g']:.3f}%(Δ{best['d_ag']:+.3f}) "
          f"ΔTotalRegret={best['d_atr']:+.2f}% ΔCVaR90={best['d_acv']:+.2f}%")
else:
    print("\n  [추천] ΔnRMSE(AR) <= +1.0%p 조건을 만족하는 λ가 없음 (λ=0.05조차 이미 초과하거나, 모든 후보가 기준 초과)")

logfile.close()
