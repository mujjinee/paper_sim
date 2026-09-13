# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_spo_regret_vs_profitmax_full_sweep.py
#
# 방법6.1(SPO+ base, PC=rate*DA*I[DA>RT]+RT)의 실제 이익식·정규화(raw+weight4)를
# 고정한 채, "학습 목적함수"만 두 가지로 구성해서 Fig.3/5/6/8 전체 sweep에서
# 비교한다:
#
#   (A) profit-max — 방법6_SPO_base_AR_MLR.py의 실제 solve_ar_spo_base/
#       solve_mlr_spo_base 그대로 (oracle 없음, sc/yc에 -W1*RT 등 직접 반영)
#   (B) regret(literal ζ) — ζ_i에 대한 3개 부등식 제약(oracle 3후보) +
#       Σzeta_i 최소화로, 보고서 3.2.6절이 서술하는 "SPO+ 후회" 정식화를
#       문자 그대로 구현
#
# Fig.3(AR W1/W2 10점)·Fig.5(AR rate 11점)·Fig.6(MLR W1/W2 10점)·
# Fig.8(MLR rate 11점) 전 지점에서 학습된 β·nRMSE·Gap이 일치하는지 확인한다.
# =====================================================================
import os, sys, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "temp_debug_spo_regret_vs_profitmax_full_sweep.log")

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            try: st.write(s)
            except UnicodeEncodeError:
                enc = getattr(st, "encoding", "utf-8") or "utf-8"
                st.write(s.encode(enc, errors="replace").decode(enc, errors="replace"))
    def flush(self):
        for st in self.streams: st.flush()

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
W2_BALANCE_WEIGHT = 4

FIG_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1),
                (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG_LABELS = ["1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]
FIG_RATES = [round(0.1 * i, 1) for i in range(11)]
KPI_RATE = 0.5
FIG58_W1, FIG58_W2 = 1, 1

# =====================================================================
# 데이터 로딩 (방법6_SPO_base_AR_MLR.py와 동일)
# =====================================================================
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

actual_flat_all = test_solar.flatten(); da_flat_all = test_da.flatten(); rt_flat_all = test_rt.flatten()
print(f"데이터: train {n_train_days}일, test {n_test_days}일, AR {HOURS_PER_DAY}시간대, MLR n_obs={n_train_obs}")


def oracle_3cand(y, da, rt, pc, indicator):
    p0 = rt * y
    pa = da * y
    surplus_full = np.maximum(y - 1.0, 0.0); shortage_full = np.maximum(1.0 - y, 0.0)
    p1 = da * 1.0 + rt * surplus_full - rt * shortage_full - pc * indicator * shortage_full
    return p0, pa, p1


def nrmse(pred, actual):
    return 100.0 * np.sqrt(np.mean((actual - pred) ** 2)) / np.mean(actual)


def compute_gap_4term(pred_flat, penalty_rate, actual, da, rt):
    """방법6.1의 실제 Gap 함수(compute_gap_4term)와 동일 — PC=rate*DA(indicator 없는 평가용), oracle 3후보"""
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


def solve_hour_common(hour, W1, W2, rate, formulation):
    """AR 시간대 하나. formulation: 'profitmax' 또는 'regret'"""
    y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
    n_obs = n_ar_rows
    W2_eff = W2 * W2_BALANCE_WEIGHT
    pc_h = rate * da_h
    indicator = (da_h > rt_h).astype(float)

    b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
    ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs

    sc = -W1 * rt_h + W2_eff
    yc = W1 * rt_h + W1 * pc_h * indicator + W2_eff
    bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)

    if formulation == "profitmax":
        n_var = n_features_ar + 3 * n_obs + n_bin
        obj = np.zeros(n_var)
        obj[x_s:x_s+n_obs] = -W1 * da_h
        obj[yp_s:yp_s+n_obs] = sc
        obj[ym_s:ym_s+n_obs] = yc
    elif formulation == "regret":
        zeta_s = z_s + n_bin
        n_var = zeta_s + n_obs
        obj = np.zeros(n_var)
        obj[yp_s:yp_s+n_obs] = W2_eff       # 정확도항만 (경제항은 zeta로 이동)
        obj[ym_s:ym_s+n_obs] = W2_eff
        obj[zeta_s:zeta_s+n_obs] = W1        # W1 * sum(zeta_i)
    else:
        raise ValueError(formulation)

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
        all_con.append(LinearConstraint(comp.tocsr(), np.full(2*n_bin, -np.inf),
                      np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

    if formulation == "regret":
        p0, pa, p1 = oracle_3cand(y_h, da_h, rt_h, pc_h, indicator)
        # zeta_i + DA_i*x_i + RT_i*y+_i - (RT_i + PC_i*indicator_i)*y-_i >= oracle_k_i  (k=0,1,2)
        rows = []
        rhs_list = []
        for k_vals in (p0, pa, p1):
            row = sparse.lil_matrix((n_obs, n_var))
            for i in range(n_obs):
                row[i, zeta_s+i] = 1.0
                row[i, x_s+i] = da_h[i]
                row[i, yp_s+i] = rt_h[i]
                row[i, ym_s+i] = -(rt_h[i] + pc_h[i]*indicator[i])
            rows.append(row); rhs_list.append(k_vals)
        big = sparse.vstack(rows, format="csr")
        rhs = np.concatenate(rhs_list)
        all_con.append(LinearConstraint(big, rhs, np.full(len(rhs), np.inf)))

    if formulation == "profitmax":
        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin)])
    else:
        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin), np.zeros(n_obs)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin), np.full(n_obs, np.inf)])
    integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1

    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con,
             options={"mip_rel_gap": 1e-9})
    if not r.success:
        return None
    return r.x[b_s:b_s+n_features_ar]


def solve_ar(W1, W2, rate, formulation):
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        beta = solve_hour_common(hour, W1, W2, rate, formulation)
        if beta is None:
            print(f"    [경고] AR MILP 실패 h={hour} formulation={formulation}")
            beta = np.zeros(n_features_ar)
        coeffs[hour] = beta
    fc = np.zeros((n_test_days, HOURS_PER_DAY)); prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return coeffs, fc.flatten()


def solve_mlr(W1, W2, rate, formulation):
    n_obs = n_train_obs
    W2_eff = W2 * W2_BALANCE_WEIGHT
    pc_all = rate * mlr_train_da
    indicator = (mlr_train_da > mlr_train_rt).astype(float)
    sc = -W1 * mlr_train_rt + W2_eff
    yc = W1 * mlr_train_rt + W1 * pc_all * indicator + W2_eff
    bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)
    b_s = 0; x_s = n_features_mlr; yp_s = n_features_mlr + n_obs
    ym_s = n_features_mlr + 2*n_obs; z_s = n_features_mlr + 3*n_obs

    if formulation == "profitmax":
        n_var = n_features_mlr + 3*n_obs + n_bin
        obj = np.zeros(n_var)
        obj[x_s:x_s+n_obs] = -W1 * mlr_train_da
        obj[yp_s:yp_s+n_obs] = sc
        obj[ym_s:ym_s+n_obs] = yc
    elif formulation == "regret":
        zeta_s = z_s + n_bin
        n_var = zeta_s + n_obs
        obj = np.zeros(n_var)
        obj[yp_s:yp_s+n_obs] = W2_eff
        obj[ym_s:ym_s+n_obs] = W2_eff
        obj[zeta_s:zeta_s+n_obs] = W1
    else:
        raise ValueError(formulation)

    X_sp = sparse.csr_matrix(X_mlr_train); I_n = sparse.eye(n_obs, format="csr")
    eq_a = sparse.lil_matrix((n_obs, n_var))
    eq_a[:, b_s:b_s+n_features_mlr] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
    eq_b = sparse.lil_matrix((n_obs, n_var))
    eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n; eq_b[:, ym_s:ym_s+n_obs] = -I_n
    all_eq = sparse.vstack([eq_a, eq_b], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_obs), mlr_train_solar])
    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

    if n_bin > 0:
        comp = sparse.lil_matrix((2*n_bin, n_var))
        for k in range(n_bin):
            r = bin_arr[k]
            comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
            comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
        all_con.append(LinearConstraint(comp.tocsr(), np.full(2*n_bin, -np.inf),
                      np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

    if formulation == "regret":
        p0, pa, p1 = oracle_3cand(mlr_train_solar, mlr_train_da, mlr_train_rt, pc_all, indicator)
        rows = []; rhs_list = []
        for k_vals in (p0, pa, p1):
            row = sparse.lil_matrix((n_obs, n_var))
            for i in range(n_obs):
                row[i, zeta_s+i] = 1.0
                row[i, x_s+i] = mlr_train_da[i]
                row[i, yp_s+i] = mlr_train_rt[i]
                row[i, ym_s+i] = -(mlr_train_rt[i] + pc_all[i]*indicator[i])
            rows.append(row); rhs_list.append(k_vals)
        big = sparse.vstack(rows, format="csr")
        rhs = np.concatenate(rhs_list)
        all_con.append(LinearConstraint(big, rhs, np.full(len(rhs), np.inf)))

    if formulation == "profitmax":
        lb = np.concatenate([np.full(n_features_mlr, -np.inf), np.zeros(3*n_obs+n_bin)])
        ub = np.concatenate([np.full(n_features_mlr, np.inf), np.ones(3*n_obs+n_bin)])
    else:
        lb = np.concatenate([np.full(n_features_mlr, -np.inf), np.zeros(3*n_obs+n_bin), np.zeros(n_obs)])
        ub = np.concatenate([np.full(n_features_mlr, np.inf), np.ones(3*n_obs+n_bin), np.full(n_obs, np.inf)])
    integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1

    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con,
             options={"mip_rel_gap": 1e-9})
    if not r.success:
        print(f"    [경고] MLR MILP 실패 formulation={formulation}: {r.message}")
        return np.zeros(n_features_mlr), np.zeros(n_test_obs)
    coeffs = r.x[b_s:b_s+n_features_mlr]
    return coeffs, np.clip(X_mlr_test @ coeffs, 0, 1)


results = []  # (fig, label, ar_or_mlr, W1, W2, rate, max_beta_diff, n_pm, g_pm, n_rg, g_rg)

def run_point(fig_name, label, W1, W2, rate, kind):
    t0 = time.time()
    if kind == "AR":
        beta_pm, pred_pm = solve_ar(W1, W2, rate, "profitmax")
        beta_rg, pred_rg = solve_ar(W1, W2, rate, "regret")
        n_pm = nrmse(pred_pm, actual_flat_all); g_pm = compute_gap_4term(pred_pm, rate, actual_flat_all, da_flat_all, rt_flat_all)
        n_rg = nrmse(pred_rg, actual_flat_all); g_rg = compute_gap_4term(pred_rg, rate, actual_flat_all, da_flat_all, rt_flat_all)
    else:
        beta_pm, pred_pm = solve_mlr(W1, W2, rate, "profitmax")
        beta_rg, pred_rg = solve_mlr(W1, W2, rate, "regret")
        n_pm = nrmse(pred_pm, actual_flat); g_pm = compute_gap_4term(pred_pm, rate, actual_flat, da_flat, rt_flat)
        n_rg = nrmse(pred_rg, actual_flat); g_rg = compute_gap_4term(pred_rg, rate, actual_flat, da_flat, rt_flat)
    max_beta_diff = float(np.max(np.abs(np.array(beta_pm) - np.array(beta_rg))))
    dt = time.time() - t0
    flag = "OK" if max_beta_diff < 1e-6 else "MISMATCH!!"
    print(f"[{fig_name}] {kind} {label:6s} W1={W1},W2={W2},rate={rate:.1f}  "
          f"maxΔβ={max_beta_diff:.3e}  nRMSE(pm={n_pm:.3f},rg={n_rg:.3f},Δ={n_pm-n_rg:+.4f})  "
          f"Gap(pm={g_pm:.3f},rg={g_rg:.3f},Δ={g_pm-g_rg:+.4f})  [{flag}]  ({dt:.1f}s)")
    results.append(dict(fig=fig_name, label=label, kind=kind, W1=W1, W2=W2, rate=rate,
                         max_beta_diff=max_beta_diff, n_pm=n_pm, g_pm=g_pm, n_rg=n_rg, g_rg=g_rg))


print("\n" + "=" * 90)
print("Fig.3 (AR, W1/W2 10점, rate=0.5)")
print("=" * 90)
for (W1, W2), lbl in zip(FIG_W_RATIOS, FIG_LABELS):
    run_point("Fig.3", lbl, W1, W2, KPI_RATE, "AR")

print("\n" + "=" * 90)
print("Fig.5 (AR, rate 11점, W1=W2=1)")
print("=" * 90)
for rate in FIG_RATES:
    run_point("Fig.5", f"{int(rate*100)}%", FIG58_W1, FIG58_W2, rate, "AR")

print("\n" + "=" * 90)
print("Fig.6 (MLR, W1/W2 10점, rate=0.5)")
print("=" * 90)
for (W1, W2), lbl in zip(FIG_W_RATIOS, FIG_LABELS):
    run_point("Fig.6", lbl, W1, W2, KPI_RATE, "MLR")

print("\n" + "=" * 90)
print("Fig.8 (MLR, rate 11점, W1=W2=1)")
print("=" * 90)
for rate in FIG_RATES:
    run_point("Fig.8", f"{int(rate*100)}%", FIG58_W1, FIG58_W2, rate, "MLR")

print("\n" + "=" * 90)
print("종합 요약")
print("=" * 90)
mismatches = [r for r in results if r["max_beta_diff"] >= 1e-6]
print(f"전체 조건 수: {len(results)}")
print(f"불일치(maxΔβ >= 1e-6) 조건 수: {len(mismatches)}")
if mismatches:
    print("\n불일치 상세:")
    for r in mismatches:
        print(f"  [{r['fig']}] {r['kind']} {r['label']} W1={r['W1']},W2={r['W2']},rate={r['rate']}: "
              f"maxΔβ={r['max_beta_diff']:.4e}, ΔnRMSE={r['n_pm']-r['n_rg']:+.4f}, ΔGap={r['g_pm']-r['g_rg']:+.4f}")
else:
    print("\n>>> 모든 지점에서 profit-max와 regret(literal ζ)이 완전히 일치했다.")

logfile.close()
