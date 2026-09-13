# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_cvar_weight_dilution.py
#
# 목적: 방법7(방법2+additive CVaR)의 λ=0.05 효과가 시도방법7.md §14가 보고한
# 것보다 훨씬 약하게(거의 0에 가깝게) 나온 이유를 진단한다.
#
# 가설: 방법2의 W2_BALANCE_WEIGHT=4 보정(W2_eff=W2*4)이 CVaR 항(W1에만 비례,
# W2와 무관)의 상대적 영향력을 약화시킨다 — 추가방법론_5_(4)의 원본 Gurobi
# CVaR 코드는 이 보정 없이 raw W2를 그대로 쓴다(방법2_4term의 §10.2 단계,
# 즉 방법2가 "너무 강한 raw W1"을 고치기 전 단계에 CVaR를 얹은 것과 같다).
#
# 검증 방법:
#   (A) W2_eff = W2*4 (방법7 그대로, "가중치 보정 있음") — λ=0 vs λ=0.05
#   (B) W2_eff = W2   (원본 참고코드처럼 "가중치 보정 없음")   — λ=0 vs λ=0.05
#   두 경우 모두 KPI 지점(W1=1,W2=20,rate=50%)에서 AR 12시간대 전부 풀고,
#   nRMSE/Gap/Total regret/CVaR90/Max regret + 각 시간대 ζ(zeta)와 tail 활성화
#   표본 수(s_i>0인 개수)를 출력한다.
#
# 실행 결과는 전부 temp_debug_cvar_weight_dilution.log 에도 저장한다.
# =====================================================================
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, linprog

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_debug_cvar_weight_dilution.log")

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

CVAR_ALPHA = 0.90
KPI_W1, KPI_W2, KPI_RATE = 1, 20, 0.5

print("=" * 70)
print("  데이터 로딩")
print("=" * 70)
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
    train_solar[d, h] = row["solar_power"]
    train_da_price[d, h] = row["da_price"]
    train_rt_price[d, h] = row["rt_price"]

test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))
for _, row in test_rows.iterrows():
    d = (row["local_date"] - TEST_START).days; h = row["hour_idx"]
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

print(f"  train days={n_train_days}, test days={n_test_days}")


def oracle_3cand_raw(y, da, rt, pc):
    p0 = rt * y
    pa = da * y
    surplus_full = np.maximum(y - 1.0, 0.0)
    shortage_full = np.maximum(1.0 - y, 0.0)
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
        p0 = scale * rp * a; pa = scale * dp * a
        sum_oracle += max(p0, pa)
    return 100.0 * (sum_oracle - sum_realized) / sum_oracle if sum_oracle > 1e-10 else 0.0


def nrmse(pred_flat, actual):
    rmse = np.sqrt(np.mean((actual - pred_flat) ** 2))
    return 100.0 * rmse / np.mean(actual)


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
    cvar90 = float(np.mean(sorted_r[-tail_count:]))
    return {
        "total_regret": float(np.sum(regret)), "mean_regret": float(np.mean(regret)),
        "cvar90_regret": cvar90, "max_regret": float(np.max(regret)),
    }


def solve_ar_proposed_cvar_debug(penalty_rate, W1, W2, lam, weight_correction):
    """weight_correction=True: W2_eff=W2*4 (방법7 그대로) / False: W2_eff=W2 (원본 참고코드처럼 보정 없음)"""
    W2_eff = W2 * 4 if weight_correction else W2
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))
    zeta_by_hour = np.zeros(HOURS_PER_DAY)
    n_active_tail_by_hour = np.zeros(HOURS_PER_DAY, dtype=int)
    obj_terms_by_hour = []

    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]; da_h = train_da_price[:, hour]; rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows
        pc_h = rt_h + penalty_rate * da_h
        sc = -W1 * rt_h + W2_eff
        yc = W1 * pc_h + W2_eff
        oracle_h = oracle_3cand_raw(y_h, da_h, rt_h, pc_h)

        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
        bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)

        b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
        ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs
        zeta_s = z_s + n_bin; s_s = zeta_s + 1
        n_var = s_s + n_obs

        obj = np.zeros(n_var)
        obj[x_s:x_s + n_obs] = -W1 * da_h
        obj[yp_s:yp_s + n_obs] = sc
        obj[ym_s:ym_s + n_obs] = yc
        obj[zeta_s] = W1 * lam * n_obs
        obj[s_s:s_s + n_obs] = W1 * lam / (1.0 - CVAR_ALPHA)

        X_sp = sparse.csr_matrix(ar_design)
        I_n = sparse.eye(n_obs, format="csr")
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
                cvar_a[i, s_s + i] = 1.0
                cvar_a[i, zeta_s] = 1.0
                cvar_a[i, x_s + i] = da_h[i]
                cvar_a[i, yp_s + i] = rt_h[i]
                cvar_a[i, ym_s + i] = -pc_h[i]
            all_con.append(LinearConstraint(cvar_a.tocsr(), oracle_h, np.full(n_obs, np.inf)))

        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin), [0.0], np.zeros(n_obs)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin), [np.inf], np.full(n_obs, np.inf)])
        integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1

        r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con,
                 options={"mip_rel_gap": 1e-9})
        if not r.success:
            raise RuntimeError(f"AR CVaR MILP failed h={hour} lam={lam}: {r.message}")
        coeffs[hour] = r.x[b_s:b_s+n_features_ar]
        if lam > 0.0:
            zeta_val = r.x[zeta_s]
            s_vals = r.x[s_s:s_s+n_obs]
            zeta_by_hour[hour] = zeta_val
            n_active_tail_by_hour[hour] = int(np.sum(s_vals > 1e-6))
        obj_terms_by_hour.append(dict(sc_min=sc.min(), sc_max=sc.max(),
                                       yc_min=yc.min(), yc_max=yc.max(),
                                       obj_zeta=W1*lam*n_obs, obj_s=W1*lam/(1.0-CVAR_ALPHA)))

    fc = np.zeros((n_test_days, HOURS_PER_DAY))
    prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return fc.flatten(), zeta_by_hour, n_active_tail_by_hour, obj_terms_by_hour


def run_case(label, weight_correction, lam):
    print(f"\n--- {label} (weight_correction={weight_correction}, lam={lam}) ---")
    pred, zeta_h, n_active_h, obj_terms = solve_ar_proposed_cvar_debug(
        KPI_RATE, KPI_W1, KPI_W2, lam, weight_correction)
    n = nrmse(pred, actual_flat_all)
    g = compute_gap_4term(pred, KPI_RATE, actual_flat_all, da_flat_all, rt_flat_all)
    diag = compute_regret_diagnostics(pred, KPI_RATE, actual_flat_all, da_flat_all, rt_flat_all)
    print(f"  nRMSE={n:.4f}%  Gap={g:.4f}%  TotalRegret={diag['total_regret']:.2f}  "
          f"CVaR90={diag['cvar90_regret']:.2f}  MaxRegret={diag['max_regret']:.2f}")
    if lam > 0.0:
        print(f"  시간대별 objective 계수 범위(1개 대표=hour0): sc=[{obj_terms[0]['sc_min']:.2f},{obj_terms[0]['sc_max']:.2f}] "
              f"yc=[{obj_terms[0]['yc_min']:.2f},{obj_terms[0]['yc_max']:.2f}] "
              f"obj_zeta={obj_terms[0]['obj_zeta']:.4f} obj_s(각 표본당)={obj_terms[0]['obj_s']:.4f}")
        print(f"  시간대별 ζ(zeta): {np.round(zeta_h, 3).tolist()}")
        print(f"  시간대별 tail 활성 표본 수(s_i>0): {n_active_h.tolist()} / (시간당 n_obs={n_ar_rows})")
    return n, g, diag


print("\n" + "#" * 70)
print("  (A) 방법7 그대로 — W2_eff = W2*4 (가중치 보정 있음)")
print("#" * 70)
nA0, gA0, dA0 = run_case("A-lam0", True, 0.0)
nA5, gA5, dA5 = run_case("A-lam0.05", True, 0.05)
print(f"\n  [A] λ=0→0.05 변화: ΔnRMSE={nA5-nA0:+.4f}%p  ΔGap={gA5-gA0:+.4f}%p  "
      f"ΔTotalRegret={100*(dA5['total_regret']-dA0['total_regret'])/abs(dA0['total_regret']):+.3f}%  "
      f"ΔCVaR90={100*(dA5['cvar90_regret']-dA0['cvar90_regret'])/abs(dA0['cvar90_regret']):+.3f}%")

print("\n" + "#" * 70)
print("  (B) 원본 참고코드처럼 — W2_eff = W2 (가중치 보정 없음, §10.2 단계)")
print("#" * 70)
nB0, gB0, dB0 = run_case("B-lam0", False, 0.0)
nB5, gB5, dB5 = run_case("B-lam0.05", False, 0.05)
print(f"\n  [B] λ=0→0.05 변화: ΔnRMSE={nB5-nB0:+.4f}%p  ΔGap={gB5-gB0:+.4f}%p  "
      f"ΔTotalRegret={100*(dB5['total_regret']-dB0['total_regret'])/abs(dB0['total_regret']):+.3f}%  "
      f"ΔCVaR90={100*(dB5['cvar90_regret']-dB0['cvar90_regret'])/abs(dB0['cvar90_regret']):+.3f}%")

print("\n" + "#" * 70)
print("  (C) 보정 있음(A) 상태에서 λ를 훨씬 크게 — CVaR가 W2_eff=80을 이길 수 있는가?")
print("#" * 70)
for lam_big in [0.5, 2.0, 5.0, 10.0]:
    n_big, g_big, d_big = run_case(f"A-lam{lam_big}", True, lam_big)
    print(f"  [A,λ={lam_big}] λ=0 대비: ΔnRMSE={n_big-nA0:+.4f}%p  ΔGap={g_big-gA0:+.4f}%p  "
          f"ΔTotalRegret={100*(d_big['total_regret']-dA0['total_regret'])/abs(dA0['total_regret']):+.3f}%  "
          f"ΔCVaR90={100*(d_big['cvar90_regret']-dA0['cvar90_regret'])/abs(dA0['cvar90_regret']):+.3f}%")

print("\n" + "=" * 70)
print("  결론 요약은 아래 로그 파일 전체를 보고 판단할 것:")
print(f"  {LOG_PATH}")
print("=" * 70)

logfile.close()
