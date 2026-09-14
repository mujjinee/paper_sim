# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_2cand_vs_3cand_oracle_method8.py
#
# 검증: 방법8(옛 방법2 Fine Tune, PC=RT+rate*DA)의 Gap 계산이
# 오라클 2후보{0,actual}만 쓰는데, rate=0(또는 낮은 rate)이면 shortage
# 기울기(DA-PC)가 surplus 기울기(DA-RT)와 같아지거나(=rate=0) 그보다
# 덜 줄어들어서(=rate 낮음), DA>RT인 시간대는 commit=1이 진짜 오라클이
# 될 수 있는지 실측 확인한다.
#
# 1. 대수적으로: rate=0일 때 두 구간 기울기가 정확히 DA-RT로 같아지는지 확인.
# 2. z03/블록18 테스트셋에서 DA>RT 시간대 비율.
# 3. 실제 방법8 스크립트(방법2_4term_ar_mlr_z03_block18.py)와 동일한
#    solve_ar_proposed()로 KPI 지점(rate=0.5, W1=1, W2=20->W2_eff=80)과
#    rate=0 지점(W1=1, W2=1->W2_eff=4, Fig5/8과 동일 조건)을 실제로 풀어서
#    나온 진짜 예측치(pred_flat)에 대해, 2후보 vs 3후보 오라클로 Gap을
#    각각 계산해 비교한다.
# =====================================================================
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, linprog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")

LOCAL_HOUR_START, LOCAL_HOUR_END, HOURS_PER_DAY = 9, 21, 12
TRAIN_START = pd.Timestamp("2013-08-25"); TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23"); TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")
CAPACITY_MW = 30.0; DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS
W2_BALANCE_WEIGHT = 4

# ---------------------------------------------------------------
# 1. 대수적 확인 (rate=0에서 두 구간 기울기가 같아지는지)
# ---------------------------------------------------------------
print("=" * 70)
print("1. 대수적 확인 — rate=0일 때 PC=RT+rate*DA=RT, 구간별 기울기")
print("=" * 70)
DA_ex, RT_ex = 50.0, 20.0  # 예시: DA>RT인 전형적 시간대
for rate_ex in [0.0, 0.1, 0.3, 0.5]:
    PC_ex = RT_ex + rate_ex * DA_ex
    slope_surplus = DA_ex - RT_ex          # x<S 구간 기울기 (DA*x + RT*(S-x))' = DA-RT
    slope_shortage = DA_ex - PC_ex          # x>S 구간 기울기 (DA*x - PC*(x-S))' = DA-PC
    print(f"  rate={rate_ex:.1f}: PC={PC_ex:.1f}  surplus 기울기(DA-RT)={slope_surplus:.1f}  "
          f"shortage 기울기(DA-PC)={slope_shortage:.1f}  {'(같음! 꺾임 없음)' if abs(slope_surplus-slope_shortage)<1e-9 else ''}")

print("""
  -> rate=0이면 PC=RT이 되어 shortage 기울기(DA-PC=DA-RT)가 surplus 기울기(DA-RT)와
     정확히 같아진다(꺾이는 점이 사라짐). DA>RT인 시간대는 두 구간 다 기울기가 양수라
     x=0~1 전 구간에서 이익이 계속 증가 -> 진짜 오라클은 commit=1이다.
     rate>0이면 shortage 기울기가 surplus 기울기보다 rate*DA 만큼 작아지지만,
     그래도 DA-PC = (DA-RT) - rate*DA > 0 인 한(=rate < (DA-RT)/DA), commit=1이
     여전히 commit=actual보다 유리하다 -> 2후보 오라클은 이 구간에서도 틀린다.
""")

# ---------------------------------------------------------------
# 2. 데이터 로딩 (방법2_4term_ar_mlr_z03_block18.py와 동일)
# ---------------------------------------------------------------
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
ar_intercept = np.ones((n_ar_rows, 1))
ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])

actual_flat_all = test_solar.flatten()
da_flat_all = test_da_price.flatten()
rt_flat_all = test_rt_price.flatten()

# ---------------------------------------------------------------
# 3. z03/블록18 테스트셋 DA>RT 비율
# ---------------------------------------------------------------
print("=" * 70)
print("2. z03/블록18 테스트셋(30일x12시간=360개) DA vs RT 비교")
print("=" * 70)
n_total = len(da_flat_all)
n_da_gt_rt = int(np.sum(da_flat_all > rt_flat_all))
print(f"  DA > RT인 시간대: {n_da_gt_rt} / {n_total} = {100.0*n_da_gt_rt/n_total:.1f}%")
print(f"  DA <= RT인 시간대: {n_total - n_da_gt_rt} / {n_total} = {100.0*(n_total-n_da_gt_rt)/n_total:.1f}%")

# rate=0.5에서 shortage 기울기(DA-PC=DA-RT-0.5DA=0.5DA-RT)가 양수인(=여전히 commit=1이 유리한) 비율도 확인
rate_kpi = 0.5
shortage_slope_kpi = (da_flat_all - rt_flat_all) - rate_kpi * da_flat_all
n_positive_slope_kpi = int(np.sum(shortage_slope_kpi > 0))
print(f"  [참고] rate=0.5(KPI)에서도 shortage 기울기(DA-PC)>0인 시간대: "
      f"{n_positive_slope_kpi} / {n_total} = {100.0*n_positive_slope_kpi/n_total:.1f}%  "
      f"(이 시간대는 KPI 지점에서도 2후보 오라클이 여전히 commit=1을 놓친다)")

# ---------------------------------------------------------------
# 4. solve_ar_proposed — 방법2_4term_ar_mlr_z03_block18.py와 동일 로직
# ---------------------------------------------------------------
def solve_ar_proposed(penalty_rate, W1, W2):
    W2_eff = W2 * W2_BALANCE_WEIGHT
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]; da_h = train_da_price[:, hour]; rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows
        pc_h = rt_h + penalty_rate * da_h
        sc = -W1 * rt_h + W2_eff
        yc = W1 * pc_h + W2_eff
        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
        bin_arr = np.array(bin_list, dtype=int); n_bin = len(bin_arr)
        b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
        ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs
        n_var = n_features_ar + 3 * n_obs + n_bin
        obj = np.zeros(n_var)
        obj[x_s:x_s + n_obs] = -W1 * da_h
        obj[yp_s:yp_s + n_obs] = sc
        obj[ym_s:ym_s + n_obs] = yc
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
        if not r.success:
            raise RuntimeError(f"AR MILP failed h={hour}: {r.message}")
        coeffs[hour] = r.x[b_s:b_s+n_features_ar]
    fc = np.zeros((n_test_days, HOURS_PER_DAY))
    prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return fc.flatten()

def compute_gap_2cand(pred_flat, penalty_rate, actual, da, rt):
    sum_r = 0.0; sum_o = 0.0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]; dp = da[i]; rp = rt[i]
        pc = rp + penalty_rate * dp
        m = a - x; sp = max(m, 0); sh = max(-m, 0)
        sum_r += scale * (dp * x + rp * sp - pc * sh)
        p0 = scale * rp * a; pa = scale * dp * a
        sum_o += max(p0, pa)
    return 100.0 * (sum_o - sum_r) / sum_o, sum_o, sum_r

def compute_gap_3cand(pred_flat, penalty_rate, actual, da, rt):
    sum_r = 0.0; sum_o = 0.0
    n_p1_wins = 0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]; dp = da[i]; rp = rt[i]
        pc = rp + penalty_rate * dp
        m = a - x; sp = max(m, 0); sh = max(-m, 0)
        sum_r += scale * (dp * x + rp * sp - pc * sh)
        p0 = scale * rp * a; pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rp * s1 - pc * y1)
        best = max(p0, pa, p1)
        if p1 > max(p0, pa) + 1e-9:
            n_p1_wins += 1
        sum_o += best
    return 100.0 * (sum_o - sum_r) / sum_o, sum_o, sum_r, n_p1_wins

print("\n" + "=" * 70)
print("3. 실제 방법8 KPI 지점(rate=0.5, W1=1, W2=20->W2_eff=80) 재현")
print("=" * 70)
pred_kpi = solve_ar_proposed(0.5, 1, 20)
gap2_kpi, o2_kpi, r2_kpi = compute_gap_2cand(pred_kpi, 0.5, actual_flat_all, da_flat_all, rt_flat_all)
gap3_kpi, o3_kpi, r3_kpi, n_p1_kpi = compute_gap_3cand(pred_kpi, 0.5, actual_flat_all, da_flat_all, rt_flat_all)
print(f"  2후보 Gap = {gap2_kpi:.2f}%  (오라클합={o2_kpi:.1f}, 실현합={r2_kpi:.1f})  <- 현재 보고서 표1 값(14.55%)과 비교")
print(f"  3후보 Gap = {gap3_kpi:.2f}%  (오라클합={o3_kpi:.1f}, 실현합={r3_kpi:.1f})")
print(f"  commit=1이 진짜 오라클인 시간대: {n_p1_kpi}/{n_total} ({100.0*n_p1_kpi/n_total:.1f}%)")
print(f"  Gap 차이(3후보-2후보): {gap3_kpi-gap2_kpi:+.2f}%p")

print("\n" + "=" * 70)
print("4. rate=0 지점(Fig.5/8과 동일 조건: W1=1, W2=1->W2_eff=4) 재현 — 가장 극단적 사례")
print("=" * 70)
pred_r0 = solve_ar_proposed(0.0, 1, 1)
gap2_r0, o2_r0, r2_r0 = compute_gap_2cand(pred_r0, 0.0, actual_flat_all, da_flat_all, rt_flat_all)
gap3_r0, o3_r0, r3_r0, n_p1_r0 = compute_gap_3cand(pred_r0, 0.0, actual_flat_all, da_flat_all, rt_flat_all)
print(f"  2후보 Gap = {gap2_r0:.2f}%  (오라클합={o2_r0:.1f}, 실현합={r2_r0:.1f})")
print(f"  3후보 Gap = {gap3_r0:.2f}%  (오라클합={o3_r0:.1f}, 실현합={r3_r0:.1f})")
print(f"  commit=1이 진짜 오라클인 시간대: {n_p1_r0}/{n_total} ({100.0*n_p1_r0/n_total:.1f}%)")
print(f"  Gap 차이(3후보-2후보): {gap3_r0-gap2_r0:+.2f}%p")

print("\n" + "=" * 70)
print("5. Fig.5/8 rate 스윕 전체(W1=W2=1, W2_eff=4) — rate별 2후보 vs 3후보 Gap")
print("=" * 70)
for rate in [round(0.1*i,1) for i in range(11)]:
    pred_r = solve_ar_proposed(rate, 1, 1)
    g2, _, _ = compute_gap_2cand(pred_r, rate, actual_flat_all, da_flat_all, rt_flat_all)
    g3, _, _, np1 = compute_gap_3cand(pred_r, rate, actual_flat_all, da_flat_all, rt_flat_all)
    print(f"  rate={rate:.1f}: 2후보Gap={g2:6.2f}%  3후보Gap={g3:6.2f}%  차이={g3-g2:+6.2f}%p  "
          f"commit=1오라클 시간대={np1:3d}/{n_total}({100.0*np1/n_total:4.1f}%)")

print("\n완료")
