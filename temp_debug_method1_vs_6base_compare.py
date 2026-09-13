# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_method1_vs_6base_compare.py
#
# 사용자 질문: 방법1(paper_eq_ar_mlr_z03_block18.py)과 방법6.1(SPO+ base,
# 방법6_SPO_base_AR_MLR.py)이 "PC=rate*DA로 이익식이 같다"면, oracle_i가
# beta와 무관한 상수이므로 Σζ_i(=Σoracle_i-Σprofit_i(β)) 최소화와
# Σ(-profit_i(β)) 최소화는 같은 argmin β를 가져야 하는데 실제 결과는 왜
# 극단적으로 다른가?
#
# 이 스크립트는:
#   1. 두 스크립트의 실제 목적함수 계수(sc, yc, obj_x)를 같은 시간대·같은
#      (W1,W2,rate)에서 직접 계산해 나란히 출력한다.
#   2. 두 스크립트의 MILP을 그대로(각자의 정규화 방식 그대로) 풀어서 학습된
#      β를 직접 비교한다 — 같은 데이터, 같은 W1/W2/rate.
#   3. 결과: 두 "이익식"이 실제로는 같지 않다는 것, 그리고 정규화 방식
#      (denom-scale vs raw+weight)도 다르다는 것을 코드 수준에서 확인한다.
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
W2_BALANCE_WEIGHT = 4  # 방법6.1이 실제로 쓰는 값

# =====================================================================
# 데이터 로딩 (두 스크립트와 완전히 동일한 절차)
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

n_features_ar = 13
history_and_train = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1)); ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])

print(f"데이터: train {n_train_days}일, 시간대 {HOURS_PER_DAY}개")
print(f"비교 조건: W1={KPI_W1}, W2={KPI_W2}, rate={KPI_RATE} (KPI 지점)")
print()


# =====================================================================
# 방법1 스타일: denom(Σoracle) + scale + n_obs 정규화, PC=rate*DA, 3항, 지시함수 없음
# (paper_eq_ar_mlr_z03_block18.py의 solve_ar 그대로 재현)
# =====================================================================
def method1_hour_coeffs(hour, W1, W2, rate):
    y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
    n_obs = n_ar_rows

    oracle = np.zeros(n_obs)
    for i in range(n_obs):
        a = y_h[i]; dp = da_h[i]; rp = rt_h[i]
        pc = rate * dp
        p0 = scale * rp * a; pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rp * s1 - pc * y1)
        oracle[i] = max(p0, pa, p1)
    denom = oracle.sum()

    sc = np.zeros(n_obs); yc = np.zeros(n_obs)
    for i in range(n_obs):
        pc = rate * da_h[i]
        sc[i] = (-W1 * scale * rt_h[i] / denom) + (W2 / n_obs)
        yc[i] = (W1 * scale * pc / denom) + (W2 / n_obs)
    obj_x = -W1 * scale * da_h / denom
    return obj_x, sc, yc, denom


def method1_solve_hour(hour, W1, W2, rate):
    y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
    n_obs = n_ar_rows
    obj_x, sc, yc, denom = method1_hour_coeffs(hour, W1, W2, rate)

    bl = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    ba = np.array(bl, dtype=int); nb = len(ba)
    bs = 0; xs = n_features_ar; yps = n_features_ar + n_obs
    yms = n_features_ar + 2*n_obs; zs = n_features_ar + 3*n_obs
    nv = n_features_ar + 3*n_obs + nb

    obj = np.zeros(nv)
    obj[xs:xs+n_obs] = obj_x
    obj[yps:yps+n_obs] = sc
    obj[yms:yms+n_obs] = yc

    Xs = sparse.csr_matrix(ar_design); In = sparse.eye(n_obs, format="csr")
    ea = sparse.lil_matrix((n_obs, nv))
    ea[:, bs:bs+n_features_ar] = -Xs; ea[:, xs:xs+n_obs] = In
    eb = sparse.lil_matrix((n_obs, nv))
    eb[:, xs:xs+n_obs] = In; eb[:, yps:yps+n_obs] = In; eb[:, yms:yms+n_obs] = -In
    all_eq = sparse.vstack([ea, eb], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

    if nb > 0:
        comp = sparse.lil_matrix((2*nb, nv))
        for k in range(nb):
            r = ba[k]
            comp[k, yps+r] = 1.0; comp[k, zs+k] = 1.0
            comp[nb+k, yms+r] = 1.0; comp[nb+k, zs+k] = -1.0
        all_con.append(LinearConstraint(comp.tocsr(), np.full(2*nb, -np.inf),
                      np.concatenate([np.ones(nb), np.zeros(nb)])))

    lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+nb)])
    ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+nb)])
    integ = np.zeros(nv, dtype=int); integ[zs:zs+nb] = 1
    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con,
             options={"mip_rel_gap": 1e-9})
    if not r.success:
        raise RuntimeError(f"방법1 MILP 실패 h={hour}: {r.message}")
    return r.x[:n_features_ar]


# =====================================================================
# 방법6.1 스타일: raw 계수 + W2_BALANCE_WEIGHT, PC=rate*DA, indicator I[DA>RT] 조건부,
# shortage cost에 RT 무조건 추가 (방법6_SPO_base_AR_MLR.py의 solve_ar_spo_base 그대로)
# =====================================================================
def method6base_hour_coeffs(hour, W1, W2, rate):
    da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
    W2_eff = W2 * W2_BALANCE_WEIGHT
    pc_h = rate * da_h
    indicator = (da_h > rt_h).astype(float)
    sc = -W1 * rt_h + W2_eff
    yc = W1 * rt_h + W1 * pc_h * indicator + W2_eff
    obj_x = -W1 * da_h
    return obj_x, sc, yc, indicator


def method6base_solve_hour(hour, W1, W2, rate):
    y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
    n_obs = n_ar_rows
    obj_x, sc, yc, indicator = method6base_hour_coeffs(hour, W1, W2, rate)

    bl = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    ba = np.array(bl, dtype=int); nb = len(ba)
    bs = 0; xs = n_features_ar; yps = n_features_ar + n_obs
    yms = n_features_ar + 2*n_obs; zs = n_features_ar + 3*n_obs
    nv = n_features_ar + 3*n_obs + nb

    obj = np.zeros(nv)
    obj[xs:xs+n_obs] = obj_x
    obj[yps:yps+n_obs] = sc
    obj[yms:yms+n_obs] = yc

    Xs = sparse.csr_matrix(ar_design); In = sparse.eye(n_obs, format="csr")
    ea = sparse.lil_matrix((n_obs, nv))
    ea[:, bs:bs+n_features_ar] = -Xs; ea[:, xs:xs+n_obs] = In
    eb = sparse.lil_matrix((n_obs, nv))
    eb[:, xs:xs+n_obs] = In; eb[:, yps:yps+n_obs] = In; eb[:, yms:yms+n_obs] = -In
    all_eq = sparse.vstack([ea, eb], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

    if nb > 0:
        comp = sparse.lil_matrix((2*nb, nv))
        for k in range(nb):
            r = ba[k]
            comp[k, yps+r] = 1.0; comp[k, zs+k] = 1.0
            comp[nb+k, yms+r] = 1.0; comp[nb+k, zs+k] = -1.0
        all_con.append(LinearConstraint(comp.tocsr(), np.full(2*nb, -np.inf),
                      np.concatenate([np.ones(nb), np.zeros(nb)])))

    lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+nb)])
    ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+nb)])
    integ = np.zeros(nv, dtype=int); integ[zs:zs+nb] = 1
    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=all_con,
             options={"mip_rel_gap": 1e-9})
    if not r.success:
        raise RuntimeError(f"방법6.1 MILP 실패 h={hour}: {r.message}")
    return r.x[:n_features_ar]


# =====================================================================
# 1. 목적함수 계수 직접 비교 (hour=0, 6 두 시간대 대표로)
# =====================================================================
print("=" * 78)
print("1. 목적함수 계수 비교 (같은 시간대, 같은 W1/W2/rate)")
print("=" * 78)
for hour in [0, 6]:
    da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
    obj_x1, sc1, yc1, denom1 = method1_hour_coeffs(hour, KPI_W1, KPI_W2, KPI_RATE)
    obj_x6, sc6, yc6, ind6 = method6base_hour_coeffs(hour, KPI_W1, KPI_W2, KPI_RATE)
    print(f"\n--- 시간대 {hour} (DA 평균={da_h.mean():.2f}, RT 평균={rt_h.mean():.2f}, "
          f"DA>RT 비율={ind6.mean()*100:.1f}%) ---")
    print(f"  [방법1]  denom(Σoracle)={denom1:.1f}")
    print(f"           obj_x  범위=[{obj_x1.min():.6f}, {obj_x1.max():.6f}]")
    print(f"           sc(surplus) 범위=[{sc1.min():.6f}, {sc1.max():.6f}]")
    print(f"           yc(shortage) 범위=[{yc1.min():.6f}, {yc1.max():.6f}]")
    print(f"  [방법6.1] obj_x  범위=[{obj_x6.min():.4f}, {obj_x6.max():.4f}]")
    print(f"           sc(surplus) 범위=[{sc6.min():.4f}, {sc6.max():.4f}]")
    print(f"           yc(shortage) 범위=[{yc6.min():.4f}, {yc6.max():.4f}]")
    print(f"  → 방법1 계수는 denom(~{denom1:.0f})로 나눠서 방법6.1보다 대략 "
          f"{abs(yc6.mean())/max(abs(yc1.mean()),1e-12):.0f}배 작다 (자릿수 자체가 다름)")

# =====================================================================
# 2. 실제 학습된 β 비교 (전체 12시간대)
# =====================================================================
print("\n" + "=" * 78)
print("2. 실제 학습된 β(AR 계수) 비교 — 전체 12시간대, 절편+lag12")
print("=" * 78)

beta1 = np.zeros((HOURS_PER_DAY, n_features_ar))
beta6 = np.zeros((HOURS_PER_DAY, n_features_ar))
for h in range(HOURS_PER_DAY):
    beta1[h] = method1_solve_hour(h, KPI_W1, KPI_W2, KPI_RATE)
    beta6[h] = method6base_solve_hour(h, KPI_W1, KPI_W2, KPI_RATE)

diff = beta1 - beta6
print(f"\n최대 |β1-β6.1| = {np.max(np.abs(diff)):.6f}  (시간대 {np.unravel_index(np.argmax(np.abs(diff)), diff.shape)[0]}, "
      f"계수 인덱스 {np.unravel_index(np.argmax(np.abs(diff)), diff.shape)[1]})")
print(f"평균 |β1-β6.1| = {np.mean(np.abs(diff)):.6f}")
print(f"\n시간대별 절편(β0) 비교:")
for h in range(HOURS_PER_DAY):
    print(f"  h={h:2d}: 방법1 β0={beta1[h,0]:+9.4f}   방법6.1 β0={beta6[h,0]:+9.4f}   차이={beta1[h,0]-beta6[h,0]:+9.4f}")

print(f"\n결론: 두 방법의 β가 사실상 완전히 다르다 — 최대 차이 {np.max(np.abs(diff)):.4f}는")
print("우연한 수치오차(1e-9 수준)가 아니라 애초에 서로 다른 최적화 문제를 풀었다는 뜻이다.")
