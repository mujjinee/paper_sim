
# -*- coding: utf-8 -*-

# =====================================================================

# zhang_asymmetric_3term_integrated.py

#

# 방법 3: Zhang et al. 방식 — ρ₊ ≠ ρ₋ 비대칭 시장가격 모델

# AR(시간대별 12개 모델) + MLR(pooled 1개 모델) 통합 실행

#

# 이익함수: DA·x + ρ₋·surplus − ρ₊·shortage

#   ρ₋ = RT (초과 시 하향조정가격)

#   ρ₊ = c·RT (부족 시 상향조정가격, c > 1)

#   oracle: 3후보 {0, actual, 1.0}

#

# 데이터: merged_for_simulation_z03.csv (블록18)

# 출력:

#   - 전체 그리드 (c 5개 x W1/W2 10개 = 50조합, AR+MLR)

#   - Best c 자동탐색

#   - 논문 KPI 조건 (c=1.5, W1=1, W2=20) 별도 print

#   - Fig.3(AR dual y), Fig.5(AR c 스윕),

#     Fig.6(MLR dual y), Fig.7(MLR c 스윕)

# =====================================================================



import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np

import pandas as pd

from scipy import sparse

from scipy.optimize import Bounds, LinearConstraint, milp, linprog

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

import csv





# =====================================================================

# 0. 한글 폰트 + matplotlib 스타일

# =====================================================================

korean_font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic"]

found_font_name = None

for font in fm.fontManager.ttflist:

    if font.name in korean_font_candidates:

        found_font_name = font.name

        break

if found_font_name:

    plt.rcParams["font.family"] = found_font_name

plt.rcParams["axes.unicode_minus"] = False



# ── 일관된 색상 팔레트 ──

C_AR_BASE   = "#4C72B0"   # AR baseline (파랑)

C_AR_PROP   = "#55A868"   # AR 제안모형 (초록)

C_MLR_BASE  = "#CC4654"   # MLR baseline (빨강)

C_MLR_PROP  = "#8CAED6"   # MLR 제안모형 (연파랑)

C_PAPER     = "#888888"   # 논문 참고값 (회색)

C_GRID      = "#DDDDDD"   # 그리드선





# =====================================================================

# 1. 설정 — z03/블록18

# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

OUT_DIR = os.path.join(RESULTS_DIR, "simulation_output")

os.makedirs(OUT_DIR, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)



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



# ── 논문 KPI 조건 ──

KPI_C = 1.5

KPI_W1 = 1

KPI_W2 = 20



# ── 그리드 ──

W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1),

            (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]

C_RATES = [1.0, 1.2, 1.5, 2.0, 3.0]



# ── Fig.3/6: W1/W2 라벨 (AR baseline 포함 11개점) ──

FIG_LABELS = ["AR/MLR", "1/20", "1/10", "1/5", "1/2",

              "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]



# ── Fig.5/7: W1=W2=1 고정, c 11개점 스윕 ──

FIG5_W1, FIG5_W2 = 1, 1

FIG_C_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]



# ── 논문 참고값 ──

PAPER_AR_BASE_NRMSE = 34.76

PAPER_AR_BASE_GAP = 15.04

PAPER_MLR_BASE_NRMSE = 21.76

PAPER_MLR_BASE_GAP = 12.59



# AR 제안모형 W1/W2 스윕 논문값 (Fig.3)

PAPER_AR_PROP_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95,

                       46.11, 48.27, 49.21, 49.61, 50.07]

PAPER_AR_PROP_GAP = [13.91, 13.42, 12.71, 11.88, 11.44,

                     11.38, 11.38, 11.36, 11.36, 11.36]



# MLR 제안모형 KPI (W1/W2=1/20 지점) — 논문 Table 4

PAPER_MLR_PROP_NRMSE_KPI = 21.92

PAPER_MLR_PROP_GAP_KPI = 11.91

# MLR 스윕값은 논문에 없으므로 AR 스윕값을 참고선으로 사용

PAPER_MLR_PROP_NRMSE = PAPER_AR_PROP_NRMSE

PAPER_MLR_PROP_GAP = PAPER_AR_PROP_GAP





# =====================================================================

# 2. 데이터 로딩 + 낮 시간대 필터

# =====================================================================

print("=" * 70)

print("  1. 데이터 로딩")

print("=" * 70)

raw_table = pd.read_csv(MERGED_FILE)

raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])



is_daylight = ((raw_table["local_hour"] >= LOCAL_HOUR_START) &
               (raw_table["local_hour"] < LOCAL_HOUR_END))

daylight_table = raw_table[is_daylight].copy()

daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START



# ── AR용: (날짜 x 12) 배열 ──

is_history = daylight_table["local_date"] == HISTORY_DATE

history_rows = daylight_table[is_history].copy().sort_values("hour_idx")

history_solar = np.zeros((1, HOURS_PER_DAY))

for _, row in history_rows.iterrows():

    history_solar[0, row["hour_idx"]] = row["solar_power"]



is_train = ((daylight_table["local_date"] >= TRAIN_START) &
            (daylight_table["local_date"] <= TRAIN_END))

train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])



is_test = ((daylight_table["local_date"] >= TEST_START) &
           (daylight_table["local_date"] <= TEST_END))

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



# ── MLR용: 1차원(flat) 배열 ──

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

X_mlr_train = np.column_stack([np.ones(n_train_obs),

                                mlr_train_dssrd, mlr_train_dtsr, mlr_train_hour])

X_mlr_test = np.column_stack([np.ones(n_test_obs),

                               test_rows["dssrd"].to_numpy(),

                               test_rows["dtsr"].to_numpy(),

                               test_rows["hour_idx"].to_numpy(dtype=float)])



# ── AR용: 설계행렬(절편 + 직전 하루 12시간 lag 역순) ──

n_features_ar = 13

history_and_train = np.vstack([history_solar, train_solar])

n_ar_rows = n_train_days

ar_intercept = np.ones((n_ar_rows, 1))

ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))

for d in range(n_ar_rows):

    ar_lag[d] = history_and_train[d][::-1]

ar_design = np.hstack([ar_intercept, ar_lag])



# ── 테스트 flat (AR용) ──

actual_flat_all = test_solar.flatten()

da_flat_all = test_da_price.flatten()

rt_flat_all = test_rt_price.flatten()

n_test_obs_all = len(actual_flat_all)



print(f"  train: {n_train_days}일({n_train_obs}행), test: {n_test_days}일({n_test_obs}행)")





# =====================================================================

# 3. Gap / nRMSE 계산 함수 — Zhang 3항

# =====================================================================

def compute_zhang_gap(pred_flat, c_rate, actual, da, rt):

    """Zhang 3항 이익함수 optimality gap (oracle: {0, actual, 1.0})"""

    sum_realized = 0.0

    sum_oracle = 0.0

    for i in range(len(pred_flat)):

        a = actual[i]; x = pred_flat[i]

        dp = da[i]; rp = rt[i]

        rho_plus = c_rate * rp

        rho_minus = rp



        mismatch = a - x

        surplus = max(mismatch, 0)

        shortage = max(-mismatch, 0)

        realized = scale * (dp * x + rho_minus * surplus - rho_plus * shortage)

        sum_realized += realized



        p0 = scale * rho_minus * a

        pa = scale * dp * a

        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)

        p1 = scale * (dp * 1.0 + rho_minus * s1 - rho_plus * y1)

        oracle = max(p0, pa, p1)

        sum_oracle += oracle



    return 100.0 * (sum_oracle - sum_realized) / sum_oracle if sum_oracle > 1e-10 else 0.0





def calc_nrmse(pred_flat, actual):

    rmse = np.sqrt(np.mean((actual - pred_flat) ** 2))

    return 100.0 * rmse / np.mean(actual)





# =====================================================================

# 4. AR baseline — bounded LAD (시간대별 12개)

# =====================================================================

print("\n" + "=" * 70)

print("  2. AR baseline (bounded LAD, 시간대별)")

print("=" * 70)

ar_coefficients = np.zeros((HOURS_PER_DAY, n_features_ar))

for h in range(HOURS_PER_DAY):

    y_h = train_solar[:, h]

    X_sp = sparse.csr_matrix(ar_design)

    I_n = sparse.eye(n_ar_rows, format="csr")

    A = sparse.vstack([sparse.hstack([X_sp, -I_n]),

                       sparse.hstack([-X_sp, -I_n])], format="csr")

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

ar_nrmse = calc_nrmse(ar_pred_flat, actual_flat_all)

print(f"  AR baseline nRMSE = {ar_nrmse:.2f}%")





# =====================================================================

# 5. MLR baseline — bounded LAD (pooled 1개)

# =====================================================================

print("\n" + "=" * 70)

print("  3. MLR baseline (bounded LAD, pooled)")

print("=" * 70)

X_sp_mlr = sparse.csr_matrix(X_mlr_train)

I_mlr = sparse.eye(n_train_obs, format="csr")

A_mlr = sparse.vstack([sparse.hstack([X_sp_mlr, -I_mlr]),

                        sparse.hstack([-X_sp_mlr, -I_mlr])], format="csr")

b_mlr = np.concatenate([mlr_train_solar, -mlr_train_solar])

c_mlr = np.concatenate([np.zeros(n_features_mlr), np.ones(n_train_obs) / n_train_obs])

vb_mlr = [(None, None)] * n_features_mlr + [(0.0, None)] * n_train_obs

res_mlr = linprog(c_mlr, A_ub=A_mlr, b_ub=b_mlr, bounds=vb_mlr, method="highs")

mlr_coefficients = res_mlr.x[:n_features_mlr]



mlr_pred_flat = np.clip(X_mlr_test @ mlr_coefficients, 0, 1)

mlr_nrmse = calc_nrmse(mlr_pred_flat, actual_flat)

print(f"  MLR baseline nRMSE = {mlr_nrmse:.2f}%")





# =====================================================================

# 6. 제안모형 MILP — AR (시간대별, Zhang 3항)

# =====================================================================

def solve_ar_proposed(c_rate, W1, W2):

    """AR 제안모형: Zhang 3항, 12시간대별 MILP 풀기"""

    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))



    for hour in range(HOURS_PER_DAY):

        y_h = train_solar[:, hour]

        da_h = train_da_price[:, hour]

        rt_h = train_rt_price[:, hour]

        n_obs = n_ar_rows



        rho_plus_h = c_rate * rt_h

        rho_minus_h = rt_h



        # training oracle: 3후보 {0, actual, 1.0}

        oracle_train = np.zeros(n_obs)

        for i in range(n_obs):

            a = y_h[i]; dp = da_h[i]

            r_p = rho_plus_h[i]; r_m = rho_minus_h[i]

            p0 = scale * r_m * a; pa = scale * dp * a

            s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)

            p1 = scale * (dp * 1.0 + r_m * s1 - r_p * y1)

            oracle_train[i] = max(p0, pa, p1)

        denom = oracle_train.sum()



        # 목적함수 계수

        sc = np.zeros(n_obs)  # surplus cost

        yc = np.zeros(n_obs)  # shortage cost

        for i in range(n_obs):

            sc[i] = (-W1 * scale * rho_minus_h[i] / denom) + (W2 / n_obs)

            yc[i] = (W1 * scale * rho_plus_h[i] / denom) + (W2 / n_obs)



        # 이진변수

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

        eq_a[:, b_s:b_s+n_features_ar] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n

        eq_b = sparse.lil_matrix((n_obs, n_var))

        eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n

        eq_b[:, ym_s:ym_s+n_obs] = -I_n

        all_eq = sparse.vstack([eq_a, eq_b], format="csr")

        eq_rhs = np.concatenate([np.zeros(n_obs), y_h])

        all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]



        if n_bin > 0:

            comp = sparse.lil_matrix((2 * n_bin, n_var))

            for k in range(n_bin):

                r = bin_arr[k]

                comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0

                comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0

            all_con.append(LinearConstraint(comp.tocsr(),

                          np.full(2*n_bin, -np.inf),

                          np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))



        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin)])

        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin)])

        integ = np.zeros(n_var, dtype=int)

        integ[z_s:z_s+n_bin] = 1



        r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),

                 constraints=all_con, options={"mip_rel_gap": 1e-9})

        if not r.success:

            raise RuntimeError(f"AR MILP failed h={hour}: {r.message}")

        coeffs[hour] = r.x[b_s:b_s+n_features_ar]



    # 예측 (rolling)

    fc = np.zeros((n_test_days, HOURS_PER_DAY))

    prev = train_solar[-1]

    for d in range(n_test_days):

        fv = np.concatenate([[1.0], prev[::-1]])

        for h in range(HOURS_PER_DAY):

            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)

        prev = test_solar[d]

    return fc.flatten()





# =====================================================================

# 7. 제안모형 MILP — MLR (pooled, Zhang 3항)

# =====================================================================

def solve_mlr_proposed(c_rate, W1, W2):

    """MLR 제안모형: Zhang 3항, pooled MILP 하나만 풀기"""

    n_obs = n_train_obs



    rho_plus = c_rate * mlr_train_rt

    rho_minus = mlr_train_rt



    # training oracle: 3후보 {0, actual, 1.0}

    oracle_train = np.zeros(n_obs)

    for i in range(n_obs):

        a = mlr_train_solar[i]; dp = mlr_train_da[i]

        r_p = rho_plus[i]; r_m = rho_minus[i]

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

    bin_arr = np.array(bin_list, dtype=int)

    n_bin = len(bin_arr)



    b_s = 0; x_s = n_features_mlr; yp_s = n_features_mlr + n_obs

    ym_s = n_features_mlr + 2 * n_obs; z_s = n_features_mlr + 3 * n_obs

    n_var = n_features_mlr + 3 * n_obs + n_bin



    obj = np.zeros(n_var)

    for i in range(n_obs):

        obj[x_s + i] = -W1 * scale * mlr_train_da[i] / denom

        obj[yp_s + i] = sc[i]; obj[ym_s + i] = yc[i]



    X_sp = sparse.csr_matrix(X_mlr_train)

    I_n = sparse.eye(n_obs, format="csr")

    eq_a = sparse.lil_matrix((n_obs, n_var))

    eq_a[:, b_s:b_s+n_features_mlr] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n

    eq_b = sparse.lil_matrix((n_obs, n_var))

    eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n

    eq_b[:, ym_s:ym_s+n_obs] = -I_n

    all_eq = sparse.vstack([eq_a, eq_b], format="csr")

    eq_rhs = np.concatenate([np.zeros(n_obs), mlr_train_solar])

    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]



    if n_bin > 0:

        comp = sparse.lil_matrix((2 * n_bin, n_var))

        for k in range(n_bin):

            r = bin_arr[k]

            comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0

            comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0

        all_con.append(LinearConstraint(comp.tocsr(),

                      np.full(2*n_bin, -np.inf),

                      np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))



    lb = np.concatenate([np.full(n_features_mlr, -np.inf), np.zeros(3*n_obs+n_bin)])

    ub = np.concatenate([np.full(n_features_mlr, np.inf), np.ones(3*n_obs+n_bin)])

    integ = np.zeros(n_var, dtype=int)

    integ[z_s:z_s+n_bin] = 1



    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),

             constraints=all_con, options={"mip_rel_gap": 1e-9})

    if not r.success:

        raise RuntimeError(f"MLR MILP failed: {r.message}")

    coeffs = r.x[b_s:b_s+n_features_mlr]

    return np.clip(X_mlr_test @ coeffs, 0, 1)





# =====================================================================

# 8. 캐시

# =====================================================================

cache_ar = {}

cache_mlr = {}



def get_ar(cr, w1, w2):

    k = (cr, w1, w2)

    if k not in cache_ar:

        pred = solve_ar_proposed(cr, w1, w2)

        cache_ar[k] = (calc_nrmse(pred, actual_flat_all),

                       compute_zhang_gap(pred, cr, actual_flat_all, da_flat_all, rt_flat_all))

    return cache_ar[k]



def get_mlr(cr, w1, w2):

    k = (cr, w1, w2)

    if k not in cache_mlr:

        pred = solve_mlr_proposed(cr, w1, w2)

        cache_mlr[k] = (calc_nrmse(pred, actual_flat),

                        compute_zhang_gap(pred, cr, actual_flat, da_flat, rt_flat))

    return cache_mlr[k]





# =====================================================================

# 9. 전체 그리드 스윕 (c 5개 x W1/W2 10개 = 50조합, AR+MLR)

# =====================================================================

print("\n" + "=" * 70)

print("  4. 그리드 스윕 (c 5개 x W1/W2 10개 = 50조합, AR+MLR)")

print("=" * 70)



all_results = []



for c_rate in C_RATES:

    print(f"\n{'=' * 60}")

    print(f"  c = {c_rate}  (rho_plus = {c_rate}*RT, rho_minus = RT)")

    print(f"{'=' * 60}")



    results_c = []

    for widx, (W1, W2) in enumerate(W_RATIOS):

        a_n, a_g = get_ar(c_rate, W1, W2)

        m_n, m_g = get_mlr(c_rate, W1, W2)

        label = f"{W1}/{W2}"

        results_c.append((label, W1, W2, a_n, a_g, m_n, m_g))

        all_results.append((c_rate, label, W1, W2, a_n, a_g, m_n, m_g))

        print(f"  {label}: AR(nRMSE={a_n:.2f}%, Gap={a_g:.2f}%) "

              f"MLR(nRMSE={m_n:.2f}%, Gap={m_g:.2f}%) "

              f"(논문: nRMSE={PAPER_AR_PROP_NRMSE[widx]:.2f}%, Gap={PAPER_AR_PROP_GAP[widx]:.2f}%)")



print("\n" + "=" * 110)

print(f"{'c':>4} {'Label':>6} {'W1':>4} {'W2':>4} "

      f"{'AR nRMSE':>10} {'AR Gap':>10} "

      f"{'MLR nRMSE':>10} {'MLR Gap':>10}")

print("-" * 110)

for c_rate, label, W1, W2, a_n, a_g, m_n, m_g in all_results:

    print(f"{c_rate:>4} {label:>6} {W1:>4} {W2:>4} "

          f"{a_n:>9.2f}% {a_g:>9.2f}% "

          f"{m_n:>9.2f}% {m_g:>9.2f}%")

print("=" * 110)



# AR/MLR baseline Gap

print(f"\n  baseline Gap (c별):")

for c_rate in C_RATES:

    ar_g = compute_zhang_gap(ar_pred_flat, c_rate,

                             actual_flat_all, da_flat_all, rt_flat_all)

    mlr_g = compute_zhang_gap(mlr_pred_flat, c_rate,

                              actual_flat, da_flat, rt_flat)

    print(f"  c={c_rate}: AR Gap={ar_g:.2f}% (논문 {PAPER_AR_BASE_GAP}%), "

          f"MLR Gap={mlr_g:.2f}% (논문 {PAPER_MLR_BASE_GAP}%)")



# 그리드 CSV

grid_csv = os.path.join(OUT_DIR, "grid_zhang_asym_3term_ar_mlr.csv")

with open(grid_csv, "w", newline="") as f:

    w = csv.writer(f)

    w.writerow(["c_rate", "Label", "W1", "W2",

                "AR_nRMSE", "AR_Gap", "MLR_nRMSE", "MLR_Gap",

                "Paper_nRMSE", "Paper_Gap"])

    for c_rate, label, W1, W2, a_n, a_g, m_n, m_g in all_results:

        idx = W_RATIOS.index((W1, W2))

        w.writerow([c_rate, label, W1, W2,

                     round(a_n, 2), round(a_g, 2),

                     round(m_n, 2), round(m_g, 2),

                     PAPER_AR_PROP_NRMSE[idx], PAPER_AR_PROP_GAP[idx]])

print(f"\n  saved: {grid_csv}")





# =====================================================================

# 10. 논문 KPI 조건 (c=1.5, W1=1, W2=20) 별도 출력

# =====================================================================

print("\n" + "=" * 70)

print(f"  5. 논문 KPI 조건 — c={KPI_C}, W1={KPI_W1}, W2={KPI_W2}")

print("=" * 70)



ar_kpi_n, ar_kpi_g = get_ar(KPI_C, KPI_W1, KPI_W2)

mlr_kpi_n, mlr_kpi_g = get_mlr(KPI_C, KPI_W1, KPI_W2)

ar_kpi_gap_base = compute_zhang_gap(ar_pred_flat, KPI_C,

                                     actual_flat_all, da_flat_all, rt_flat_all)

mlr_kpi_gap_base = compute_zhang_gap(mlr_pred_flat, KPI_C,

                                      actual_flat, da_flat, rt_flat)



# KPI 조건에서论文 대응 인덱스 (W1/W2=1/20 → idx=0)

kpi_idx = W_RATIOS.index((KPI_W1, KPI_W2))



print(f"""



  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐

  │     모델          │  nRMSE (%)   │  Gap (%)     │  nRMSE (%)   │  Gap (%)     │

  │                  │  (제안모형)   │  (제안모형)   │  (baseline)  │  (baseline)  │

  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤

  │  AR              │ {ar_kpi_n:>10.2f}   │ {ar_kpi_g:>10.2f}   │ {ar_nrmse:>10.2f}   │ {ar_kpi_gap_base:>10.2f}   │

  │  MLR             │ {mlr_kpi_n:>10.2f}   │ {mlr_kpi_g:>10.2f}   │ {mlr_nrmse:>10.2f}   │ {mlr_kpi_gap_base:>10.2f}   │

  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘



  ※ 논문 KPI 기준값 (W1/W2=1/20 지점):

      AR  : nRMSE={PAPER_AR_PROP_NRMSE[kpi_idx]}%, Gap={PAPER_AR_PROP_GAP[kpi_idx]}%

      MLR : nRMSE={PAPER_AR_PROP_NRMSE[kpi_idx]}%, Gap={PAPER_AR_PROP_GAP[kpi_idx]}%

      (baseline) AR nRMSE={PAPER_AR_BASE_NRMSE}%, Gap={PAPER_AR_BASE_GAP}%

      (baseline) MLR nRMSE={PAPER_MLR_BASE_NRMSE}%, Gap={PAPER_MLR_BASE_GAP}%



  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐

  │     모델          │  Δ nRMSE     │  Δ Gap       │  Δ nRMSE     │  Δ Gap       │

  │                  │  (제안모형)   │  (제안모형)   │  (baseline)  │  (baseline)  │

  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤

  │  AR              │ {ar_kpi_n-PAPER_AR_PROP_NRMSE[kpi_idx]:>+10.2f}%p │ {ar_kpi_g-PAPER_AR_PROP_GAP[kpi_idx]:>+10.2f}%p │ {ar_nrmse-PAPER_AR_BASE_NRMSE:+>10.2f}%p │ {ar_kpi_gap_base-PAPER_AR_BASE_GAP:+>10.2f}%p │

  │  MLR             │ {mlr_kpi_n-PAPER_AR_PROP_NRMSE[kpi_idx]:>+10.2f}%p │ {mlr_kpi_g-PAPER_AR_PROP_GAP[kpi_idx]:>+10.2f}%p │ {mlr_nrmse-PAPER_MLR_BASE_NRMSE:+>10.2f}%p │ {mlr_kpi_gap_base-PAPER_MLR_BASE_GAP:+>10.2f}%p │

  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

""")





# =====================================================================

# 11. Best c 자동탐색 (|ΔnRMSE|+|ΔGap| 최소, AR+MLR 평균)

# =====================================================================

best_total_err = np.inf

best_combo = None

for c_rate, label, W1, W2, a_n, a_g, m_n, m_g in all_results:

    idx = W_RATIOS.index((W1, W2))

    ar_err = abs(a_n - PAPER_AR_PROP_NRMSE[idx]) + abs(a_g - PAPER_AR_PROP_GAP[idx])

    mlr_err = abs(m_n - PAPER_AR_PROP_NRMSE[idx]) + abs(m_g - PAPER_AR_PROP_GAP[idx])

    err = (ar_err + mlr_err) / 2

    if err < best_total_err:

        best_total_err = err

        best_combo = (c_rate, label, a_n, a_g, m_n, m_g, idx)



BEST_C = best_combo[0]

print(f"\n  Best match: c={BEST_C}, W1/W2={best_combo[1]}")

print(f"  AR  (nRMSE={best_combo[2]:.2f}%, Gap={best_combo[3]:.2f}%, "

      f"ΔnRMSE={best_combo[2]-PAPER_AR_PROP_NRMSE[best_combo[6]]:+.2f}%p, "

      f"ΔGap={best_combo[3]-PAPER_AR_PROP_GAP[best_combo[6]]:+.2f}%p)")

print(f"  MLR (nRMSE={best_combo[4]:.2f}%, Gap={best_combo[5]:.2f}%, "

      f"ΔnRMSE={best_combo[4]-PAPER_AR_PROP_NRMSE[best_combo[6]]:+.2f}%p, "

      f"ΔGap={best_combo[5]-PAPER_AR_PROP_GAP[best_combo[6]]:+.2f}%p)")

print(f"  -> Fig.3/Fig.5 (AR), Fig.6/Fig.7 (MLR) 에서 이 c를 그대로 씀")





# =====================================================================

# 12. Fig.3 (AR, W1/W2 스윕) + Fig.6 (MLR, W1/W2 스윕) 데이터

# =====================================================================

print("\n" + "=" * 70)

print(f"  6. Fig.3/6 데이터 — W1/W2 10개점 스윕 (c={BEST_C})")

print("=" * 70)



fig3_ar_n = [ar_nrmse]

fig3_ar_g = [compute_zhang_gap(ar_pred_flat, BEST_C,

                                actual_flat_all, da_flat_all, rt_flat_all)]

fig6_mlr_n = [mlr_nrmse]

fig6_mlr_g = [compute_zhang_gap(mlr_pred_flat, BEST_C,

                                 actual_flat, da_flat, rt_flat)]



for W1, W2 in W_RATIOS:

    a_n, a_g = get_ar(BEST_C, W1, W2)

    m_n, m_g = get_mlr(BEST_C, W1, W2)

    fig3_ar_n.append(a_n); fig3_ar_g.append(a_g)

    fig6_mlr_n.append(m_n); fig6_mlr_g.append(m_g)

    print(f"  {W1}/{W2}: AR(nRMSE={a_n:.2f}%, Gap={a_g:.2f}%) "

          f"MLR(nRMSE={m_n:.2f}%, Gap={m_g:.2f}%)")





# =====================================================================

# 13. Fig.5 (AR, c 스윕) + Fig.7 (MLR, c 스윕) 데이터

#     W1=W2=1 고정, c 11개점

# =====================================================================

print("\n" + "=" * 70)

print(f"  7. Fig.5/7 데이터 — c 11개점 스윕 (W1/W2={FIG5_W1}/{FIG5_W2})")

print("=" * 70)



fig5_ar_n_list = []; fig5_ar_g_list = []

fig5_ar_prop_n = []; fig5_ar_prop_g = []

fig7_mlr_n_list = []; fig7_mlr_g_list = []

fig7_mlr_prop_n = []; fig7_mlr_prop_g = []



for c_val in FIG_C_VALUES:

    # AR baseline gap

    ar_g_r = compute_zhang_gap(ar_pred_flat, c_val,

                               actual_flat_all, da_flat_all, rt_flat_all)

    fig5_ar_n_list.append(ar_nrmse); fig5_ar_g_list.append(ar_g_r)



    # MLR baseline gap

    mlr_g_r = compute_zhang_gap(mlr_pred_flat, c_val,

                                actual_flat, da_flat, rt_flat)

    fig7_mlr_n_list.append(mlr_nrmse); fig7_mlr_g_list.append(mlr_g_r)



    # 제안모형

    a_n, a_g = get_ar(c_val, FIG5_W1, FIG5_W2)

    m_n, m_g = get_mlr(c_val, FIG5_W1, FIG5_W2)

    fig5_ar_prop_n.append(a_n); fig5_ar_prop_g.append(a_g)

    fig7_mlr_prop_n.append(m_n); fig7_mlr_prop_g.append(m_g)



    print(f"  c={c_val:.1f}: "

          f"AR(base={ar_g_r:.2f}%, prop={a_n:.2f}%/{a_g:.2f}%) "

          f"MLR(base={mlr_g_r:.2f}%, prop={m_n:.2f}%/{m_g:.2f}%)")





# =====================================================================

# 14. CSV 저장

# =====================================================================

def save_csv(path, header, rows):

    with open(path, "w", newline="") as f:

        csv.writer(f).writerow(header)

        for row in rows:

            csv.writer(f).writerow(row)

    print(f"  saved: {path}")



paper_ar_nrmse_all = [PAPER_AR_BASE_NRMSE] + PAPER_AR_PROP_NRMSE

paper_ar_gap_all = [PAPER_AR_BASE_GAP] + PAPER_AR_PROP_GAP

paper_mlr_nrmse_all = [PAPER_MLR_BASE_NRMSE] + PAPER_MLR_PROP_NRMSE

paper_mlr_gap_all = [PAPER_MLR_BASE_GAP] + PAPER_MLR_PROP_GAP



save_csv(os.path.join(OUT_DIR, "fig3_ar_zhang_asym.csv"),

         ["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"],

         [[FIG_LABELS[i], round(fig3_ar_n[i], 2), round(fig3_ar_g[i], 2),

           paper_ar_nrmse_all[i], paper_ar_gap_all[i]]

          for i in range(len(FIG_LABELS))])



save_csv(os.path.join(OUT_DIR, "fig5_ar_zhang_asym.csv"),

         ["c", "AR_base_nRMSE", "AR_base_Gap", "AR_prop_nRMSE", "AR_prop_Gap"],

         [[FIG_C_VALUES[i], round(fig5_ar_n_list[i], 2), round(fig5_ar_g_list[i], 2),

           round(fig5_ar_prop_n[i], 2), round(fig5_ar_prop_g[i], 2)]

          for i in range(len(FIG_C_VALUES))])



save_csv(os.path.join(OUT_DIR, "fig6_mlr_zhang_asym.csv"),

         ["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"],

         [[FIG_LABELS[i], round(fig6_mlr_n[i], 2), round(fig6_mlr_g[i], 2),

           paper_mlr_nrmse_all[i], paper_mlr_gap_all[i]]

          for i in range(len(FIG_LABELS))])



save_csv(os.path.join(OUT_DIR, "fig7_mlr_zhang_asym.csv"),

         ["c", "MLR_base_nRMSE", "MLR_base_Gap", "MLR_prop_nRMSE", "MLR_prop_Gap"],

         [[FIG_C_VALUES[i], round(fig7_mlr_n_list[i], 2), round(fig7_mlr_g_list[i], 2),

           round(fig7_mlr_prop_n[i], 2), round(fig7_mlr_prop_g[i], 2)]

          for i in range(len(FIG_C_VALUES))])





# =====================================================================

# 15. Fig.3 — AR, dual y축 (nRMSE 좌, Gap 우)

# =====================================================================

print("\n" + "=" * 70)

print("  8. Fig.3 plotting (AR, W1/W2 스윕, dual y축)")

print("=" * 70)



x = list(range(len(FIG_LABELS)))

paper_nrmse_all = [PAPER_AR_BASE_NRMSE] + PAPER_AR_PROP_NRMSE

paper_gap_all = [PAPER_AR_BASE_GAP] + PAPER_AR_PROP_GAP



fig, ax_l = plt.subplots(figsize=(10, 6))

fig.suptitle(f"Fig.3 — AR, Zhang 비대칭(rho_+=c*RT), c={BEST_C}, W1/W2 스윕 (dual y축)",

             fontsize=13)

ax_r = ax_l.twinx()



ln1 = ax_l.plot(x, paper_nrmse_all, marker="o", color=C_PAPER,

                linestyle=":", label="논문 nRMSE", alpha=0.7)

ln2 = ax_l.plot(x, fig3_ar_n,   marker="o", color=C_AR_BASE,

                linewidth=2, label="재현 nRMSE")

ln3 = ax_r.plot(x, paper_gap_all, marker="s", color=C_PAPER,

                linestyle=":", label="논문 Gap", alpha=0.7)

ln4 = ax_r.plot(x, fig3_ar_g,   marker="s", color=C_AR_PROP,

                linewidth=2, linestyle="--", label="재현 Gap")



ax_l.set_xticks(x); ax_l.set_xticklabels(FIG_LABELS)

ax_l.set_xlabel("W1/W2"); ax_l.set_ylabel("nRMSE (%)", color=C_AR_BASE)

ax_r.set_ylabel("Optimality Gap (%)", color=C_AR_PROP)

ax_l.tick_params(axis="y", labelcolor=C_AR_BASE)

ax_r.tick_params(axis="y", labelcolor=C_AR_PROP)

ax_l.grid(True, alpha=0.3)

ax_l.legend(ln1+ln2+ln3+ln4, [l.get_label() for l in ln1+ln2+ln3+ln4],

            loc="upper left", fontsize=9)

fig.tight_layout()

p3 = os.path.join(RESULTS_DIR, "fig3_ar_zhang_asym.png")

fig.savefig(p3, dpi=150); plt.close(fig)

print(f"  saved: {p3}")





# =====================================================================

# 16. Fig.5 — AR, c 스윕 (nRMSE / Gap 2분할)

# =====================================================================

print("\n" + "=" * 70)

print("  9. Fig.5 plotting (AR, c 스윕, 2분할)")

print("=" * 70)



x5 = list(range(len(FIG_C_VALUES)))

lbl5 = [f"{c:.1f}" for c in FIG_C_VALUES]

hl_idx = FIG_C_VALUES.index(BEST_C) if BEST_C in FIG_C_VALUES else None



fig5_fig, (ax5n, ax5g) = plt.subplots(1, 2, figsize=(13, 5))

fig5_fig.suptitle(f"Fig.5 — AR, Zhang 비대칭, W1/W2={FIG5_W1}/{FIG5_W2}, c 스윕",

                  fontsize=13)



# nRMSE

ax5n.axhline(ar_nrmse, linestyle="--", color=C_AR_BASE, alpha=0.6, label="재현 AR")

ax5n.plot(x5, fig5_ar_prop_n, marker="o", color=C_AR_PROP, linewidth=2,

          label="재현 제안모형")

if hl_idx is not None:

    ax5n.axvline(hl_idx, color=C_GRID, linestyle=":", linewidth=1.5,

                 label=f"Best c={BEST_C}")

ax5n.set_xticks(x5); ax5n.set_xticklabels(lbl5)

ax5n.set_xlabel("c (rho_plus = c * RT)")

ax5n.set_ylabel("nRMSE (%)")

ax5n.set_title("nRMSE"); ax5n.grid(True, alpha=0.3); ax5n.legend(fontsize=9)



# Gap

ax5g.plot(x5, fig5_ar_g_list, marker="o", color=C_AR_BASE, linewidth=2,

          label="재현 AR Gap")

ax5g.plot(x5, fig5_ar_prop_g, marker="o", color=C_AR_PROP, linewidth=2,

          label="재현 제안모형 Gap")

if hl_idx is not None:

    ax5g.axvline(hl_idx, color=C_GRID, linestyle=":", linewidth=1.5,

                 label=f"Best c={BEST_C}")

ax5g.set_xticks(x5); ax5g.set_xticklabels(lbl5)

ax5g.set_xlabel("c (rho_plus = c * RT)")

ax5g.set_ylabel("Optimality Gap (%)")

ax5g.set_title("Optimality Gap"); ax5g.grid(True, alpha=0.3); ax5g.legend(fontsize=9)



fig5_fig.tight_layout()

p5 = os.path.join(RESULTS_DIR, "fig5_ar_zhang_asym.png")

fig5_fig.savefig(p5, dpi=150); plt.close(fig5_fig)

print(f"  saved: {p5}")





# =====================================================================

# 17. Fig.6 — MLR, dual y축 (nRMSE 좌, Gap 우)

# =====================================================================

print("\n" + "=" * 70)

print("  10. Fig.6 plotting (MLR, W1/W2 스윕, dual y축)")

print("=" * 70)



paper_nrmse_all_mlr = [PAPER_MLR_BASE_NRMSE] + PAPER_MLR_PROP_NRMSE

paper_gap_all_mlr = [PAPER_MLR_BASE_GAP] + PAPER_MLR_PROP_GAP



fig6_fig, ax_l = plt.subplots(figsize=(10, 6))

fig6_fig.suptitle(f"Fig.6 — MLR, Zhang 비대칭(rho_+=c*RT), c={BEST_C}, W1/W2 스윕 (dual y축)",

                  fontsize=13)

ax_r = ax_l.twinx()



ln1 = ax_l.plot(x, paper_nrmse_all_mlr, marker="o", color=C_PAPER,

                linestyle=":", label="논문 nRMSE", alpha=0.7)

ln2 = ax_l.plot(x, fig6_mlr_n,  marker="o", color=C_MLR_BASE,

                linewidth=2, label="재현 nRMSE")

ln3 = ax_r.plot(x, paper_gap_all_mlr, marker="s", color=C_PAPER,

                linestyle=":", label="논문 Gap", alpha=0.7)

ln4 = ax_r.plot(x, fig6_mlr_g,  marker="s", color=C_MLR_PROP,

                linewidth=2, linestyle="--", label="재현 Gap")



ax_l.set_xticks(x); ax_l.set_xticklabels(FIG_LABELS)

ax_l.set_xlabel("W1/W2"); ax_l.set_ylabel("nRMSE (%)", color=C_MLR_BASE)

ax_r.set_ylabel("Optimality Gap (%)", color=C_MLR_PROP)

ax_l.tick_params(axis="y", labelcolor=C_MLR_BASE)

ax_r.tick_params(axis="y", labelcolor=C_MLR_PROP)

ax_l.grid(True, alpha=0.3)

ax_l.legend(ln1+ln2+ln3+ln4, [l.get_label() for l in ln1+ln2+ln3+ln4],

            loc="upper left", fontsize=9)

fig6_fig.tight_layout()

p6 = os.path.join(RESULTS_DIR, "fig6_mlr_zhang_asym.png")

fig6_fig.savefig(p6, dpi=150); plt.close(fig6_fig)

print(f"  saved: {p6}")





# =====================================================================

# 18. Fig.7 — MLR, c 스윕 (nRMSE / Gap 2분할)

# =====================================================================

print("\n" + "=" * 70)

print("  11. Fig.7 plotting (MLR, c 스윕, 2분할)")

print("=" * 70)



fig7_fig, (ax7n, ax7g) = plt.subplots(1, 2, figsize=(13, 5))

fig7_fig.suptitle(f"Fig.7 — MLR, Zhang 비대칭, W1/W2={FIG5_W1}/{FIG5_W2}, c 스윕",

                  fontsize=13)



# nRMSE

ax7n.axhline(mlr_nrmse, linestyle="--", color=C_MLR_BASE, alpha=0.6, label="재현 MLR")

ax7n.plot(x5, fig7_mlr_prop_n, marker="o", color=C_MLR_PROP, linewidth=2,

          label="재현 제안모형")

if hl_idx is not None:

    ax7n.axvline(hl_idx, color=C_GRID, linestyle=":", linewidth=1.5,

                 label=f"Best c={BEST_C}")

ax7n.set_xticks(x5); ax7n.set_xticklabels(lbl5)

ax7n.set_xlabel("c (rho_plus = c * RT)")

ax7n.set_ylabel("nRMSE (%)")

ax7n.set_title("nRMSE"); ax7n.grid(True, alpha=0.3); ax7n.legend(fontsize=9)



# Gap

ax7g.plot(x5, fig7_mlr_g_list, marker="o", color=C_MLR_BASE, linewidth=2,

          label="재현 MLR Gap")

ax7g.plot(x5, fig7_mlr_prop_g, marker="o", color=C_MLR_PROP, linewidth=2,

          label="재현 제안모형 Gap")

if hl_idx is not None:

    ax7g.axvline(hl_idx, color=C_GRID, linestyle=":", linewidth=1.5,

                 label=f"Best c={BEST_C}")

ax7g.set_xticks(x5); ax7g.set_xticklabels(lbl5)

ax7g.set_xlabel("c (rho_plus = c * RT)")

ax7g.set_ylabel("Optimality Gap (%)")

ax7g.set_title("Optimality Gap"); ax7g.grid(True, alpha=0.3); ax7g.legend(fontsize=9)



fig7_fig.tight_layout()

p7 = os.path.join(RESULTS_DIR, "fig7_mlr_zhang_asym.png")

fig7_fig.savefig(p7, dpi=150); plt.close(fig7_fig)

print(f"  saved: {p7}")





# =====================================================================

print("\n" + "=" * 70)

print("  완료 — Zhang 비대칭 3항식 통합 실험 (AR+MLR, z03/블록18)")

print("=" * 70)

print(f"""

  생성 파일:

    Fig.3 (AR W1/W2):    {p3}

    Fig.5 (AR c 스윕):   {p5}

    Fig.6 (MLR W1/W2):   {p6}

    Fig.7 (MLR c 스윕):  {p7}

    Grid CSV:            {grid_csv}

""")

