# -*- coding: utf-8 -*-
# =====================================================================
# integrated_spo_plus_4term_fig3568_AR_MLR.py
#
# SPO+ (Smart Policy Optimization+) — 4항 이익식 + decomposition 방식
# Elmachtoub & Grigas (2022) 프레임워크를 oracle normalization으로
# decomposition하여 목적함수 계수로 구현.
#
# 4항 이익식:
#   profit = scale × (DA·x + RP·surplus − RP·shortage − PC·shortage)
#   PC = rate × DA
#
# SPO+ decomposition:
#   regret = Σ oracle_i · (optimal − ours) / Σoracle
#   → 목적함수 계수 분해:
#     sc_i = (−W1·scale·RT/denom) + W2/n       (surplus cost)
#     yc_i = ( W1·scale·(RT+PC)/denom) + W2/n   (shortage cost, 4항)
#   denom = Σ oracle_train ({0, actual, 1.0} 3후보)
#
# 보완성 제약: yp·ym = 0 (이진 변수 + big-M)
#
# AR(시간대별 12개 모델) + MLR(pooled 1개 모델) 통합 실행
# 대상: z01/블럭21 (2013년 봄)
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
# 1. 설정 — z01/블럭21
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z01.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "simulation_output")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21
HOURS_PER_DAY = 12

TRAIN_START = pd.Timestamp("2013-01-26")
TRAIN_END = pd.Timestamp("2013-04-25")
TEST_START  = pd.Timestamp("2013-04-26")
TEST_END    = pd.Timestamp("2013-05-25")
HISTORY_DATE = pd.Timestamp("2013-01-25")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS

# ── 논문 KPI 조건 ──
KPI_W1 = 1
KPI_W2 = 20
KPI_RATE = 0.5

# ── 그리드 ──
GRID_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 1), (1, 0)]
GRID_PENALTY_RATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

# ── Fig3/6: W1/W2 10개점 ──
FIG_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1),
                (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG_LABELS = ["AR/MLR", "1/20", "1/10", "1/5", "1/2",
              "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]

# ── Fig5/8: penalty rate 0~150% ──
FIG_RATES = [round(0.1 * i, 1) for i in range(16)]

# ── 논문 참고값 (z03/블록18 기준 — z01은 다른 블록이라 참고선일 뿐, 직접 비교 아님) ──
PAPER_AR_NRMSE = 34.76
PAPER_AR_GAP = 15.04
PAPER_MLR_NRMSE = 21.76
PAPER_MLR_GAP = 12.59
# 논문 Table 3 (AR 제안모형, W1/W2 10개점)
PAPER_PROP_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95,
                    46.11, 48.27, 49.21, 49.61, 50.07]
PAPER_PROP_GAP = [13.91, 13.42, 12.71, 11.88, 11.44,
                  11.38, 11.38, 11.36, 11.36, 11.36]
# 논문 Table 4 (MLR 제안모형, W1/W2 10개점) — AR과 다른 값
PAPER_PROP_MLR_NRMSE = [21.92, 21.84, 21.75, 21.66, 22.01,
                        23.32, 27.75, 30.62, 35.76, 37.67]
PAPER_PROP_MLR_GAP = [11.91, 11.68, 11.15, 10.65, 10.28,
                      9.91, 9.51, 9.32, 9.27, 9.28]
# 논문 원본 Fig.5(a)/(b) 육안 판독값 (AR, rate 0~100% 11개점) — 50%만 Table 3 실측치,
# 나머지는 CodefromJiWon/model_proposed_ar_profit_change_v3.py의 판독값을 그대로 재사용
FIG5_PAPER_AR_NRMSE   = [34.76] * 11
FIG5_PAPER_AR_GAP     = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]
FIG5_PAPER_PROP_NRMSE = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
FIG5_PAPER_PROP_GAP   = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]
# 논문 원본 Fig.8(a)/(b) 육안 판독값 (MLR, rate 0~100% 11개점) — 50%만 Table 4 실측치,
# 나머지는 논문 PDF Fig.8(a)/(b) 그래프에서 직접 판독
FIG8_PAPER_MLR_NRMSE  = [21.76] * 11
FIG8_PAPER_PROP_NRMSE = [24.0, 22.5, 21.8, 21.3, 21.8, 22.01, 22.4, 23.0, 23.7, 25.0, 26.2]
FIG8_PAPER_MLR_GAP    = [8.0, 8.9, 9.8, 10.7, 11.6, 12.59, 13.4, 14.3, 15.2, 16.1, 17.0]
FIG8_PAPER_PROP_GAP   = [7.7, 9.2, 9.5, 10.0, 10.3, 10.28, 10.7, 10.9, 10.9, 11.0, 10.9]


# =====================================================================
# 2. 데이터 로딩
# =====================================================================
print("=" * 70)
print("  1. 데이터 로딩")
print("=" * 70)
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & \
              (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START

# ── AR용: (날짜 x 12) 배열 ──
is_history = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history].copy().sort_values("hour_idx")
history_solar = np.zeros((1, HOURS_PER_DAY))
for _, row in history_rows.iterrows():
    history_solar[0, row["hour_idx"]] = row["solar_power"]

is_train = (daylight_table["local_date"] >= TRAIN_START) & \
           (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])

is_test = (daylight_table["local_date"] >= TEST_START) & \
          (daylight_table["local_date"] <= TEST_END)
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

# MLR 입력행렬
n_features_mlr = 4
X_mlr_train = np.column_stack([np.ones(n_train_obs),
                                mlr_train_dssrd, mlr_train_dtsr, mlr_train_hour])
X_mlr_test = np.column_stack([np.ones(n_test_obs),
                               test_rows["dssrd"].to_numpy(),
                               test_rows["dtsr"].to_numpy(),
                               test_rows["hour_idx"].to_numpy(dtype=float)])

# ── AR용: 설계행렬(절편 + 직전 하루 12시간 lag 역순) ──
n_features_ar = 13  # 1(intercept) + 12(lag)
history_and_train = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1))
ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])

# ── 테스트 flat ──
actual_flat_all = test_solar.flatten()
da_flat_all = test_da_price.flatten()
rt_flat_all = test_rt_price.flatten()
n_test_obs_all = len(actual_flat_all)

print(f"  train: {n_train_days}일({n_train_obs}행), test: {n_test_days}일({n_test_obs}행)")


# =====================================================================
# 3. Gap 계산 함수 — 4항식, oracle 3후보 {0, actual, 1.0}
# =====================================================================
def compute_gap_4term(pred_flat, penalty_rate, actual, da, rt):
    """4항 이익함수 optimality gap 계산 (oracle: {0, actual, 1.0})"""
    sum_realized = 0.0
    sum_oracle = 0.0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]
        dp = da[i]; rp = rt[i]
        pc = penalty_rate * dp
        mismatch = a - x
        surplus = max(mismatch, 0)
        shortage = max(-mismatch, 0)
        # 4항: DA*x + RP*surplus - RP*shortage - PC*shortage
        realized = scale * (dp * x + rp * surplus - rp * shortage - pc * shortage)
        sum_realized += realized
        # oracle 후보
        p0 = scale * rp * a  # commit=0: surplus=a → RP*a
        pa = scale * dp * a  # commit=actual: DA*a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rp * s1 - rp * y1 - pc * y1)  # commit=1
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

# AR 예측 (rolling)
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
# 6. SPO+ MILP — AR (시간대별, 4항 decomposition 방식)
#
# SPO+ regret를 oracle normalization으로 decomposition:
#   regret = Σ oracle_i · (optimal_decision − our_decision) / Σoracle
#          → 목적함수 계수로 분해
#
# realized / scale = DA*x + RP*yp − (RP+PC)*ym
#   (yp: surplus, ym: shortage, 분해 제약 x + yp − ym = a)
#
# 목적: min Σ[ −W1·scale·DA·x/denom + sc·yp + yc·ym ]
#   sc_i = (−W1·scale·RT/denom) + W2/n   (surplus cost)
#   yc_i = ( W1·scale·(RT+PC)/denom) + W2/n   (shortage cost, 4항)
#   denom = Σ oracle_train ({0, actual, 1.0} 3후보)
#
# 보완성 제약: yp·ym = 0 (이진 변수 + big-M)
# =====================================================================
def solve_ar_spo_plus(penalty_rate, W1, W2):
    """AR SPO+: 4항 이익식 + oracle normalization decomposition"""
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))

    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]
        da_h = train_da_price[:, hour]
        rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows

        # training oracle (3후보: {0, actual, 1.0})
        oracle_train = np.zeros(n_obs)
        for i in range(n_obs):
            a = y_h[i]; dp = da_h[i]; rp = rt_h[i]
            pc = penalty_rate * dp
            p0 = scale * rp * a; pa = scale * dp * a
            s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
            p1 = scale * (dp * 1.0 + rp * s1 - rp * y1 - pc * y1)
            oracle_train[i] = max(p0, pa, p1)
        denom = oracle_train.sum()

        # 목적함수 계수 (4항)
        sc = np.zeros(n_obs)  # surplus cost
        yc = np.zeros(n_obs)  # shortage cost
        for i in range(n_obs):
            pc = penalty_rate * da_h[i]
            sc[i] = (-W1 * scale * rt_h[i] / denom) + (W2 / n_obs)
            yc[i] = (W1 * scale * (rt_h[i] + pc) / denom) + (W2 / n_obs)

        # 이진변수: sc+yc < 0 인 관측치만 보완성 제약 필요
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
            print(f"  AR MILP 실패 h={hour}: {r.message}")
            coeffs[hour] = 0.0
        else:
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
# 7. SPO+ MILP — MLR (pooled, 4항 decomposition 방식)
# =====================================================================
def solve_mlr_spo_plus(penalty_rate, W1, W2):
    """MLR SPO+: 4항 이익식 + oracle normalization decomposition"""
    n_obs = n_train_obs

    oracle_train = np.zeros(n_obs)
    for i in range(n_obs):
        a = mlr_train_solar[i]; dp = mlr_train_da[i]; rp = mlr_train_rt[i]
        pc = penalty_rate * dp
        p0 = scale * rp * a; pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rp * s1 - rp * y1 - pc * y1)
        oracle_train[i] = max(p0, pa, p1)
    denom = oracle_train.sum()

    sc = np.zeros(n_obs); yc = np.zeros(n_obs)
    for i in range(n_obs):
        pc = penalty_rate * mlr_train_da[i]
        sc[i] = (-W1 * scale * mlr_train_rt[i] / denom) + (W2 / n_obs)
        yc[i] = (W1 * scale * (mlr_train_rt[i] + pc) / denom) + (W2 / n_obs)

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
        raise RuntimeError(f"MLR SPO+ MILP 실패: {r.message}")
    coeffs = r.x[b_s:b_s+n_features_mlr]
    return np.clip(X_mlr_test @ coeffs, 0, 1)


# =====================================================================
# 8. 캐시
# =====================================================================
cache_ar = {}
cache_mlr = {}

def get_ar(pr, w1, w2):
    k = (pr, w1, w2)
    if k not in cache_ar:
        pred = solve_ar_spo_plus(pr, w1, w2)
        cache_ar[k] = (calc_nrmse(pred, actual_flat_all),
                       compute_gap_4term(pred, pr, actual_flat_all, da_flat_all, rt_flat_all))
    return cache_ar[k]

def get_mlr(pr, w1, w2):
    k = (pr, w1, w2)
    if k not in cache_mlr:
        pred = solve_mlr_spo_plus(pr, w1, w2)
        cache_mlr[k] = (calc_nrmse(pred, actual_flat),
                        compute_gap_4term(pred, pr, actual_flat, da_flat, rt_flat))
    return cache_mlr[k]


# =====================================================================
# 9. 전체 그리드 스윕 (50조합) — AR + MLR
# =====================================================================
print("\n" + "=" * 70)
print("  4. 그리드 스윕 (rate 10개 x W1/W2 5개 = 50조합, SPO+ AR+MLR)")
print("=" * 70)

grid_results = []  # (rate, W1, W2, ar_nrmse, ar_gap, mlr_nrmse, mlr_gap)

for pr in GRID_PENALTY_RATES:
    print(f"\n  ── rate = {pr} ──")
    for W1, W2 in GRID_W_RATIOS:
        a_n, a_g = get_ar(pr, W1, W2)
        m_n, m_g = get_mlr(pr, W1, W2)
        grid_results.append((pr, W1, W2, a_n, a_g, m_n, m_g))
        label = f"{W1}/{W2}"
        print(f"  {label}: AR(nRMSE={a_n:.2f}%, Gap={a_g:.2f}%) "
              f"MLR(nRMSE={m_n:.2f}%, Gap={m_g:.2f}%)")

# 그리드 CSV
grid_csv = os.path.join(OUT_DIR, "grid_spo_plus_4term_ar_mlr_z01_block21.csv")
with open(grid_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rate", "W1", "W2",
                "AR_nRMSE", "AR_Gap", "MLR_nRMSE", "MLR_Gap"])
    for row in grid_results:
        w.writerow([row[0], row[1], row[2],
                     round(row[3], 2), round(row[4], 2),
                     round(row[5], 2), round(row[6], 2)])
print(f"\n  saved: {grid_csv}")


# =====================================================================
# 10. 논문 KPI 조건 (W1=1, W2=20, rate=50%) 별도 출력
# =====================================================================
print("\n" + "=" * 70)
print(f"  5. 논문 KPI 조건 — W1={KPI_W1}, W2={KPI_W2}, penalty rate={KPI_RATE} (50%)")
print("=" * 70)

ar_kpi_n, ar_kpi_g = get_ar(KPI_RATE, KPI_W1, KPI_W2)
mlr_kpi_n, mlr_kpi_g = get_mlr(KPI_RATE, KPI_W1, KPI_W2)
print(f"""
  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
  │     모델          │  nRMSE (%)   │  Gap (%)     │  nRMSE (%)   │  Gap (%)     │
  │                  │  (제안모형)   │  (제안모형)   │  (논문 기준)  │  (논문 기준)  │
  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
  │  AR (z01/블럭21)  │ {ar_kpi_n:>10.2f}   │ {ar_kpi_g:>10.2f}   │ {PAPER_AR_NRMSE:>10.2f}   │ {PAPER_AR_GAP:>10.2f}   │
  │  MLR (z01/블럭21) │ {mlr_kpi_n:>10.2f}   │ {mlr_kpi_g:>10.2f}   │ {PAPER_MLR_NRMSE:>10.2f}   │ {PAPER_MLR_GAP:>10.2f}   │
  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

  ※ 논문 KPI 기준값 (z03/블록18):
      AR  nRMSE={PAPER_AR_NRMSE}%, Gap={PAPER_AR_GAP}%
      MLR nRMSE={PAPER_MLR_NRMSE}%, Gap={PAPER_MLR_GAP}%
""")


# =====================================================================
# 11. Fig.3/6 데이터 — W1/W2 10개점 스윕 (rate=0.5 고정)
# =====================================================================
FIG36_RATE = KPI_RATE  # 0.5
print("\n" + "=" * 70)
print(f"  6. Fig.3/6 데이터 — W1/W2 10개점 스윕 (rate={FIG36_RATE})")
print("=" * 70)

fig3_ar_n = [ar_nrmse]
fig3_ar_g = [compute_gap_4term(ar_pred_flat, FIG36_RATE,
                                actual_flat_all, da_flat_all, rt_flat_all)]
fig6_mlr_n = [mlr_nrmse]
fig6_mlr_g = [compute_gap_4term(mlr_pred_flat, FIG36_RATE,
                                 actual_flat, da_flat, rt_flat)]

for W1, W2 in FIG_W_RATIOS:
    a_n, a_g = get_ar(FIG36_RATE, W1, W2)
    m_n, m_g = get_mlr(FIG36_RATE, W1, W2)
    fig3_ar_n.append(a_n); fig3_ar_g.append(a_g)
    fig6_mlr_n.append(m_n); fig6_mlr_g.append(m_g)
    print(f"  {W1}/{W2}: AR(nRMSE={a_n:.2f}%, Gap={a_g:.2f}%) "
          f"MLR(nRMSE={m_n:.2f}%, Gap={m_g:.2f}%)")


# =====================================================================
# 12. Fig.5/8 데이터 — rate 0~150% 스윕 (W1=W2=1)
# =====================================================================
print("\n" + "=" * 70)
print(f"  7. Fig.5/8 데이터 — rate 0~150% 스윕 (W1=W2=1)")
print("=" * 70)

fig5_ar_n_list = []; fig5_ar_g_list = []
fig8_mlr_n_list = []; fig8_mlr_g_list = []
fig5_ar_prop_n = []; fig5_ar_prop_g = []
fig8_mlr_prop_n = []; fig8_mlr_prop_g = []

for rate in FIG_RATES:
    ar_g_r = compute_gap_4term(ar_pred_flat, rate,
                                actual_flat_all, da_flat_all, rt_flat_all)
    mlr_g_r = compute_gap_4term(mlr_pred_flat, rate,
                                 actual_flat, da_flat, rt_flat)
    fig5_ar_n_list.append(ar_nrmse); fig5_ar_g_list.append(ar_g_r)
    fig8_mlr_n_list.append(mlr_nrmse); fig8_mlr_g_list.append(mlr_g_r)

    a_n, a_g = get_ar(rate, 1, 1)
    m_n, m_g = get_mlr(rate, 1, 1)
    fig5_ar_prop_n.append(a_n); fig5_ar_prop_g.append(a_g)
    fig8_mlr_prop_n.append(m_n); fig8_mlr_prop_g.append(m_g)

    print(f"  rate={rate:.1f}: "
          f"AR(base={ar_g_r:.2f}%, prop={a_n:.2f}%/{a_g:.2f}%) "
          f"MLR(base={mlr_g_r:.2f}%, prop={m_n:.2f}%/{m_g:.2f}%)")


# =====================================================================
# 13. CSV 저장
# =====================================================================
def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)
        for row in rows:
            csv.writer(f).writerow(row)
    print(f"  saved: {path}")

save_csv(os.path.join(OUT_DIR, "fig3_spo_plus_4term_AR.csv"),
         ["Label", "nRMSE", "Gap"],
         [[FIG_LABELS[i], round(fig3_ar_n[i], 2), round(fig3_ar_g[i], 2)]
          for i in range(len(FIG_LABELS))])

save_csv(os.path.join(OUT_DIR, "fig5_spo_plus_4term_AR.csv"),
         ["Rate", "AR_base_nRMSE", "AR_base_Gap", "AR_prop_nRMSE", "AR_prop_Gap"],
         [[FIG_RATES[i], round(fig5_ar_n_list[i], 2), round(fig5_ar_g_list[i], 2),
           round(fig5_ar_prop_n[i], 2), round(fig5_ar_prop_g[i], 2)]
          for i in range(len(FIG_RATES))])

save_csv(os.path.join(OUT_DIR, "fig6_spo_plus_4term_MLR.csv"),
         ["Label", "nRMSE", "Gap"],
         [[FIG_LABELS[i], round(fig6_mlr_n[i], 2), round(fig6_mlr_g[i], 2)]
          for i in range(len(FIG_LABELS))])

save_csv(os.path.join(OUT_DIR, "fig8_spo_plus_4term_MLR.csv"),
         ["Rate", "MLR_base_nRMSE", "MLR_base_Gap", "MLR_prop_nRMSE", "MLR_prop_Gap"],
         [[FIG_RATES[i], round(fig8_mlr_n_list[i], 2), round(fig8_mlr_g_list[i], 2),
           round(fig8_mlr_prop_n[i], 2), round(fig8_mlr_prop_g[i], 2)]
          for i in range(len(FIG_RATES))])


# =====================================================================
# 14. Fig.3 — AR, dual y축 (nRMSE 좌, Gap 우)
# =====================================================================
print("\n" + "=" * 70)
print("  8. Fig.3 plotting (AR, W1/W2 스윕, dual y축)")
print("=" * 70)

x = list(range(len(FIG_LABELS)))
fig, ax_l = plt.subplots(figsize=(10, 6))
fig.suptitle(f"Fig.3 — AR SPO+ 4항식, rate={FIG36_RATE}, W1/W2 스윕", fontsize=13)
ax_r = ax_l.twinx()

paper_nrmse_all = [PAPER_AR_NRMSE] + PAPER_PROP_NRMSE
paper_gap_all = [PAPER_AR_GAP] + PAPER_PROP_GAP
ln1 = ax_l.plot(x, paper_nrmse_all, marker="o", color=C_PAPER,
                linewidth=2, linestyle=":", label="논문 nRMSE", alpha=0.8)
ln2 = ax_l.plot(x, fig3_ar_n,   marker="s", color=C_AR_PROP,
                linewidth=2, linestyle="-", label="재현 nRMSE")
ln3 = ax_r.plot(x, paper_gap_all, marker="^", color=C_PAPER,
                linewidth=2, linestyle=":", label="논문 Gap", alpha=0.8)
ln4 = ax_r.plot(x, fig3_ar_g,   marker="d", color=C_AR_BASE,
                linewidth=2, linestyle="-", label="재현 Gap")

ax_l.set_xticks(x); ax_l.set_xticklabels(FIG_LABELS)
ax_l.set_xlabel("W1/W2"); ax_l.set_ylabel("nRMSE (%)", color=C_AR_PROP)
ax_r.set_ylabel("Optimality Gap (%)", color=C_AR_BASE)
ax_l.tick_params(axis="y", labelcolor=C_AR_PROP)
ax_r.tick_params(axis="y", labelcolor=C_AR_BASE)
ax_l.grid(True, alpha=0.3)
ax_l.set_ylim(30, 80)
ax_r.set_ylim(0, 25)
all_ln = ln1+ln2+ln3+ln4
ax_l.legend(all_ln, [l.get_label() for l in all_ln],
            loc="upper left", fontsize=9)
fig.tight_layout()
p3 = os.path.join(RESULTS_DIR, "fig3_spo_plus_4term_W1W2_AR.png")
fig.savefig(p3, dpi=150); plt.close(fig)
print(f"  saved: {p3}")


# =====================================================================
# 15. Fig.5 — AR, rate 스윕 (nRMSE / Gap 2분할)
# =====================================================================
print("\n" + "=" * 70)
print("  9. Fig.5 plotting (AR, rate 스윕, 2분할)")
print("=" * 70)

x5 = list(range(len(FIG_RATES)))
lbl5 = [f"{int(r*100)}%" for r in FIG_RATES]
x5_paper = x5[:11]  # 논문 판독값은 rate 0~100%(11개점)까지만 있음

fig5_fig, (ax5n, ax5g) = plt.subplots(1, 2, figsize=(13, 5))
fig5_fig.suptitle(f"Fig.5 — AR SPO+ 4항식, W1=W2=1, penalty rate 스윕", fontsize=13)

# nRMSE
ax5n.plot(x5_paper, FIG5_PAPER_AR_NRMSE, linestyle="--", color=C_PAPER, alpha=0.8,
          marker="o", markersize=4, label="논문 AR")
ax5n.plot(x5_paper, FIG5_PAPER_PROP_NRMSE, linestyle="--", color=C_AR_PROP, alpha=0.5,
          marker="s", markersize=4, label="논문 제안모형")
ax5n.plot(x5, fig5_ar_n_list, marker="o", color=C_AR_BASE, linewidth=2, linestyle="-", label="재현 AR")
ax5n.plot(x5, fig5_ar_prop_n, marker="s", color=C_AR_PROP, linewidth=2, linestyle="-", label="재현 제안모형")
ax5n.set_xticks(x5); ax5n.set_xticklabels(lbl5, rotation=45)
ax5n.set_xlabel("벌금비용률"); ax5n.set_ylabel("nRMSE (%)")
ax5n.set_title("nRMSE"); ax5n.grid(True, alpha=0.3); ax5n.legend(fontsize=8)
ax5n.set_ylim(30, 80)

# Gap
ax5g.plot(x5_paper, FIG5_PAPER_AR_GAP, linestyle="--", color=C_PAPER, alpha=0.8,
          marker="o", markersize=4, label="논문 AR")
ax5g.plot(x5_paper, FIG5_PAPER_PROP_GAP, linestyle="--", color=C_AR_PROP, alpha=0.5,
          marker="s", markersize=4, label="논문 제안모형")
ax5g.plot(x5, fig5_ar_g_list, marker="o", color=C_AR_BASE, linewidth=2, linestyle="-", label="재현 AR")
ax5g.plot(x5, fig5_ar_prop_g, marker="s", color=C_AR_PROP, linewidth=2, linestyle="-", label="재현 제안모형")
hl = FIG_RATES.index(KPI_RATE) if KPI_RATE in FIG_RATES else None
if hl is not None:
    ax5g.axvline(hl, color=C_GRID, linestyle=":", linewidth=1.5,
                 label=f"KPI rate={int(KPI_RATE*100)}%")
ax5g.set_xticks(x5); ax5g.set_xticklabels(lbl5, rotation=45)
ax5g.set_xlabel("벌금비용률"); ax5g.set_ylabel("Optimality Gap (%)")
ax5g.set_title("Optimality Gap"); ax5g.grid(True, alpha=0.3); ax5g.legend(fontsize=8)
ax5g.set_ylim(0, 25)

fig5_fig.tight_layout()
p5 = os.path.join(RESULTS_DIR, "fig5_spo_plus_4term_rate_AR.png")
fig5_fig.savefig(p5, dpi=150); plt.close(fig5_fig)
print(f"  saved: {p5}")


# =====================================================================
# 16. Fig.6 — MLR, dual y축 (nRMSE 좌, Gap 우)
# =====================================================================
print("\n" + "=" * 70)
print("  10. Fig.6 plotting (MLR, W1/W2 스윕, dual y축)")
print("=" * 70)

fig6_fig, ax_l = plt.subplots(figsize=(10, 6))
fig6_fig.suptitle(f"Fig.6 — MLR SPO+ 4항식, rate={FIG36_RATE}, W1/W2 스윕", fontsize=13)
ax_r = ax_l.twinx()

paper_mlr_nrmse_all = [PAPER_MLR_NRMSE] + PAPER_PROP_MLR_NRMSE
paper_mlr_gap_all = [PAPER_MLR_GAP] + PAPER_PROP_MLR_GAP
ln1 = ax_l.plot(x, paper_mlr_nrmse_all, marker="o", color=C_PAPER,
                linewidth=2, linestyle=":", label="논문 nRMSE", alpha=0.8)
ln2 = ax_l.plot(x, fig6_mlr_n, marker="s", color=C_MLR_PROP,
                linewidth=2, linestyle="-", label="재현 nRMSE")
ln3 = ax_r.plot(x, paper_mlr_gap_all, marker="^", color=C_PAPER,
                linewidth=2, linestyle=":", label="논문 Gap", alpha=0.8)
ln4 = ax_r.plot(x, fig6_mlr_g, marker="d", color=C_MLR_BASE,
                linewidth=2, linestyle="-", label="재현 Gap")

ax_l.set_xticks(x); ax_l.set_xticklabels(FIG_LABELS)
ax_l.set_xlabel("W1/W2"); ax_l.set_ylabel("nRMSE (%)", color=C_MLR_PROP)
ax_r.set_ylabel("Optimality Gap (%)", color=C_MLR_BASE)
ax_l.tick_params(axis="y", labelcolor=C_MLR_PROP)
ax_r.tick_params(axis="y", labelcolor=C_MLR_BASE)
ax_l.grid(True, alpha=0.3)
ax_l.set_ylim(0, 80)
ax_r.set_ylim(0, 25)
all_ln = ln1+ln2+ln3+ln4
ax_l.legend(all_ln, [l.get_label() for l in all_ln],
            loc="upper left", fontsize=9)
fig6_fig.tight_layout()
p6 = os.path.join(RESULTS_DIR, "fig6_spo_plus_4term_W1W2_MLR.png")
fig6_fig.savefig(p6, dpi=150); plt.close(fig6_fig)
print(f"  saved: {p6}")


# =====================================================================
# 17. Fig.8 — MLR, rate 스윕 (nRMSE / Gap 2분할)
# =====================================================================
print("\n" + "=" * 70)
print("  11. Fig.8 plotting (MLR, rate 스윕, 2분할)")
print("=" * 70)

fig8_fig, (ax8n, ax8g) = plt.subplots(1, 2, figsize=(13, 5))
fig8_fig.suptitle(f"Fig.8 — MLR SPO+ 4항식, W1=W2=1, penalty rate 스윕", fontsize=13)

# nRMSE
ax8n.plot(x5_paper, FIG8_PAPER_MLR_NRMSE, linestyle="--", color=C_PAPER, alpha=0.8,
          marker="o", markersize=4, label="논문 MLR")
ax8n.plot(x5_paper, FIG8_PAPER_PROP_NRMSE, linestyle="--", color=C_MLR_PROP, alpha=0.5,
          marker="s", markersize=4, label="논문 제안모형")
ax8n.plot(x5, fig8_mlr_n_list, marker="o", color=C_MLR_BASE, linewidth=2, linestyle="-", label="재현 MLR")
ax8n.plot(x5, fig8_mlr_prop_n, marker="s", color=C_MLR_PROP, linewidth=2, linestyle="-", label="재현 제안모형")
ax8n.set_xticks(x5); ax8n.set_xticklabels(lbl5, rotation=45)
ax8n.set_xlabel("벌금비용률"); ax8n.set_ylabel("nRMSE (%)")
ax8n.set_title("nRMSE"); ax8n.grid(True, alpha=0.3); ax8n.legend(fontsize=8)
ax8n.set_ylim(0, 80)

# Gap
ax8g.plot(x5_paper, FIG8_PAPER_MLR_GAP, linestyle="--", color=C_PAPER, alpha=0.8,
          marker="o", markersize=4, label="논문 MLR")
ax8g.plot(x5_paper, FIG8_PAPER_PROP_GAP, linestyle="--", color=C_MLR_PROP, alpha=0.5,
          marker="s", markersize=4, label="논문 제안모형")
ax8g.plot(x5, fig8_mlr_g_list, marker="o", color=C_MLR_BASE, linewidth=2, linestyle="-", label="재현 MLR")
ax8g.plot(x5, fig8_mlr_prop_g, marker="s", color=C_MLR_PROP, linewidth=2, linestyle="-", label="재현 제안모형")
if hl is not None:
    ax8g.axvline(hl, color=C_GRID, linestyle=":", linewidth=1.5,
                 label=f"KPI rate={int(KPI_RATE*100)}%")
ax8g.set_xticks(x5); ax8g.set_xticklabels(lbl5, rotation=45)
ax8g.set_xlabel("벌금비용률"); ax8g.set_ylabel("Optimality Gap (%)")
ax8g.set_title("Optimality Gap"); ax8g.grid(True, alpha=0.3); ax8g.legend(fontsize=8)
ax8g.set_ylim(0, 25)

fig8_fig.tight_layout()
p8 = os.path.join(RESULTS_DIR, "fig8_spo_plus_4term_rate_MLR.png")
fig8_fig.savefig(p8, dpi=150); plt.close(fig8_fig)
print(f"  saved: {p8}")


# =====================================================================
# 18. 전체 결과 요약
# =====================================================================
print("\n" + "=" * 70)
print("  완료 — SPO+ 4항식 통합 실험 (AR+MLR, z01/블럭21)")
print("=" * 70)
print(f"""
  생성 파일:
    Fig.3 (AR W1/W2):   {p3}
    Fig.5 (AR rate):    {p5}
    Fig.6 (MLR W1/W2):  {p6}
    Fig.8 (MLR rate):   {p8}
    Grid CSV:           {grid_csv}

  ── KPI 조건 (W1=1, W2=20, rate=50%) ──
    AR  제안모형: nRMSE={ar_kpi_n:.2f}%, Gap={ar_kpi_g:.2f}%
    MLR 제안모형: nRMSE={mlr_kpi_n:.2f}%, Gap={mlr_kpi_g:.2f}%
    AR  논문 기준: nRMSE={PAPER_AR_NRMSE:.2f}%, Gap={PAPER_AR_GAP:.2f}%
    MLR 논문 기준: nRMSE={PAPER_MLR_NRMSE:.2f}%, Gap={PAPER_MLR_GAP:.2f}%

  ※ 4항식 SPO+ = decomposition 방식 (oracle normalization)
     (ζ-제약 방식과 달리 예측 정확도 + regret 동시 최적화)
""")