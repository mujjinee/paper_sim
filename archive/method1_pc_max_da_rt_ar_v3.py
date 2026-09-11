
# -*- coding: utf-8 -*-

# =====================================================================
# method1_pc_max_da_rt_ar_v3.py
#
# method1_pc_max_da_rt_ar_v2.py 와 동일한 계산(전체 그리드 스윕 + Best
# rate 자동 탐색 + Fig.3/Fig.5 계산)이지만, Fig.3 그리기만 좌우 2분할
# 대신 dual y축(nRMSE 왼쪽, Gap 오른쪽) 하나로 바꿨다 - model_proposed_ar_
# profit_change_v3.py / method1_pc_max_da_rt_mlr_v2.py의 Fig.6과 동일한
# 형태. Fig.5는 기존처럼 좌우 2분할(AR+제안모형) 그대로 유지.
#
#   1. 전체 그리드(PENALTY_RATES x W_RATIOS) 스윕 -> CSV 저장 -> Best 탐색
#   2. Best rate 고정, Fig.3용 W1/W2 10개점 스윕 (그리드와 겹치는 5개는 재사용)
#   3. W1=W2=1 고정, Fig.5용 rate 16개점(0~150%) 스윕 (그리드와 겹치는
#      8개는 재사용). Best rate 지점은 세로 점선으로 강조.
#   4. Fig.3(dual y축) / Fig.5(좌우 2분할) 그리기 (논문값과 비교)
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
# 0. 한글이 깨지지 않도록 시스템에 있는 한글 폰트를 찾아서 지정
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


# =====================================================================
# 1. 설정
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")

HOURS_PER_DAY = 12
LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21

TRAIN_START = pd.Timestamp("2013-08-25")
TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23")
TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS

# --- 전체 그리드 설정 (method1_pc_max_da_rt_ar.py 그대로) ---
GRID_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 1), (1, 0)]
GRID_PENALTY_RATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

PAPER_AR_NRMSE = 34.76
PAPER_AR_GAP = 15.04
GRID_PAPER_NRMSE = [34.89, 35.14, 36.28, 44.95, 50.07]
GRID_PAPER_GAP = [13.91, 13.42, 12.71, 11.44, 11.36]

# --- Fig.3 설정: best rate 고정, W1/W2 10개점 스윕 (논문 표3/Fig.3 전체) ---
FIG3_LABELS = ["AR", "1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]
FIG3_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG3_PAPER_NRMSE = [34.76, 34.89, 35.14, 36.28, 41.09, 44.95, 46.11, 48.27, 49.21, 49.61, 50.07]
FIG3_PAPER_GAP   = [15.04, 13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36]

# --- Fig.5 설정: W1/W2 고정, penalty rate 0~150% 스윕 (그리드 최대값 1.5까지
#     범위를 넉넉히 잡아서, best rate가 100%를 넘게 나오더라도 항상 x축
#     안에 들어오도록 함 - method1_pc_max_da_rt_mlr_v2.py의 Fig.8과 동일한 안전장치) ---
FIG5_W1, FIG5_W2 = 1, 1
FIG5_RATES = [round(0.1 * i, 1) for i in range(16)]   # 0.0, 0.1, ..., 1.5

# 논문 Fig.5를 육안으로 읽은 참고값(50%만 Table 3 실측치, 나머지는 근사치).
# 논문 x축은 0~100%까지만 있어서 11개만 채우고, 110%~150%는 None으로 비워
# 그 구간만 논문 곡선이 자연스럽게 끊기게 한다.
FIG5_PAPER_AR_NRMSE   = [34.76] * 11 + [None] * 5
FIG5_PAPER_AR_GAP     = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22] + [None] * 5
FIG5_PAPER_PROP_NRMSE = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67] + [None] * 5
FIG5_PAPER_PROP_GAP   = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5] + [None] * 5


# =====================================================================
# 2. 데이터 로딩 및 (날짜 x 12) 배열 구성
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START

is_history = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history].copy().sort_values("hour_idx")

is_train = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])

is_test = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])

print(f"history: {len(history_rows)}, train: {len(train_rows)}, test: {len(test_rows)}")

history_solar = np.zeros((1, HOURS_PER_DAY))
for _, row in history_rows.iterrows():
    history_solar[0, row["hour_idx"]] = row["solar_power"]

train_dates_sorted = sorted(train_rows["local_date"].unique())
n_train_days = len(train_dates_sorted)

train_solar = np.zeros((n_train_days, HOURS_PER_DAY))
train_da_price = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt_price = np.zeros((n_train_days, HOURS_PER_DAY))

for _, row in train_rows.iterrows():
    d = (row["local_date"] - TRAIN_START).days
    h = row["hour_idx"]
    train_solar[d, h] = row["solar_power"]
    train_da_price[d, h] = row["da_price"]
    train_rt_price[d, h] = row["rt_price"]

test_dates_sorted = sorted(test_rows["local_date"].unique())
n_test_days = len(test_dates_sorted)

test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))

for _, row in test_rows.iterrows():
    d = (row["local_date"] - TEST_START).days
    h = row["hour_idx"]
    test_solar[d, h] = row["solar_power"]
    test_da_price[d, h] = row["da_price"]
    test_rt_price[d, h] = row["rt_price"]


# =====================================================================
# 3. AR 학습용 입력 (절편 + lag 12개, 날짜를 거꾸로 뒤집어 lag 특징으로 사용)
# =====================================================================
history_and_train_solar = np.vstack([history_solar, train_solar])

n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1))
ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train_solar[d][::-1]

ar_design_matrix = np.hstack([ar_intercept, ar_lag])
n_features = ar_design_matrix.shape[1]

actual_flat = test_solar.flatten()
da_flat = test_da_price.flatten()
rt_flat = test_rt_price.flatten()
n_test_obs = len(actual_flat)


# =====================================================================
# 4. Gap 계산 함수 — PC = rate * max(DA, RT), oracle 3후보 {0, actual, 1.0}
# =====================================================================
def compute_gap(pred_flat, penalty_rate):
    sum_realized = 0.0
    sum_oracle = 0.0
    for i in range(n_test_obs):
        a = actual_flat[i]; x = pred_flat[i]
        dp = da_flat[i]; rp = rt_flat[i]
        pc = penalty_rate * max(dp, rp)

        mismatch = a - x
        surplus = max(mismatch, 0)
        shortage = max(-mismatch, 0)

        realized = scale * (dp * x + rp * surplus - pc * shortage)
        sum_realized += realized

        p0 = scale * rp * a
        pa = scale * dp * a
        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rp * s1 - pc * y1)
        oracle = max(p0, pa, p1)
        sum_oracle += oracle

    return 100 * (sum_oracle - sum_realized) / sum_oracle


# =====================================================================
# 5. 기본 AR (penalty rate와 무관 — 회귀 예측만으로 결정됨, 한 번만 계산)
# =====================================================================
ar_coefficients = np.zeros((HOURS_PER_DAY, n_features))
for h in range(HOURS_PER_DAY):
    y_h = train_solar[:, h]
    X_sp = sparse.csr_matrix(ar_design_matrix)
    I = sparse.eye(n_ar_rows, format="csr")
    A = sparse.vstack([sparse.hstack([X_sp, -I]), sparse.hstack([-X_sp, -I])], format="csr")
    b_lim = np.concatenate([y_h, -y_h])
    c_obj = np.concatenate([np.zeros(n_features), np.ones(n_ar_rows) / n_ar_rows])
    vb = [(None, None)] * n_features + [(0.0, None)] * n_ar_rows
    res = linprog(c_obj, A_ub=A, b_ub=b_lim, bounds=vb, method="highs")
    ar_coefficients[h] = res.x[:n_features]

test_forecast_ar = np.zeros((n_test_days, HOURS_PER_DAY))
prev_day = train_solar[-1]
for d in range(n_test_days):
    feat = np.concatenate([[1.0], prev_day[::-1]])
    for h in range(HOURS_PER_DAY):
        test_forecast_ar[d, h] = np.clip(np.dot(ar_coefficients[h], feat), 0, 1)
    prev_day = test_solar[d]

ar_pred_flat = test_forecast_ar.flatten()
ar_rmse = np.sqrt(np.mean((actual_flat - ar_pred_flat) ** 2))
ar_nrmse = 100 * ar_rmse / np.mean(actual_flat)


# =====================================================================
# 6. (penalty_rate, W1, W2) 하나를 받아 제안모형 MILP를 풀고 (nrmse, gap)을
#    반환하는 함수. method1_pc_max_da_rt_ar.py 6절 로직을 그대로 함수화.
# =====================================================================
def run_proposed_model(penalty_rate, W1, W2):
    coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))

    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]
        da_h = train_da_price[:, hour]
        rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows

        # Training oracle: 3후보
        oracle_profit_train = np.zeros(n_obs)
        for i in range(n_obs):
            a = y_h[i]; dp = da_h[i]; rp = rt_h[i]
            pc = penalty_rate * max(dp, rp)
            p0 = scale * rp * a
            pa = scale * dp * a
            s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
            p1 = scale * (dp * 1.0 + rp * s1 - pc * y1)
            oracle_profit_train[i] = max(p0, pa, p1)
        training_denom = oracle_profit_train.sum()

        surplus_cost = np.zeros(n_obs)
        shortage_cost = np.zeros(n_obs)
        for i in range(n_obs):
            pc = penalty_rate * max(da_h[i], rt_h[i])
            surplus_cost[i] = (-W1 * scale * rt_h[i] / training_denom) + (W2 / n_obs)
            shortage_cost[i] = (W1 * scale * pc / training_denom) + (W2 / n_obs)

        binary_row_list = [i for i in range(n_obs) if surplus_cost[i] + shortage_cost[i] < 0]
        binary_rows_arr = np.array(binary_row_list, dtype=int)
        n_binary = len(binary_rows_arr)

        beta_s = 0; x_s = n_features; yp_s = n_features + n_obs
        ym_s = n_features + 2 * n_obs; z_s = n_features + 3 * n_obs
        n_variables = n_features + 3 * n_obs + n_binary

        objective = np.zeros(n_variables)
        for i in range(n_obs):
            objective[x_s + i] = -W1 * scale * da_h[i] / training_denom
            objective[yp_s + i] = surplus_cost[i]
            objective[ym_s + i] = shortage_cost[i]

        X_sparse = sparse.csr_matrix(ar_design_matrix)
        identity_n = sparse.eye(n_obs, format="csr")

        eq_a = sparse.lil_matrix((n_obs, n_variables))
        eq_a[:, beta_s:beta_s + n_features] = -X_sparse
        eq_a[:, x_s:x_s + n_obs] = identity_n

        eq_b = sparse.lil_matrix((n_obs, n_variables))
        eq_b[:, x_s:x_s + n_obs] = identity_n
        eq_b[:, yp_s:yp_s + n_obs] = identity_n
        eq_b[:, ym_s:ym_s + n_obs] = -identity_n

        all_eq = sparse.vstack([eq_a, eq_b], format="csr")
        eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
        eq_con = LinearConstraint(all_eq, eq_rhs, eq_rhs)
        all_con = [eq_con]

        if n_binary > 0:
            comp_mat = sparse.lil_matrix((2 * n_binary, n_variables))
            for k in range(n_binary):
                row = binary_rows_arr[k]
                comp_mat[k, yp_s + row] = 1.0; comp_mat[k, z_s + k] = 1.0
                comp_mat[n_binary + k, ym_s + row] = 1.0; comp_mat[n_binary + k, z_s + k] = -1.0
            comp_upper = np.concatenate([np.ones(n_binary), np.zeros(n_binary)])
            comp_lower = np.full(2 * n_binary, -np.inf)
            all_con.append(LinearConstraint(comp_mat.tocsr(), comp_lower, comp_upper))

        lb = np.concatenate([np.full(n_features, -np.inf), np.zeros(3 * n_obs + n_binary)])
        ub = np.concatenate([np.full(n_features, np.inf), np.ones(3 * n_obs + n_binary)])
        bounds = Bounds(lb, ub)

        integrality = np.zeros(n_variables, dtype=int)
        for k in range(n_binary):
            integrality[z_s + k] = 1

        milp_result = milp(c=objective, integrality=integrality, bounds=bounds,
                            constraints=all_con, options={"mip_rel_gap": 1e-9})
        coefficients_by_hour[hour] = milp_result.x[beta_s:beta_s + n_features]

    test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))
    prev_day = train_solar[-1]
    for d in range(n_test_days):
        feat = np.concatenate([[1.0], prev_day[::-1]])
        for h in range(HOURS_PER_DAY):
            raw = np.dot(coefficients_by_hour[h], feat)
            test_forecast[d, h] = np.clip(raw, 0, 1)
        prev_day = test_solar[d]

    pred_flat = test_forecast.flatten()
    rmse = np.sqrt(np.mean((actual_flat - pred_flat) ** 2))
    nrmse = 100 * rmse / np.mean(actual_flat)
    gap = compute_gap(pred_flat, penalty_rate)
    return nrmse, gap


# =====================================================================
# 7. 캐시가 달린 조회 함수 - (rate, W1, W2) 조합을 이미 풀었으면 재사용
# =====================================================================
solved_cache = {}   # {(rate, W1, W2): (nrmse, gap)}


def get_proposed_result(penalty_rate, W1, W2):
    key = (penalty_rate, W1, W2)
    if key not in solved_cache:
        solved_cache[key] = run_proposed_model(penalty_rate, W1, W2)
    return solved_cache[key]


# =====================================================================
# 8. 전체 그리드 스윕 - rate 10개 x W1/W2 5개 = 50조합
#    (method1_pc_max_da_rt_ar.py 6절과 동일, 결과는 캐시에 저장돼서
#     Fig.3/Fig.5에서 겹치는 조합은 다시 풀지 않는다)
# =====================================================================
all_results = []

for penalty_rate in GRID_PENALTY_RATES:
    print(f"\n{'='*60}")
    print(f"  rate = {penalty_rate}  (PC = rate*max(DA, RT))")
    print(f"{'='*60}")

    results_c = []
    for widx, (W1, W2) in enumerate(GRID_W_RATIOS):
        nrmse, gap = get_proposed_result(penalty_rate, W1, W2)
        label = f"{W1}/{W2}"
        results_c.append((label, W1, W2, nrmse, gap))
        print(f"  {label}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}% "
              f"(paper: nRMSE={GRID_PAPER_NRMSE[widx]:.2f}%, Gap={GRID_PAPER_GAP[widx]:.2f}%)")

    all_results.append((penalty_rate, results_c))


# =====================================================================
# 9. 그리드 결과 표 + CSV 저장
# =====================================================================
print()
print("=" * 110)
print(f"{'rate':>5} {'Label':>6} {'W1':>4} {'W2':>4} {'nRMSE':>10} {'Gap':>10} {'Paper nRMSE':>12} {'Paper Gap':>12} {'d nRMSE':>10} {'d Gap':>10}")
print("-" * 110)

for penalty_rate, results_c in all_results:
    for label, W1, W2, nrmse, gap in results_c:
        idx = GRID_W_RATIOS.index((W1, W2))
        print(f"{penalty_rate:>5} {label:>6} {W1:>4} {W2:>4} {nrmse:>9.2f}% {gap:>9.2f}% "
              f"{GRID_PAPER_NRMSE[idx]:>11.2f}% {GRID_PAPER_GAP[idx]:>11.2f}% "
              f"{nrmse-GRID_PAPER_NRMSE[idx]:>+9.2f}%p {gap-GRID_PAPER_GAP[idx]:>+9.2f}%p")
print("=" * 110)

print(f"\nAR baseline:")
for penalty_rate in GRID_PENALTY_RATES:
    arg = compute_gap(ar_pred_flat, penalty_rate)
    print(f"  rate={penalty_rate}: nRMSE={ar_nrmse:.2f}%, Gap={arg:.2f}% "
          f"(paper: nRMSE={PAPER_AR_NRMSE:.2f}%, Gap={PAPER_AR_GAP:.2f}%)")

grid_out_dir = os.path.join(BASE_DIR, "results", "simulation_output")
os.makedirs(grid_out_dir, exist_ok=True)
grid_csv_path = os.path.join(grid_out_dir, "fig3_method1_pc_max_da_rt.csv")
with open(grid_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rate", "Label", "W1", "W2", "nRMSE", "Gap",
                      "Paper_nRMSE", "Paper_Gap", "Delta_nRMSE", "Delta_Gap"])
    for penalty_rate, results_c in all_results:
        for label, W1, W2, nrmse, gap in results_c:
            idx = GRID_W_RATIOS.index((W1, W2))
            writer.writerow([penalty_rate, label, W1, W2, round(nrmse, 2), round(gap, 2),
                             GRID_PAPER_NRMSE[idx], GRID_PAPER_GAP[idx],
                             round(nrmse - GRID_PAPER_NRMSE[idx], 2), round(gap - GRID_PAPER_GAP[idx], 2)])
print(f"\nsaved: {grid_csv_path}")


# =====================================================================
# 10. Best (rate, W1/W2) 자동 탐색 - |ΔnRMSE|+|ΔGap| 최소 조합
# =====================================================================
best_total_err = np.inf
best_combo = None
for penalty_rate, results_c in all_results:
    for label, W1, W2, nrmse, gap in results_c:
        idx = GRID_W_RATIOS.index((W1, W2))
        err = abs(nrmse - GRID_PAPER_NRMSE[idx]) + abs(gap - GRID_PAPER_GAP[idx])
        if err < best_total_err:
            best_total_err = err
            best_combo = (penalty_rate, label, nrmse, gap, idx)

BEST_RATE = best_combo[0]
print(f"\nBest: rate={BEST_RATE}, W1/W2={best_combo[1]} "
      f"(nRMSE={best_combo[2]:.2f}%, Gap={best_combo[3]:.2f}%, "
      f"dnRMSE={best_combo[2]-GRID_PAPER_NRMSE[best_combo[4]]:+.2f}%p, "
      f"dGap={best_combo[3]-GRID_PAPER_GAP[best_combo[4]]:+.2f}%p) "
      f"-> Fig.3/Fig.5에서 이 rate를 그대로 씀")


# =====================================================================
# 11. Fig.3 데이터 생성 — rate=BEST_RATE 고정, W1/W2 10개점 스윕
#     (그리드와 겹치는 5개 조합은 캐시에서 재사용)
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  Fig.3 데이터 생성 (rate={BEST_RATE} 고정 — 전체 그리드에서 자동 탐색된 최적값)")
print(f"{'=' * 60}")

fig3_ar_gap = compute_gap(ar_pred_flat, BEST_RATE)
fig3_nrmse_list = [ar_nrmse]
fig3_gap_list = [fig3_ar_gap]
print(f"  AR: nRMSE={ar_nrmse:.2f}%, Gap={fig3_ar_gap:.2f}%")

for (W1, W2) in FIG3_W_RATIOS:
    nrmse, gap = get_proposed_result(BEST_RATE, W1, W2)
    fig3_nrmse_list.append(nrmse)
    fig3_gap_list.append(gap)
    print(f"  {W1}/{W2}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}%")


# =====================================================================
# 12. Fig.5 데이터 생성 — W1/W2=1/1 고정, penalty rate 0~150% 스윕
#     (그리드와 겹치는 rate는 캐시에서 재사용)
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  Fig.5 데이터 생성 (W1/W2={FIG5_W1}/{FIG5_W2} 고정)")
print(f"{'=' * 60}")

fig5_ar_nrmse_list = []
fig5_ar_gap_list = []
fig5_prop_nrmse_list = []
fig5_prop_gap_list = []

for rate in FIG5_RATES:
    ar_gap_r = compute_gap(ar_pred_flat, rate)
    fig5_ar_nrmse_list.append(ar_nrmse)
    fig5_ar_gap_list.append(ar_gap_r)

    prop_nrmse_r, prop_gap_r = get_proposed_result(rate, FIG5_W1, FIG5_W2)
    fig5_prop_nrmse_list.append(prop_nrmse_r)
    fig5_prop_gap_list.append(prop_gap_r)

    print(f"  rate={rate:.1f}: AR Gap={ar_gap_r:.2f}%, "
          f"제안 nRMSE={prop_nrmse_r:.2f}%, Gap={prop_gap_r:.2f}%")


# =====================================================================
# 13. Fig.3/Fig.5 결과 CSV 저장
# =====================================================================
out_dir = os.path.join(BASE_DIR, "results", "simulation_output")
os.makedirs(out_dir, exist_ok=True)

fig3_csv_path = os.path.join(out_dir, "fig3_method1_pc_max_da_rt_ar_v3.csv")
with open(fig3_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"])
    for label, nrmse, gap, p_nrmse, p_gap in zip(
            FIG3_LABELS, fig3_nrmse_list, fig3_gap_list, FIG3_PAPER_NRMSE, FIG3_PAPER_GAP):
        writer.writerow([label, round(nrmse, 2), round(gap, 2), p_nrmse, p_gap])
print(f"\nsaved: {fig3_csv_path}")

fig5_csv_path = os.path.join(out_dir, "fig5_method1_pc_max_da_rt_ar_v3.csv")
with open(fig5_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Rate", "AR_nRMSE", "AR_Gap", "Proposed_nRMSE", "Proposed_Gap"])
    for rate, an, ag, pn, pg in zip(
            FIG5_RATES, fig5_ar_nrmse_list, fig5_ar_gap_list, fig5_prop_nrmse_list, fig5_prop_gap_list):
        writer.writerow([rate, round(an, 2), round(ag, 2), round(pn, 2), round(pg, 2)])
print(f"saved: {fig5_csv_path}")


# =====================================================================
# 14. Fig.3 그리기 — dual y축 (nRMSE 왼쪽, Gap 오른쪽) - 논문 원본과 동일한
#     축 구성 (model_proposed_ar_profit_change_v3_fig3.png 와 같은 형태)
# =====================================================================
results_out_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_out_dir, exist_ok=True)

x3 = list(range(len(FIG3_LABELS)))

figure3, axis_left = plt.subplots(figsize=(10, 6))
figure3.suptitle(f"Fig.3 재현 — 방법1(PC=rate·max(DA,RT)), rate={BEST_RATE}(자동탐색 최적값), W1/W2 스윕 (dual y축)", fontsize=13)

axis_right = axis_left.twinx()

line1 = axis_left.plot(x3, FIG3_PAPER_NRMSE, marker="o", color="#2a78d6",
                        linestyle="-", label="논문 nRMSE")
line2 = axis_left.plot(x3, fig3_nrmse_list, marker="o", color="#0b3d78",
                        linestyle="-", label="재현 nRMSE")

line3 = axis_right.plot(x3, FIG3_PAPER_GAP, marker="s", color="#eb9834",
                         linestyle="--", label="논문 Gap")
line4 = axis_right.plot(x3, fig3_gap_list, marker="s", color="#b34700",
                         linestyle="--", label="재현 Gap")

axis_left.set_xticks(x3)
axis_left.set_xticklabels(FIG3_LABELS)
axis_left.set_xlabel("W1/W2")
axis_left.set_ylabel("nRMSE (%)", color="#0b3d78")
axis_right.set_ylabel("Optimality Gap (%)", color="#b34700")
axis_left.tick_params(axis="y", labelcolor="#0b3d78")
axis_right.tick_params(axis="y", labelcolor="#b34700")
axis_left.grid(True, alpha=0.3)

all_lines = line1 + line2 + line3 + line4
all_labels = [one_line.get_label() for one_line in all_lines]
axis_left.legend(all_lines, all_labels, loc="upper left", fontsize=9)

figure3.tight_layout()
fig3_png_path = os.path.join(results_out_dir, "fig3_method1_pc_max_da_rt_ar_v3.png")
figure3.savefig(fig3_png_path, dpi=150)
print(f"\nsaved: {fig3_png_path}")


# =====================================================================
# 15. Fig.5 그리기 — nRMSE / Gap 나란히, 논문값과 비교 (좌우 2분할 유지)
#     BEST_RATE 지점은 세로 점선으로 강조 표시
# =====================================================================
def _to_nan(values):
    return [np.nan if v is None else v for v in values]

fig5_paper_ar_nrmse_arr = _to_nan(FIG5_PAPER_AR_NRMSE)
fig5_paper_prop_nrmse_arr = _to_nan(FIG5_PAPER_PROP_NRMSE)
fig5_paper_ar_gap_arr = _to_nan(FIG5_PAPER_AR_GAP)
fig5_paper_prop_gap_arr = _to_nan(FIG5_PAPER_PROP_GAP)

x5_labels = [f"{int(round(r * 100))}%" for r in FIG5_RATES]
x5 = list(range(len(x5_labels)))
highlight_idx = FIG5_RATES.index(BEST_RATE)

fig5, (ax5_nrmse, ax5_gap) = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle(f"Fig.5 재현 — 방법1(PC=rate·max(DA,RT)), W1/W2={FIG5_W1}/{FIG5_W2}, rate 스윕 "
              "(점선=논문, 0~100%만 존재)", fontsize=13)

ax5_nrmse.plot(x5, fig5_paper_ar_nrmse_arr, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 AR")
ax5_nrmse.plot(x5, fig5_paper_prop_nrmse_arr, linestyle="--", color="#eb6834", alpha=0.6, label="논문 제안모형")
ax5_nrmse.plot(x5, fig5_ar_nrmse_list, marker="o", color="#2a78d6", label="재현 AR")
ax5_nrmse.plot(x5, fig5_prop_nrmse_list, marker="o", color="#eb6834", label="재현 제안모형")
ax5_nrmse.axvline(highlight_idx, color="gray", linestyle=":", alpha=0.7)
ax5_nrmse.set_xticks(x5)
ax5_nrmse.set_xticklabels(x5_labels)
ax5_nrmse.set_xlabel("벌금비용률(penalty cost rate)")
ax5_nrmse.set_ylabel("nRMSE (%)")
ax5_nrmse.set_title(f"nRMSE 비교 (점선=rate={int(BEST_RATE*100)}%)")
ax5_nrmse.grid(True, alpha=0.3)
ax5_nrmse.legend(fontsize=8)

ax5_gap.plot(x5, fig5_paper_ar_gap_arr, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 AR")
ax5_gap.plot(x5, fig5_paper_prop_gap_arr, linestyle="--", color="#eb6834", alpha=0.6, label="논문 제안모형")
ax5_gap.plot(x5, fig5_ar_gap_list, marker="o", color="#2a78d6", label="재현 AR")
ax5_gap.plot(x5, fig5_prop_gap_list, marker="o", color="#eb6834", label="재현 제안모형")
ax5_gap.axvline(highlight_idx, color="gray", linestyle=":", alpha=0.7)
ax5_gap.set_xticks(x5)
ax5_gap.set_xticklabels(x5_labels)
ax5_gap.set_xlabel("벌금비용률(penalty cost rate)")
ax5_gap.set_ylabel("Optimality Gap (%)")
ax5_gap.set_title(f"Optimality Gap 비교 (점선=rate={int(BEST_RATE*100)}%)")
ax5_gap.grid(True, alpha=0.3)
ax5_gap.legend(fontsize=8)

fig5.tight_layout()
fig5_png_path = os.path.join(results_out_dir, "fig5_method1_pc_max_da_rt_ar_v3.png")
fig5.savefig(fig5_png_path, dpi=150)
print(f"saved: {fig5_png_path}")

print("\n완료.")
