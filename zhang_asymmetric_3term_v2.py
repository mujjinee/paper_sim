
# -*- coding: utf-8 -*-

# =====================================================================
# zhang_asymmetric_3term_v2.py
#
# zhang_asymmetric_3term.py(방법 3: Zhang 비대칭 시장가격, c=1.0/1.2/1.5/
# 2.0/3.0 x W1/W2 10개 스윕 + Best 조합 자동 탐색)에 Fig.3/Fig.5 그리기를
# 추가한다 - method1_pc_max_da_rt_ar_v3.py / method1_pc_max_da_rt_mlr_v2.py
# 와 동일한 패턴(전체 그리드 -> Best 자동탐색 -> Fig.3/Fig.5 -> 그림).
#
#   - Zhang 모형엔 W1/W2 10개점(1/20~1/0)이 이미 그리드 자체와 똑같아서,
#     Fig.3은 best c에서 그리드 결과를 그대로 재사용 - 추가 MILP 없음.
#   - Zhang엔 method1의 "penalty rate" 같은 별도 손잡이가 없고 c가 그
#     역할을 하므로, Fig.5는 W1=W2=1 고정, c 자체를 스윕한다(1.0~3.0,
#     그리드보다 촘촘한 11개점 - 그리드와 겹치는 5개는 캐시에서 재사용).
#
#   1. 전체 그리드(C_RATES x W_RATIOS) 스윕 -> CSV 저장 -> Best c 탐색
#   2. Best c 고정, Fig.3용 W1/W2 스윕 (그리드와 완전히 겹침 - 재사용만 함)
#   3. W1=W2=1 고정, Fig.5용 c 11개점 스윕 (그리드와 겹치는 5개는 재사용)
#   4. Fig.3(dual y축) / Fig.5(좌우 2분할, 논문 비교선 없음 - c 스윕 자체가
#      논문에 없음) 그리기
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
# 1. 설정값
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

# --- 전체 그리드 설정 (zhang_asymmetric_3term.py 그대로) ---
W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
C_RATES = [1.0, 1.2, 1.5, 2.0, 3.0]

PAPER_AR_NRMSE = 34.76
PAPER_AR_GAP = 15.04
PAPER_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95, 46.11, 48.27, 49.21, 49.61, 50.07]
PAPER_GAP = [13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36]

# --- Fig.3 설정: W_RATIOS와 완전히 동일 (라벨만 따로 둠, AR 포함 11개점) ---
FIG3_LABELS = ["AR", "1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]

# --- Fig.5 설정: W1=W2=1 고정, c 자체를 스윕 (그리드 5개보다 촘촘하게 11개) ---
FIG5_W1, FIG5_W2 = 1, 1
FIG5_C_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]

# 논문 Fig.5(벌금율 0~100% 스윕, AR+제안모형)를 육안으로 읽은 참고값. Zhang의
# c 스윕(1.0~3.0)과는 x축 의미 자체가 다르므로(논문엔 "c 스윕"이 없음),
# 11개 지점 개수만 맞춰 같은 자리에 점선으로 얹는다 - 값끼리 정확히
# 대응되는 비교가 아니라 "논문이 대략 이런 범위였다"는 참고선일 뿐이다.
FIG5_PAPER_AR_NRMSE   = [34.76] * 11
FIG5_PAPER_AR_GAP     = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]
FIG5_PAPER_PROP_NRMSE = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
FIG5_PAPER_PROP_GAP   = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]


# =====================================================================
# 2. 데이터 읽기 + 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START


# =====================================================================
# 3. 이력/학습/테스트 구간
# =====================================================================
is_history = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history].copy().sort_values("hour_idx")

is_train = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])

is_test = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])

print(f"이력: {len(history_rows)}행, 학습: {len(train_rows)}행, 테스트: {len(test_rows)}행")


# =====================================================================
# 4. (날짜 x 12시간) 배열
# =====================================================================
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
# 5. AR 학습용 입력(X) - 논문 Eq.(3)
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
# 6. Gap 계산 함수 (Zhang 3항: DA*x + rho_-*surplus - rho_+*shortage)
#    rho_- = RT, rho_+ = c*RT, oracle 3후보 {0, actual, 1.0}
# =====================================================================
def compute_zhang_gap(pred_flat, c_rate):
    sum_realized = 0.0
    sum_oracle = 0.0
    for i in range(n_test_obs):
        a = actual_flat[i]
        x = pred_flat[i]
        dp = da_flat[i]
        rp = rt_flat[i]
        rho_plus = c_rate * rp
        rho_minus = rp

        mismatch = a - x
        surplus = max(mismatch, 0)
        shortage = max(-mismatch, 0)

        realized = scale * (dp * x + rho_minus * surplus - rho_plus * shortage)
        sum_realized += realized

        p0 = scale * rho_minus * a
        pa = scale * dp * a
        s1 = max(a - 1.0, 0)
        y1 = max(1.0 - a, 0)
        p1 = scale * (dp * 1.0 + rho_minus * s1 - rho_plus * y1)
        oracle = max(p0, pa, p1)
        sum_oracle += oracle

    return 100 * (sum_oracle - sum_realized) / sum_oracle


# =====================================================================
# 7. 기본 AR (baseline) — c와 무관하므로 한 번만 계산
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
# 8. (c_rate, W1, W2) 하나를 받아 Zhang 제안모형 MILP를 풀고 (nrmse, gap)을
#    반환하는 함수. zhang_asymmetric_3term.py 7절 로직을 그대로 함수화.
# =====================================================================
def run_proposed_model(c_rate, W1, W2):
    coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))

    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]
        da_h = train_da_price[:, hour]
        rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows

        rho_plus_h = c_rate * rt_h
        rho_minus_h = rt_h

        # Training oracle: 3후보 {0, actual, 1.0}
        oracle_profit_train = np.zeros(n_obs)
        for i in range(n_obs):
            a = y_h[i]; dp = da_h[i]
            r_p = rho_plus_h[i]; r_m = rho_minus_h[i]
            p0 = scale * r_m * a
            pa = scale * dp * a
            s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)
            p1 = scale * (dp * 1.0 + r_m * s1 - r_p * y1)
            oracle_profit_train[i] = max(p0, pa, p1)
        training_denom = oracle_profit_train.sum()

        surplus_cost = np.zeros(n_obs)
        shortage_cost = np.zeros(n_obs)
        for i in range(n_obs):
            surplus_cost[i] = (-W1 * scale * rho_minus_h[i] / training_denom) + (W2 / n_obs)
            shortage_cost[i] = (W1 * scale * rho_plus_h[i] / training_denom) + (W2 / n_obs)

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
    gap = compute_zhang_gap(pred_flat, c_rate)
    return nrmse, gap


# =====================================================================
# 9. 캐시가 달린 조회 함수 - (c, W1, W2) 조합을 이미 풀었으면 재사용
# =====================================================================
solved_cache = {}   # {(c_rate, W1, W2): (nrmse, gap)}


def get_proposed_result(c_rate, W1, W2):
    key = (c_rate, W1, W2)
    if key not in solved_cache:
        solved_cache[key] = run_proposed_model(c_rate, W1, W2)
    return solved_cache[key]


# =====================================================================
# 10. 전체 그리드 스윕 - c 5개 x W1/W2 10개 = 50조합
#    (zhang_asymmetric_3term.py 7절과 동일, 결과는 캐시에 저장돼서 Fig.3은
#     그리드와 완전히 겹쳐 추가 MILP 없이, Fig.5는 일부만 재사용한다)
# =====================================================================
all_results = []

for c_rate in C_RATES:
    print(f"\n{'='*60}")
    print(f"  c = {c_rate}  (rho_plus = {c_rate}*RT, rho_minus = RT)")
    print(f"{'='*60}")

    results_c = []
    for widx, (W1, W2) in enumerate(W_RATIOS):
        nrmse, gap = get_proposed_result(c_rate, W1, W2)
        label = f"{W1}/{W2}"
        results_c.append((label, W1, W2, nrmse, gap))
        print(f"  {label}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}% "
              f"(paper: nRMSE={PAPER_NRMSE[widx]:.2f}%, Gap={PAPER_GAP[widx]:.2f}%)")

    all_results.append((c_rate, results_c))


# =====================================================================
# 11. 전체 결과 표 + CSV 저장
# =====================================================================
print()
print("=" * 110)
print(f"{'c':>4} {'Label':>6} {'W1':>4} {'W2':>4} {'nRMSE':>10} {'Gap':>10} {'Paper nRMSE':>12} {'Paper Gap':>12} {'Δ nRMSE':>10} {'Δ Gap':>10}")
print("-" * 110)

for c_rate, results_c in all_results:
    for label, W1, W2, nrmse, gap in results_c:
        idx = W_RATIOS.index((W1, W2))
        print(f"{c_rate:>4} {label:>6} {W1:>4} {W2:>4} {nrmse:>9.2f}% {gap:>9.2f}% "
              f"{PAPER_NRMSE[idx]:>11.2f}% {PAPER_GAP[idx]:>11.2f}% "
              f"{nrmse-PAPER_NRMSE[idx]:>+9.2f}%p {gap-PAPER_GAP[idx]:>+9.2f}%p")
print("=" * 110)

print(f"\nAR baseline:")
for c_rate in C_RATES:
    arg = compute_zhang_gap(ar_pred_flat, c_rate)
    print(f"  c={c_rate}: nRMSE={ar_nrmse:.2f}%, Gap={arg:.2f}% "
          f"(paper: nRMSE={PAPER_AR_NRMSE:.2f}%, Gap={PAPER_AR_GAP:.2f}%)")

grid_out_dir = os.path.join(BASE_DIR, "results", "simulation_output")
os.makedirs(grid_out_dir, exist_ok=True)
grid_csv_path = os.path.join(grid_out_dir, "fig3_zhang_asymmetric_3term.csv")
with open(grid_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["c_rate", "Label", "W1", "W2", "nRMSE", "Gap",
                      "Paper_nRMSE", "Paper_Gap", "Delta_nRMSE", "Delta_Gap"])
    for c_rate, results_c in all_results:
        for label, W1, W2, nrmse, gap in results_c:
            idx = W_RATIOS.index((W1, W2))
            writer.writerow([c_rate, label, W1, W2, round(nrmse, 2), round(gap, 2),
                             PAPER_NRMSE[idx], PAPER_GAP[idx],
                             round(nrmse - PAPER_NRMSE[idx], 2), round(gap - PAPER_GAP[idx], 2)])
        writer.writerow([c_rate, "AR", "", "", round(ar_nrmse, 2),
                         round(compute_zhang_gap(ar_pred_flat, c_rate), 2),
                         PAPER_AR_NRMSE, PAPER_AR_GAP, "", ""])
        writer.writerow([])
print(f"\n저장: {grid_csv_path}")


# =====================================================================
# 12. Best (c, W1/W2) 자동 탐색 - |ΔnRMSE|+|ΔGap| 최소 조합
# =====================================================================
best_total_err = np.inf
best_combo = None
for c_rate, results_c in all_results:
    for label, W1, W2, nrmse, gap in results_c:
        idx = W_RATIOS.index((W1, W2))
        err = abs(nrmse - PAPER_NRMSE[idx]) + abs(gap - PAPER_GAP[idx])
        if err < best_total_err:
            best_total_err = err
            best_combo = (c_rate, label, nrmse, gap, idx)

BEST_C = best_combo[0]
print(f"\nBest match: c={BEST_C}, W1/W2={best_combo[1]} "
      f"(nRMSE={best_combo[2]:.2f}%, Gap={best_combo[3]:.2f}%, "
      f"ΔnRMSE={best_combo[2]-PAPER_NRMSE[best_combo[4]]:+.2f}%p, "
      f"ΔGap={best_combo[3]-PAPER_GAP[best_combo[4]]:+.2f}%p) "
      f"-> Fig.3/Fig.5에서 이 c를 그대로 씀")


# =====================================================================
# 13. Fig.3 데이터 생성 — c=BEST_C 고정, W1/W2 스윕
#     (W_RATIOS가 그리드와 완전히 동일해서 전부 캐시에서 재사용 - 추가 MILP 없음)
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  Fig.3 데이터 생성 (c={BEST_C} 고정 - 전체 그리드에서 자동 탐색된 최적값)")
print(f"{'=' * 60}")

fig3_ar_gap = compute_zhang_gap(ar_pred_flat, BEST_C)
fig3_nrmse_list = [ar_nrmse]
fig3_gap_list = [fig3_ar_gap]
print(f"  AR: nRMSE={ar_nrmse:.2f}%, Gap={fig3_ar_gap:.2f}%")

for (W1, W2) in W_RATIOS:
    nrmse, gap = get_proposed_result(BEST_C, W1, W2)
    fig3_nrmse_list.append(nrmse)
    fig3_gap_list.append(gap)
    print(f"  {W1}/{W2}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}%")


# =====================================================================
# 14. Fig.5 데이터 생성 — W1=W2=1 고정, c 11개점 스윕
#     (그리드와 겹치는 5개 c값은 캐시에서 재사용)
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  Fig.5 데이터 생성 (W1/W2={FIG5_W1}/{FIG5_W2} 고정, c 스윕)")
print(f"{'=' * 60}")

fig5_ar_gap_list = []
fig5_prop_nrmse_list = []
fig5_prop_gap_list = []

for c_val in FIG5_C_VALUES:
    ar_gap_c = compute_zhang_gap(ar_pred_flat, c_val)
    fig5_ar_gap_list.append(ar_gap_c)

    prop_nrmse_c, prop_gap_c = get_proposed_result(c_val, FIG5_W1, FIG5_W2)
    fig5_prop_nrmse_list.append(prop_nrmse_c)
    fig5_prop_gap_list.append(prop_gap_c)

    print(f"  c={c_val:.1f}: AR Gap={ar_gap_c:.2f}%, "
          f"제안 nRMSE={prop_nrmse_c:.2f}%, Gap={prop_gap_c:.2f}%")


# =====================================================================
# 15. Fig.3/Fig.5 결과 CSV 저장
# =====================================================================
out_dir = os.path.join(BASE_DIR, "results", "simulation_output")
os.makedirs(out_dir, exist_ok=True)

fig3_csv_path = os.path.join(out_dir, "fig3_zhang_asymmetric_3term_v2.csv")
with open(fig3_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"])
    paper_nrmse_all = [PAPER_AR_NRMSE] + PAPER_NRMSE
    paper_gap_all = [PAPER_AR_GAP] + PAPER_GAP
    for label, nrmse, gap, p_nrmse, p_gap in zip(
            FIG3_LABELS, fig3_nrmse_list, fig3_gap_list, paper_nrmse_all, paper_gap_all):
        writer.writerow([label, round(nrmse, 2), round(gap, 2), p_nrmse, p_gap])
print(f"\nsaved: {fig3_csv_path}")

fig5_csv_path = os.path.join(out_dir, "fig5_zhang_asymmetric_3term_v2.csv")
with open(fig5_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["c", "AR_Gap", "Proposed_nRMSE", "Proposed_Gap"])
    for c_val, ag, pn, pg in zip(FIG5_C_VALUES, fig5_ar_gap_list, fig5_prop_nrmse_list, fig5_prop_gap_list):
        writer.writerow([c_val, round(ag, 2), round(pn, 2), round(pg, 2)])
print(f"saved: {fig5_csv_path}")


# =====================================================================
# 16. Fig.3 그리기 — dual y축 (nRMSE 왼쪽, Gap 오른쪽) - 논문 원본과 동일한
#     축 구성 (method1_pc_max_da_rt_ar_v3.py의 Fig.3과 동일한 형태)
# =====================================================================
results_out_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_out_dir, exist_ok=True)

x3 = list(range(len(FIG3_LABELS)))
paper_nrmse_all = [PAPER_AR_NRMSE] + PAPER_NRMSE
paper_gap_all = [PAPER_AR_GAP] + PAPER_GAP

figure3, axis_left = plt.subplots(figsize=(10, 6))
figure3.suptitle(f"Fig.3 재현 — Zhang 비대칭(rho_+=c*RT), c={BEST_C}(자동탐색 최적값), W1/W2 스윕 (dual y축)", fontsize=13)

axis_right = axis_left.twinx()

line1 = axis_left.plot(x3, paper_nrmse_all, marker="o", color="#2a78d6",
                        linestyle="-", label="논문 nRMSE")
line2 = axis_left.plot(x3, fig3_nrmse_list, marker="o", color="#0b3d78",
                        linestyle="-", label="재현 nRMSE")

line3 = axis_right.plot(x3, paper_gap_all, marker="s", color="#eb9834",
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
fig3_png_path = os.path.join(results_out_dir, "fig3_zhang_asymmetric_3term_v2.png")
figure3.savefig(fig3_png_path, dpi=150)
print(f"\nsaved: {fig3_png_path}")


# =====================================================================
# 17. Fig.5 그리기 — nRMSE / Gap 나란히, c=BEST_C 지점을 세로 점선으로 강조
#     (논문에는 Zhang c 스윕 자체가 없으므로 논문 비교선 없이 재현값만 표시)
# =====================================================================
x5 = list(range(len(FIG5_C_VALUES)))
x5_labels = [f"{c:.1f}" for c in FIG5_C_VALUES]
highlight_idx = FIG5_C_VALUES.index(BEST_C)

fig5, (ax5_nrmse, ax5_gap) = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle(f"Fig.5 재현 — Zhang 비대칭(rho_+=c*RT), W1/W2={FIG5_W1}/{FIG5_W2}, c 스윕", fontsize=14)

ax5_nrmse.axhline(ar_nrmse, linestyle="--", color="#2a78d6", alpha=0.6, label="재현 AR")
ax5_nrmse.plot(x5, fig5_prop_nrmse_list, marker="o", color="#eb6834", label="재현 제안모형")
ax5_nrmse.axvline(highlight_idx, color="gray", linestyle=":", alpha=0.7)
ax5_nrmse.set_xticks(x5)
ax5_nrmse.set_xticklabels(x5_labels)
ax5_nrmse.set_xlabel("c (rho_plus = c * RT)")
ax5_nrmse.set_ylabel("nRMSE (%)")
ax5_nrmse.set_title(f"nRMSE 비교 (점선=c={BEST_C})")
ax5_nrmse.grid(True, alpha=0.3)
ax5_nrmse.legend(fontsize=9)

ax5_gap.plot(x5, fig5_ar_gap_list, marker="o", color="#2a78d6", label="재현 AR")
ax5_gap.plot(x5, fig5_prop_gap_list, marker="o", color="#eb6834", label="재현 제안모형")
ax5_gap.axvline(highlight_idx, color="gray", linestyle=":", alpha=0.7)
ax5_gap.set_xticks(x5)
ax5_gap.set_xticklabels(x5_labels)
ax5_gap.set_xlabel("c (rho_plus = c * RT)")
ax5_gap.set_ylabel("Optimality Gap (%)")
ax5_gap.set_title(f"Optimality Gap 비교 (점선=c={BEST_C})")
ax5_gap.grid(True, alpha=0.3)
ax5_gap.legend(fontsize=9)

fig5.tight_layout()
fig5_png_path = os.path.join(results_out_dir, "fig5_zhang_asymmetric_3term_v2.png")
fig5.savefig(fig5_png_path, dpi=150)
print(f"saved: {fig5_png_path}")

print("\n완료.")
