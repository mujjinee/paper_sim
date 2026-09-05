
# -*- coding: utf-8 -*-

# =====================================================================
# method1_pc_max_da_rt_mlr_v2.py
#
# method1_pc_max_da_rt_mlr.py(전체 그리드: rate 10개 x W1/W2 5개 스윕 +
# Best 조합 자동 탐색)와 method1_pc_max_da_rt_mlr_rate12_fig6_fig8_그리기.py
# (rate=1.2 하드코딩 + Fig.6/Fig.8 계산+그림)를 한 파일로 합친다.
#
#   - 이전엔 "best rate=1.2"를 이미 계산된 CSV를 보고 손으로 넣었는데,
#     이 파일은 전체 그리드를 직접 스윕해서 best rate를 그 자리에서
#     자동으로 찾아낸 뒤, 그 값으로 Fig.6/Fig.8까지 그린다.
#   - 그리드에서 이미 계산된 (rate, W1, W2) 조합은 캐시에 저장해뒀다가
#     Fig.6/Fig.8 계산 시 재사용해서 MILP를 다시 풀지 않는다
#     (Fig.6은 W1/W2 10개 중 5개, Fig.8은 rate 16개 중 10개가 그리드와 겹침).
#
#   1. 전체 그리드(PENALTY_RATES x W_RATIOS) 스윕 -> CSV 저장 -> Best 탐색
#   2. Best rate 고정, Fig.6용 W1/W2 10개점 스윕 (그리드와 겹치는 5개는 재사용)
#   3. W1=W2=1 고정, Fig.8용 rate 16개점(0~150%) 스윕 (그리드와 겹치는
#      10개는 재사용)
#   4. Fig.6(dual y축) / Fig.8(좌우 분할 + 논문 육안판독 곡선) 그리기
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

LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21

TRAIN_START = pd.Timestamp("2013-08-25")
TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23")
TEST_END = pd.Timestamp("2013-12-22")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS

# --- 전체 그리드 설정 (method1_pc_max_da_rt_mlr.py 그대로) ---
GRID_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 1), (1, 0)]
GRID_PENALTY_RATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

PAPER_MLR_NRMSE = 21.76
PAPER_MLR_GAP = 12.59
GRID_PAPER_NRMSE = [21.92, 21.84, 21.75, 22.01, 37.67]
GRID_PAPER_GAP = [11.91, 11.68, 11.15, 10.28, 9.28]

# --- Fig.6 설정: best rate 고정, W1/W2 10개점 스윕 (논문 표4/그림6 전체) ---
FIG6_LABELS = ["MLR", "1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]
FIG6_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG6_PAPER_NRMSE = [21.92, 21.84, 21.75, 21.66, 22.01, 23.32, 27.75, 30.62, 35.76, 37.67]
FIG6_PAPER_GAP   = [11.91, 11.68, 11.15, 10.65, 10.28, 9.91, 9.51, 9.32, 9.27, 9.28]

# --- Fig.8 설정: W1=W2=1 고정, rate 0.0~1.5(0.1 단위, 16개점) 스윕 ---
FIG8_W1, FIG8_W2 = 1.0, 1.0
FIG8_PENALTY_RATES = [round(0.1 * i, 1) for i in range(16)]   # 0.0, 0.1, ..., 1.5

# 논문 그림8(a)/(b)을 PDF에서 직접 렌더링해 육안으로 읽은 근사치 (0~100%만
# 존재, 110%~150%는 None으로 비워 그 구간은 선이 끊기게 함)
FIG8_PAPER_MLR_NRMSE = [21.8] * 11 + [None] * 5
FIG8_PAPER_PROP_NRMSE = [24.0, 22.5, 21.7, 21.3, 21.8, 22.0, 22.6, 23.0, 23.7, 24.8, 26.2] + [None] * 5
FIG8_PAPER_MLR_GAP = [7.8, 8.7, 9.6, 10.5, 11.4, 12.4, 13.3, 14.2, 15.1, 16.1, 17.0] + [None] * 5
FIG8_PAPER_PROP_GAP = [7.3, 8.8, 9.7, 10.0, 10.3, 10.3, 10.5, 10.7, 10.8, 10.9, 11.0] + [None] * 5


# =====================================================================
# 2. 데이터
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START

is_train = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])

is_test = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])

print(f"train: {len(train_rows)}, test: {len(test_rows)}")


# =====================================================================
# 3. 학습/테스트용 1차원(flat) 배열
# =====================================================================
n_train_obs = len(train_rows)
train_solar = train_rows["solar_power"].to_numpy()
train_dssrd = train_rows["dssrd"].to_numpy()
train_dtsr = train_rows["dtsr"].to_numpy()
train_hour = train_rows["hour_idx"].to_numpy(dtype=float)
train_da_price = train_rows["da_price"].to_numpy()
train_rt_price = train_rows["rt_price"].to_numpy()

n_test_obs = len(test_rows)
actual_flat = test_rows["solar_power"].to_numpy()
test_dssrd = test_rows["dssrd"].to_numpy()
test_dtsr = test_rows["dtsr"].to_numpy()
test_hour = test_rows["hour_idx"].to_numpy(dtype=float)
da_flat = test_rows["da_price"].to_numpy()
rt_flat = test_rows["rt_price"].to_numpy()


# =====================================================================
# 4. 회귀 입력행렬(X) - 절편 + dSSRD + dTSR + Hour
# =====================================================================
n_features = 4
X_train = np.column_stack([np.ones(n_train_obs), train_dssrd, train_dtsr, train_hour])
X_test = np.column_stack([np.ones(n_test_obs), test_dssrd, test_dtsr, test_hour])


# =====================================================================
# 5. Gap 계산 함수 - PC = rate * max(DA, RT)
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
# 6. 기본 MLR (bounded LAD, pooled) - rate/W1/W2와 무관하게 한 번만
# =====================================================================
X_sp = sparse.csr_matrix(X_train)
I_train = sparse.eye(n_train_obs, format="csr")
A_lad = sparse.vstack([sparse.hstack([X_sp, -I_train]), sparse.hstack([-X_sp, -I_train])], format="csr")
b_lad = np.concatenate([train_solar, -train_solar])
c_lad = np.concatenate([np.zeros(n_features), np.ones(n_train_obs) / n_train_obs])
bounds_lad = [(None, None)] * n_features + [(0.0, None)] * n_train_obs

res_lad = linprog(c_lad, A_ub=A_lad, b_ub=b_lad, bounds=bounds_lad, method="highs")
mlr_coefficients = res_lad.x[:n_features]

mlr_pred_flat = np.clip(X_test @ mlr_coefficients, 0, 1)
mlr_rmse = np.sqrt(np.mean((actual_flat - mlr_pred_flat) ** 2))
mlr_nrmse = 100 * mlr_rmse / np.mean(actual_flat)
print(f"기본 MLR: nRMSE={mlr_nrmse:.2f}% (paper: {PAPER_MLR_NRMSE:.2f}%)")


# =====================================================================
# 7. (penalty_rate, W1, W2) 하나를 받아 제안모형 pooled MILP를 풀고
#    (nrmse, gap)을 반환하는 함수 - method1_pc_max_da_rt_mlr.py 6절과 동일
# =====================================================================
def solve_proposed_milp(penalty_rate, W1, W2):
    n_obs = n_train_obs

    oracle_profit_train = np.zeros(n_obs)
    for i in range(n_obs):
        a = train_solar[i]; dp = train_da_price[i]; rp = train_rt_price[i]
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
        pc = penalty_rate * max(train_da_price[i], train_rt_price[i])
        surplus_cost[i] = (-W1 * scale * train_rt_price[i] / training_denom) + (W2 / n_obs)
        shortage_cost[i] = (W1 * scale * pc / training_denom) + (W2 / n_obs)

    binary_row_list = [i for i in range(n_obs) if surplus_cost[i] + shortage_cost[i] < 0]
    binary_rows_arr = np.array(binary_row_list, dtype=int)
    n_binary = len(binary_rows_arr)

    beta_s = 0; x_s = n_features; yp_s = n_features + n_obs
    ym_s = n_features + 2 * n_obs; z_s = n_features + 3 * n_obs
    n_variables = n_features + 3 * n_obs + n_binary

    objective = np.zeros(n_variables)
    for i in range(n_obs):
        objective[x_s + i] = -W1 * scale * train_da_price[i] / training_denom
        objective[yp_s + i] = surplus_cost[i]
        objective[ym_s + i] = shortage_cost[i]

    X_sparse = sparse.csr_matrix(X_train)
    identity_n = sparse.eye(n_obs, format="csr")

    eq_a = sparse.lil_matrix((n_obs, n_variables))
    eq_a[:, beta_s:beta_s + n_features] = -X_sparse
    eq_a[:, x_s:x_s + n_obs] = identity_n

    eq_b = sparse.lil_matrix((n_obs, n_variables))
    eq_b[:, x_s:x_s + n_obs] = identity_n
    eq_b[:, yp_s:yp_s + n_obs] = identity_n
    eq_b[:, ym_s:ym_s + n_obs] = -identity_n

    all_eq = sparse.vstack([eq_a, eq_b], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_obs), train_solar])
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
    coefficients = milp_result.x[beta_s:beta_s + n_features]

    pred_flat = np.clip(X_test @ coefficients, 0, 1)
    rmse = np.sqrt(np.mean((actual_flat - pred_flat) ** 2))
    nrmse = 100 * rmse / np.mean(actual_flat)
    gap = compute_gap(pred_flat, penalty_rate)
    return nrmse, gap


# =====================================================================
# 8. 캐시가 달린 조회 함수 - (rate, W1, W2) 조합을 이미 풀었으면 그 결과를
#    재사용하고, 없으면 solve_proposed_milp()를 새로 호출해서 캐시에 저장
# =====================================================================
solved_cache = {}   # {(rate, W1, W2): (nrmse, gap)}


def get_proposed_result(penalty_rate, W1, W2):
    key = (penalty_rate, W1, W2)
    if key not in solved_cache:
        solved_cache[key] = solve_proposed_milp(penalty_rate, W1, W2)
    return solved_cache[key]


# =====================================================================
# 9. 전체 그리드 스윕 - rate 10개 x W1/W2 5개 = 50조합
#    (method1_pc_max_da_rt_mlr.py 7절과 동일, solve 결과는 캐시에 저장돼서
#     Fig.6/Fig.8에서 겹치는 조합은 다시 풀지 않는다)
# =====================================================================
all_results = []

for penalty_rate in GRID_PENALTY_RATES:
    print(f"\n{'='*60}")
    print(f"  rate = {penalty_rate}  (PC = rate*max(DA, RT), MLR)")
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
# 10. 그리드 결과 표 + CSV 저장
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

grid_out_dir = os.path.join(BASE_DIR, "results", "simulation_output")
os.makedirs(grid_out_dir, exist_ok=True)
grid_csv_path = os.path.join(grid_out_dir, "fig3_method1_pc_max_da_rt_mlr.csv")
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
# 11. Best (rate, W1/W2) 자동 탐색 - |ΔnRMSE|+|ΔGap| 최소 조합
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
      f"-> Fig.6/Fig.8에서 이 rate를 그대로 씀")


# =====================================================================
# 12. Fig.6 데이터 생성 - rate=BEST_RATE 고정, W1/W2 10개점 스윕
#     (그리드와 겹치는 5개 조합은 캐시에서 재사용 - MILP 다시 안 풂)
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  Fig.6 데이터 생성 (rate={BEST_RATE} 고정 — 전체 그리드에서 자동 탐색된 최적값)")
print(f"{'=' * 60}")

fig6_mlr_gap = compute_gap(mlr_pred_flat, BEST_RATE)
fig6_nrmse_list = [mlr_nrmse]
fig6_gap_list = [fig6_mlr_gap]
print(f"  MLR: nRMSE={mlr_nrmse:.2f}%, Gap={fig6_mlr_gap:.2f}%  "
      f"(논문: nRMSE={PAPER_MLR_NRMSE:.2f}%, Gap={PAPER_MLR_GAP:.2f}%)")

for idx, (W1, W2) in enumerate(FIG6_W_RATIOS):
    nrmse, gap = get_proposed_result(BEST_RATE, W1, W2)
    fig6_nrmse_list.append(nrmse)
    fig6_gap_list.append(gap)
    print(f"  {W1}/{W2}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}%  "
          f"(논문: nRMSE={FIG6_PAPER_NRMSE[idx]:.2f}%, Gap={FIG6_PAPER_GAP[idx]:.2f}%)")


# =====================================================================
# 13. Fig.8 데이터 생성 - W1/W2=1/1 고정, rate 0.0~1.5 스윕
#     (그리드와 겹치는 rate는 캐시에서 재사용)
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  Fig.8 데이터 생성 (W1/W2={FIG8_W1:.0f}/{FIG8_W2:.0f} 고정, rate 스윕)")
print(f"{'=' * 60}")

fig8_mlr_gap_list = []
fig8_prop_nrmse_list = []
fig8_prop_gap_list = []

for rate in FIG8_PENALTY_RATES:
    mlr_gap_r = compute_gap(mlr_pred_flat, rate)
    fig8_mlr_gap_list.append(mlr_gap_r)

    prop_nrmse_r, prop_gap_r = get_proposed_result(rate, FIG8_W1, FIG8_W2)
    fig8_prop_nrmse_list.append(prop_nrmse_r)
    fig8_prop_gap_list.append(prop_gap_r)

    print(f"  rate={rate:.1f}: MLR Gap={mlr_gap_r:.2f}%, "
          f"제안 nRMSE={prop_nrmse_r:.2f}%, Gap={prop_gap_r:.2f}%")


# =====================================================================
# 14. Fig.6/Fig.8 CSV 저장
# =====================================================================
fig6_paper_nrmse_all = [PAPER_MLR_NRMSE] + FIG6_PAPER_NRMSE
fig6_paper_gap_all = [PAPER_MLR_GAP] + FIG6_PAPER_GAP

fig6_csv_path = os.path.join(grid_out_dir, "fig6_method1_pc_max_da_rt_mlr_v2.csv")
with open(fig6_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"])
    for label, nrmse, gap, p_nrmse, p_gap in zip(
            FIG6_LABELS, fig6_nrmse_list, fig6_gap_list, fig6_paper_nrmse_all, fig6_paper_gap_all):
        writer.writerow([label, round(nrmse, 4), round(gap, 4), p_nrmse, p_gap])
print(f"\nsaved: {fig6_csv_path}")

fig8_csv_path = os.path.join(grid_out_dir, "fig8_method1_pc_max_da_rt_mlr_v2.csv")
with open(fig8_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Rate", "MLR_nRMSE", "MLR_Gap", "Proposed_nRMSE", "Proposed_Gap",
                      "Paper_MLR_nRMSE", "Paper_MLR_Gap", "Paper_Proposed_nRMSE", "Paper_Proposed_Gap"])
    for i, rate in enumerate(FIG8_PENALTY_RATES):
        writer.writerow([rate, round(mlr_nrmse, 4), round(fig8_mlr_gap_list[i], 4),
                         round(fig8_prop_nrmse_list[i], 4), round(fig8_prop_gap_list[i], 4),
                         FIG8_PAPER_MLR_NRMSE[i], FIG8_PAPER_MLR_GAP[i],
                         FIG8_PAPER_PROP_NRMSE[i], FIG8_PAPER_PROP_GAP[i]])
print(f"saved: {fig8_csv_path}")


# =====================================================================
# 15. Fig.6 그리기 - dual y축 (nRMSE 왼쪽, Gap 오른쪽)
# =====================================================================
results_out_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_out_dir, exist_ok=True)

x6 = list(range(len(FIG6_LABELS)))

figure6, axis_left = plt.subplots(figsize=(10, 6))
figure6.suptitle(f"Fig.6 재현 — MLR 제안 모형, rate={BEST_RATE}(자동탐색 최적값), W1/W2 스윕 (dual y축)", fontsize=13)

axis_right = axis_left.twinx()

line1 = axis_left.plot(x6, fig6_paper_nrmse_all, marker="o", color="#2a78d6",
                        linestyle="-", label="논문 nRMSE")
line2 = axis_left.plot(x6, fig6_nrmse_list, marker="o", color="#0b3d78",
                        linestyle="-", label="재현 nRMSE")

line3 = axis_right.plot(x6, fig6_paper_gap_all, marker="s", color="#eb9834",
                         linestyle="--", label="논문 Gap")
line4 = axis_right.plot(x6, fig6_gap_list, marker="s", color="#b34700",
                         linestyle="--", label="재현 Gap")

axis_left.set_xticks(x6)
axis_left.set_xticklabels(FIG6_LABELS)
axis_left.set_xlabel("W1/W2")
axis_left.set_ylabel("nRMSE (%)", color="#0b3d78")
axis_right.set_ylabel("Optimality Gap (%)", color="#b34700")
axis_left.tick_params(axis="y", labelcolor="#0b3d78")
axis_right.tick_params(axis="y", labelcolor="#b34700")
axis_left.grid(True, alpha=0.3)

all_lines = line1 + line2 + line3 + line4
all_labels = [one_line.get_label() for one_line in all_lines]
axis_left.legend(all_lines, all_labels, loc="upper left", fontsize=9)

figure6.tight_layout()
fig6_png_path = os.path.join(results_out_dir, "fig6_method1_pc_max_da_rt_mlr_v2.png")
figure6.savefig(fig6_png_path, dpi=150)
print(f"\nsaved: {fig6_png_path}")


# =====================================================================
# 16. Fig.8 그리기 - nRMSE / Gap 좌우 분할, BEST_RATE 지점을 점선으로 강조,
#     논문 곡선(점선, 육안판독 근사치)도 0~100% 구간까지 같이 표시
# =====================================================================
def _to_nan(values):
    return [np.nan if v is None else v for v in values]

paper_mlr_nrmse_arr = _to_nan(FIG8_PAPER_MLR_NRMSE)
paper_prop_nrmse_arr = _to_nan(FIG8_PAPER_PROP_NRMSE)
paper_mlr_gap_arr = _to_nan(FIG8_PAPER_MLR_GAP)
paper_prop_gap_arr = _to_nan(FIG8_PAPER_PROP_GAP)

x8_labels = [f"{int(round(r * 100))}%" for r in FIG8_PENALTY_RATES]
x8 = list(range(len(x8_labels)))
highlight_idx = FIG8_PENALTY_RATES.index(BEST_RATE)

figure8, (axis_nrmse, axis_gap) = plt.subplots(1, 2, figsize=(13, 5))
figure8.suptitle(f"Fig.8 재현 — MLR 제안 모형, W1/W2={FIG8_W1:.0f}/{FIG8_W2:.0f}, rate 스윕 "
                  "(점선=논문 육안판독 근사치, 0~100%만 존재)", fontsize=13)

axis_nrmse.plot(x8, paper_mlr_nrmse_arr, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 MLR")
axis_nrmse.plot(x8, paper_prop_nrmse_arr, linestyle="--", color="#eb6834", alpha=0.6, label="논문 제안모형")
axis_nrmse.axhline(mlr_nrmse, color="#2a78d6", alpha=0.9, label="재현 MLR")
axis_nrmse.plot(x8, fig8_prop_nrmse_list, marker="o", color="#eb6834", label="재현 제안모형")
axis_nrmse.axvline(highlight_idx, color="gray", linestyle=":", alpha=0.7)
axis_nrmse.set_xticks(x8)
axis_nrmse.set_xticklabels(x8_labels, rotation=45)
axis_nrmse.set_xlabel("벌금비용률(penalty cost rate)")
axis_nrmse.set_ylabel("nRMSE (%)")
axis_nrmse.set_title(f"nRMSE 비교 (점선=rate={int(round(BEST_RATE*100))}%)")
axis_nrmse.grid(True, alpha=0.3)
axis_nrmse.legend(fontsize=8)

axis_gap.plot(x8, paper_mlr_gap_arr, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 MLR")
axis_gap.plot(x8, paper_prop_gap_arr, linestyle="--", color="#eb6834", alpha=0.6, label="논문 제안모형")
axis_gap.plot(x8, fig8_mlr_gap_list, marker="o", color="#2a78d6", label="재현 MLR")
axis_gap.plot(x8, fig8_prop_gap_list, marker="o", color="#eb6834", label="재현 제안모형")
axis_gap.axvline(highlight_idx, color="gray", linestyle=":", alpha=0.7)
axis_gap.set_xticks(x8)
axis_gap.set_xticklabels(x8_labels, rotation=45)
axis_gap.set_xlabel("벌금비용률(penalty cost rate)")
axis_gap.set_ylabel("Optimality Gap (%)")
axis_gap.set_title(f"Optimality Gap 비교 (점선=rate={int(round(BEST_RATE*100))}%)")
axis_gap.grid(True, alpha=0.3)
axis_gap.legend(fontsize=8)

figure8.tight_layout()
fig8_png_path = os.path.join(results_out_dir, "fig8_method1_pc_max_da_rt_mlr_v2.png")
figure8.savefig(fig8_png_path, dpi=150)
print(f"saved: {fig8_png_path}")

print("\n완료.")
