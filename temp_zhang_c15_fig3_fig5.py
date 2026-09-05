
# -*- coding: utf-8 -*-

# =====================================================================
# temp_zhang_c15_fig3_fig5.py
#
# 목적: 방법 3 (Zhang et al. 비대칭 시장가격, rho_plus=c*RT, rho_minus=RT,
#       zhang_asymmetric_3term.py 참고)에서 c=1.0/1.2/1.5/2.0/3.0 전체를
#       스윕한 결과(results/simulation_output/fig3_zhang_asymmetric_3term.csv,
#       논문AR_정리2.md 6.2절) 가장 성능이 좋은 c=1.5 하나만 골라 재사용한다.
#
#   - zhang_asymmetric_3term.py 처럼 C_RATES 5개 x W_RATIOS 10개를 매번
#     다 스윕하면 MILP를 600번(12시간 x 10 W비율 x 5 c) 풀어야 해서 느리다.
#     이미 최적 c=1.5가 확인됐으므로, 이 스크립트는 c=1.5로 고정해서:
#       · Fig.3: c=1.5 고정, W1/W2 스윕 (AR 포함 11개 지점, 논문 Fig.3과 비교)
#       · Fig.5: W1/W2=1/1 고정, c 자체를 스윕(1.0~3.0) — c=1.5 지점을
#                세로 점선으로 강조. (Zhang 모형에는 별도의 "penalty rate"가
#                없고 c가 그 역할을 하므로, method1의 rate 스윕 Fig.5를
#                c 스윕으로 그대로 대응시킴)
#
# 근거: MILP/오라클 계산 로직은 zhang_asymmetric_3term.py 5~7절을 그대로
#       재사용(공식만 동일, c 값과 W1/W2 조합만 재구성) - 형식은
#       temp_method1_rate09_fig3_fig5.py 를 그대로 따름.
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

# 이미 zhang_asymmetric_3term.py 전체 스윕(c=1.0/1.2/1.5/2.0/3.0)에서
# 확인된 최적값 (논문AR_정리2.md 6.2절 "c=1.5에서 논문 재현 성공")
BEST_C = 1.5

# --- Fig.3 설정: c=1.5 고정, W1/W2 10개점 스윕 (AR 포함 11개 지점) ---
FIG3_LABELS = ["AR", "1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]
FIG3_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG3_PAPER_NRMSE = [34.76, 34.89, 35.14, 36.28, 41.09, 44.95, 46.11, 48.27, 49.21, 49.61, 50.07]
FIG3_PAPER_GAP   = [15.04, 13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36]

# --- Fig.5 설정: W1/W2=1/1 고정, c 자체를 스윕 (Zhang엔 penalty rate가
#     없어 c가 그 역할을 함) - BEST_C=1.5 지점을 세로 점선으로 강조 ---
FIG5_W1, FIG5_W2 = 1, 1
FIG5_C_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0]
FIG5_HIGHLIGHT_C = BEST_C


# =====================================================================
# 2. 데이터 로딩 및 (날짜 x 12) 배열 구성 (zhang_asymmetric_3term.py 1~4절과 동일)
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
# 3. AR 학습용 입력 (절편 + lag 12개)
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
# 4. Gap 계산 함수 (Zhang 3항: DA*x + rho_-*surplus - rho_+*shortage)
#    rho_- = RT, rho_+ = c*RT, oracle 3후보 {0, actual, 1.0}
#    (zhang_asymmetric_3term.py 5절과 동일)
# =====================================================================
def compute_zhang_gap(pred_flat, c_rate):
    sum_realized = 0.0
    sum_oracle = 0.0
    for i in range(n_test_obs):
        a = actual_flat[i]; x = pred_flat[i]
        dp = da_flat[i]; rp = rt_flat[i]
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

    return 100 * (sum_oracle - sum_realized) / sum_oracle


# =====================================================================
# 5. 기본 AR (c와 무관 — 회귀 예측만으로 결정됨, 한 번만 계산)
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
# 6. (c_rate, W1, W2) 하나를 받아 Zhang 제안모형 MILP를 풀고 nRMSE/Gap을
#    반환하는 함수. zhang_asymmetric_3term.py 7절 로직을 그대로 재사용해서
#    Fig.3(c 고정, W1/W2 스윕)와 Fig.5(W1/W2 고정, c 스윕) 양쪽에서 호출한다.
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
# 7. Fig.3 데이터 생성 — c=1.5(BEST_C) 고정, W1/W2 스윕
# =====================================================================
print(f"\n{'=' * 60}")
print(f"  Fig.3 데이터 생성 (c={BEST_C} 고정 — 이미 확인된 최적값)")
print(f"{'=' * 60}")

fig3_ar_gap = compute_zhang_gap(ar_pred_flat, BEST_C)
fig3_nrmse_list = [ar_nrmse]
fig3_gap_list = [fig3_ar_gap]
print(f"  AR: nRMSE={ar_nrmse:.2f}%, Gap={fig3_ar_gap:.2f}%")

for (W1, W2) in FIG3_W_RATIOS:
    nrmse, gap = run_proposed_model(BEST_C, W1, W2)
    fig3_nrmse_list.append(nrmse)
    fig3_gap_list.append(gap)
    print(f"  {W1}/{W2}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}%")


# =====================================================================
# 8. Fig.5 데이터 생성 — W1/W2=1/1 고정, c 자체를 스윕
#    (Zhang엔 penalty rate가 따로 없어 c가 그 역할을 함)
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

    prop_nrmse_c, prop_gap_c = run_proposed_model(c_val, FIG5_W1, FIG5_W2)
    fig5_prop_nrmse_list.append(prop_nrmse_c)
    fig5_prop_gap_list.append(prop_gap_c)

    print(f"  c={c_val:.1f}: AR Gap={ar_gap_c:.2f}%, "
          f"제안 nRMSE={prop_nrmse_c:.2f}%, Gap={prop_gap_c:.2f}%")


# =====================================================================
# 9. 결과 CSV 저장
# =====================================================================
out_dir = os.path.join(BASE_DIR, "results", "simulation_output")
os.makedirs(out_dir, exist_ok=True)

fig3_csv_path = os.path.join(out_dir, "temp_fig3_zhang_c15.csv")
with open(fig3_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"])
    for label, nrmse, gap, p_nrmse, p_gap in zip(
            FIG3_LABELS, fig3_nrmse_list, fig3_gap_list, FIG3_PAPER_NRMSE, FIG3_PAPER_GAP):
        writer.writerow([label, round(nrmse, 2), round(gap, 2), p_nrmse, p_gap])
print(f"\nsaved: {fig3_csv_path}")

fig5_csv_path = os.path.join(out_dir, "temp_fig5_zhang_c15.csv")
with open(fig5_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["c", "AR_Gap", "Proposed_nRMSE", "Proposed_Gap"])
    for c_val, ag, pn, pg in zip(FIG5_C_VALUES, fig5_ar_gap_list, fig5_prop_nrmse_list, fig5_prop_gap_list):
        writer.writerow([c_val, round(ag, 2), round(pn, 2), round(pg, 2)])
print(f"saved: {fig5_csv_path}")


# =====================================================================
# 10. Fig.3 그리기 — nRMSE / Gap 나란히, 논문값과 비교
# =====================================================================
x3 = list(range(len(FIG3_LABELS)))

fig3, (ax3_nrmse, ax3_gap) = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(f"Fig.3 재현 — Zhang 비대칭(rho_+=c*RT), c={BEST_C}, W1/W2 스윕", fontsize=14)

ax3_nrmse.plot(x3, FIG3_PAPER_NRMSE, marker="o", color="#2a78d6", label="논문")
ax3_nrmse.plot(x3, fig3_nrmse_list, marker="o", color="#eb6834", label="재현")
ax3_nrmse.set_xticks(x3)
ax3_nrmse.set_xticklabels(FIG3_LABELS)
ax3_nrmse.set_xlabel("W1/W2")
ax3_nrmse.set_ylabel("nRMSE (%)")
ax3_nrmse.set_title("nRMSE 비교")
ax3_nrmse.grid(True, alpha=0.3)
ax3_nrmse.legend()

ax3_gap.plot(x3, FIG3_PAPER_GAP, marker="o", color="#2a78d6", label="논문")
ax3_gap.plot(x3, fig3_gap_list, marker="o", color="#eb6834", label="재현")
ax3_gap.set_xticks(x3)
ax3_gap.set_xticklabels(FIG3_LABELS)
ax3_gap.set_xlabel("W1/W2")
ax3_gap.set_ylabel("Optimality Gap (%)")
ax3_gap.set_title("Optimality Gap 비교")
ax3_gap.grid(True, alpha=0.3)
ax3_gap.legend()

fig3.tight_layout()
fig3_png_path = os.path.join(BASE_DIR, "results", "temp_fig3_zhang_c15.png")
os.makedirs(os.path.dirname(fig3_png_path), exist_ok=True)
fig3.savefig(fig3_png_path, dpi=150)
print(f"saved: {fig3_png_path}")


# =====================================================================
# 11. Fig.5 그리기 — nRMSE / Gap 나란히, c=BEST_C 지점을 세로 점선으로 강조
#     (논문에는 Zhang c 스윕 자체가 없으므로 논문 비교선 없이 재현값만 표시)
# =====================================================================
x5 = list(range(len(FIG5_C_VALUES)))
x5_labels = [f"{c:.1f}" for c in FIG5_C_VALUES]
highlight_idx = FIG5_C_VALUES.index(FIG5_HIGHLIGHT_C)

fig5, (ax5_nrmse, ax5_gap) = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle(f"Fig.5 재현 — Zhang 비대칭(rho_+=c*RT), W1/W2={FIG5_W1}/{FIG5_W2}, c 스윕", fontsize=14)

ax5_nrmse.axhline(ar_nrmse, linestyle="--", color="#2a78d6", alpha=0.6, label="재현 AR")
ax5_nrmse.plot(x5, fig5_prop_nrmse_list, marker="o", color="#eb6834", label="재현 제안모형")
ax5_nrmse.axvline(highlight_idx, color="gray", linestyle=":", alpha=0.7)
ax5_nrmse.set_xticks(x5)
ax5_nrmse.set_xticklabels(x5_labels)
ax5_nrmse.set_xlabel("c (rho_plus = c * RT)")
ax5_nrmse.set_ylabel("nRMSE (%)")
ax5_nrmse.set_title(f"nRMSE 비교 (점선=c={FIG5_HIGHLIGHT_C})")
ax5_nrmse.grid(True, alpha=0.3)
ax5_nrmse.legend(fontsize=9)

ax5_gap.plot(x5, fig5_ar_gap_list, marker="o", color="#2a78d6", label="재현 AR")
ax5_gap.plot(x5, fig5_prop_gap_list, marker="o", color="#eb6834", label="재현 제안모형")
ax5_gap.axvline(highlight_idx, color="gray", linestyle=":", alpha=0.7)
ax5_gap.set_xticks(x5)
ax5_gap.set_xticklabels(x5_labels)
ax5_gap.set_xlabel("c (rho_plus = c * RT)")
ax5_gap.set_ylabel("Optimality Gap (%)")
ax5_gap.set_title(f"Optimality Gap 비교 (점선=c={FIG5_HIGHLIGHT_C})")
ax5_gap.grid(True, alpha=0.3)
ax5_gap.legend(fontsize=9)

fig5.tight_layout()
fig5_png_path = os.path.join(BASE_DIR, "results", "temp_fig5_zhang_c15.png")
fig5.savefig(fig5_png_path, dpi=150)
print(f"saved: {fig5_png_path}")

print("\n완료.")
