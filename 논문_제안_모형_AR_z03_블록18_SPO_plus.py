

# -*- coding: utf-8 -*-



# =====================================================================

# 논문_제안_모형_AR_z03_블록18_SPO_plus.py

#

# 방법 4: Smart SPO+ loss (Elmachtoub & Grigas 2022)

# 기존 W1/W2 혼합 대신 SPO+ loss로 3항 식의 구조적 arbitrage가

# 어떻게 바뀌는지 검증.

#

# SPO+ 수식:

#   의결손실(ζ) 상한: oracle_profit − realized_profit

#   oracle = max{commit=0, commit=actual, commit=1.0}  3후보

#

#   각 후보 k에 대해:

#     ζ ≥ profit_k − realized_profit   (i = 1..n_obs)

#

#   realized = scale × (DA·x + RP·surplus − PC·shortage)

#   profit_k = scale × 각 후보별 profit

#

#   목적함수:

#     min W1 × Σζ_i/n + W2 × Σ|e|_i/n

#

#   기존 3항 스크립트와 동일한 데이터/AR 구조 사용.

# =====================================================================



import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np

import pandas as pd

from scipy import sparse

from scipy.optimize import Bounds, LinearConstraint, milp, linprog





# =====================================================================

# 0. 설정값

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

PENALTY_RATE = 0.5



# Paper Table 3 W1/W2 ratios (SPO+ 가중치 / 예측오차 가중치)

W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1),

            (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]



# Paper baseline values for comparison

PAPER_AR_NRMSE = 34.76

PAPER_AR_GAP = 15.04

PAPER_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95,

               46.11, 48.27, 49.21, 49.61, 50.07]

PAPER_GAP = [13.91, 13.42, 12.71, 11.88, 11.44,

             11.38, 11.38, 11.36, 11.36, 11.36]





# =====================================================================

# 1. 데이터 읽기 + 낮 시간대만 남기기

# =====================================================================

raw_table = pd.read_csv(MERGED_FILE)

raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])



is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)

daylight_table = raw_table[is_daylight].copy()

daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START





# =====================================================================

# 2. 이력/학습/테스트 구간으로 자르기

# =====================================================================

is_history = daylight_table["local_date"] == HISTORY_DATE

history_rows = daylight_table[is_history].copy().sort_values("hour_idx")



is_train = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)

train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])



is_test = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)

test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])



print(f"이력: {len(history_rows)}행, 학습: {len(train_rows)}행, 테스트: {len(test_rows)}행")





# =====================================================================

# 3. (날짜 x 12시간) 배열로 만들기

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

# 4. AR 학습용 입력(X) - 논문 Eq.(3): 직전 하루의 12시간

# =====================================================================

history_and_train_solar = np.vstack([history_solar, train_solar])



n_ar_rows = n_train_days

ar_intercept = np.ones((n_ar_rows, 1))

ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))

for d in range(n_ar_rows):

    ar_lag[d] = history_and_train_solar[d][::-1]



ar_design_matrix = np.hstack([ar_intercept, ar_lag])

n_features = ar_design_matrix.shape[1]



scale = CAPACITY_MW * DURATION_HOURS

actual_flat = test_solar.flatten()

da_flat = test_da_price.flatten()

rt_flat = test_rt_price.flatten()

n_test_obs = len(actual_flat)





# =====================================================================

# 5. SPO+ MILP — 시간대별 계수 학습

#    변수: β, x_i, yp_i, ym_i, e_i, ζ_i  (i = 1..n_obs)

#

#    SPO+ loss = max over 3 oracle candidates:

#      (1) ζ ≥ scale*(RP*a) − realized

#      (2) ζ ≥ scale*(DP*a) − realized

#      (3) ζ ≥ scale*(DP*1 + RP*s1 − PC*y1) − realized

#          where s1 = max(a−1, 0), y1 = max(1−a, 0)

#          → simplifies to: DP + RP*(a−1) − PC*(1−a) = (DP−RP) + (RP+PC)*(a−1)

#

#    realized = scale*(DP*x + RP*yp − PC*ym)

#

#    목적: min W1*Σζ/n + W2*Σ|e|/n

# =====================================================================

results = []



for widx, (W1, W2) in enumerate(W_RATIOS):

    print(f"\n--- SPO+: W1={W1}, W2={W2} ---")



    coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))



    for hour in range(HOURS_PER_DAY):

        y_h = train_solar[:, hour]

        da_h = train_da_price[:, hour]

        rt_h = train_rt_price[:, hour]

        n_obs = n_ar_rows



        pc_h = PENALTY_RATE * da_h



        # 변수: β(n_features), x(n_obs), yp(n_obs), ym(n_obs), ζ(n_obs)

        # 기존 3항 스크립트와 동일하게 e_i 분리 변수 없이,

        # surplus_cost/shortage_cost에 W2 항을 포함

        n_variables = n_features + 4 * n_obs



        beta_s = 0

        x_s = n_features

        yp_s = n_features + n_obs

        ym_s = n_features + 2 * n_obs

        zeta_s = n_features + 3 * n_obs



        # 목적함수: SPO+ ζ (의결손실) + W2 예측오차(서플러스/쇼트리지 비용에 포함)

        objective = np.zeros(n_variables)

        objective[zeta_s:zeta_s + n_obs] = W1 / n_obs

        # W2 예측오차 항: 기존 3항 스크립트와 동일하게 surplus_cost/shortage_cost에 포함

        # surplus(초과예측): 실제 > 예측 → 오차 = yp

        # shortage(부족예측): 예측 > 실제 → 오차 = ym

        for i in range(n_obs):

            objective[yp_s + i] += W2 / n_obs  # |e|의 양의 부분

            objective[ym_s + i] += W2 / n_obs  # |e|의 음의 부분



        # ── 제약 1: AR 연결  x_i = Xβ  ──

        eq_a = sparse.lil_matrix((n_obs, n_variables))

        eq_a[:, beta_s:beta_s + n_features] = -sparse.csr_matrix(ar_design_matrix)

        eq_a[:, x_s:x_s + n_obs] = sparse.eye(n_obs, format="lil")

        eq_rhs_1 = np.zeros(n_obs)



        # ── 제약 2: 분해  x_i + yp_i − ym_i = y_h  ──

        eq_b = sparse.lil_matrix((n_obs, n_variables))

        eq_b[:, x_s:x_s + n_obs] = sparse.eye(n_obs, format="lil")

        eq_b[:, yp_s:yp_s + n_obs] = sparse.eye(n_obs, format="lil")

        eq_b[:, ym_s:ym_s + n_obs] = -sparse.eye(n_obs, format="lil")

        eq_rhs_2 = y_h



        # ── 제약 3: SPO+ loss 상한 (3후보 × n_obs) ──

        # realized_i / scale = da*x_i + rt*yp_i − pc*ym_i

        #

        # 후보 1 (commit=0): profit = RP*a

        #   ζ ≥ RP*a − (DA*x + RP*yp − PC*ym)

        #   ζ + DA*x + RP*yp − PC*ym ≥ RP*a

        #

        # 후보 2 (commit=actual): profit = DP*a

        #   ζ + DA*x + RP*yp − PC*ym ≥ DP*a

        #

        # 후보 3 (commit=1): profit = DP + (RP+PC)*(a−1)

        #   ζ + DA*x + RP*yp − PC*ym ≥ DP + (RP+PC)*(a−1)

        #                       = DP − (RP+PC)*(1−a)

        spo_rhs = np.zeros(3 * n_obs)

        spo_mat = sparse.lil_matrix((3 * n_obs, n_variables))



        for i in range(n_obs):

            spo_mat[3*i, zeta_s + i] = 1.0

            spo_mat[3*i, x_s + i] = scale * da_h[i]

            spo_mat[3*i, yp_s + i] = scale * rt_h[i]

            spo_mat[3*i, ym_s + i] = -scale * pc_h[i]

            spo_rhs[3*i] = scale * rt_h[i] * y_h[i]  # 후보 1: RP*a



            spo_mat[3*i+1, zeta_s + i] = 1.0

            spo_mat[3*i+1, x_s + i] = scale * da_h[i]

            spo_mat[3*i+1, yp_s + i] = scale * rt_h[i]

            spo_mat[3*i+1, ym_s + i] = -scale * pc_h[i]

            spo_rhs[3*i+1] = scale * da_h[i] * y_h[i]  # 후보 2: DP*a



            spo_mat[3*i+2, zeta_s + i] = 1.0

            spo_mat[3*i+2, x_s + i] = scale * da_h[i]

            spo_mat[3*i+2, yp_s + i] = scale * rt_h[i]

            spo_mat[3*i+2, ym_s + i] = -scale * pc_h[i]

            spo_rhs[3*i+2] = scale * (da_h[i] + (rt_h[i] + pc_h[i]) * (y_h[i] - 1.0))



        # ── 제약 4: ζ ≥ 0 (의결손실 음수 불가) ──

        zeta_nonneg = sparse.lil_matrix((n_obs, n_variables))

        zeta_nonneg[:, zeta_s:zeta_s + n_obs] = sparse.eye(n_obs, format="lil")



        # 모든 제약 합치기

        all_mat = sparse.vstack([

            eq_a, eq_b,

            spo_mat.tocsr(),

            zeta_nonneg.tocsr()

        ], format="csr")

        all_rhs = np.concatenate([

            eq_rhs_1, eq_rhs_2,

            spo_rhs,

            np.zeros(n_obs)

        ])



        # 등식/부등식 구분

        eq_count = 2 * n_obs

        eq_con = LinearConstraint(all_mat[:eq_count], all_rhs[:eq_count], all_rhs[:eq_count])

        ineq_con = LinearConstraint(all_mat[eq_count:], all_rhs[eq_count:], np.full(all_rhs[eq_count:].size, np.inf))



        all_con = [eq_con, ineq_con]



        # 변수 경계: β, x, yp, ym, ζ

        lb = np.concatenate([

            np.full(n_features, -np.inf),  # β: 무제약

            np.zeros(3 * n_obs),           # x, yp, ym ≥ 0

            np.zeros(n_obs)                # ζ ≥ 0

        ])

        ub = np.concatenate([

            np.full(n_features, np.inf),   # β: 무제약

            np.ones(3 * n_obs),            # x, yp, ym ≤ 1

            np.full(n_obs, np.inf)         # ζ 상한 없음

        ])

        bounds = Bounds(lb, ub)



        milp_result = milp(c=objective, bounds=bounds,

                           constraints=all_con, options={"mip_rel_gap": 1e-9})

        if not milp_result.success:

            print(f"  시간대 {hour}: MILP 실패 (status={milp_result.message})")

            coefficients_by_hour[hour] = 0.0

        else:

            coefficients_by_hour[hour] = milp_result.x[beta_s:beta_s + n_features]



    # ── 테스트 예측 ──

    test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))

    prev_day = train_solar[-1]

    for d in range(n_test_days):

        feat = np.concatenate([[1.0], prev_day[::-1]])

        for h in range(HOURS_PER_DAY):

            raw = np.dot(coefficients_by_hour[h], feat)

            test_forecast[d, h] = np.clip(raw, 0, 1)

        prev_day = test_solar[d]



    pred_flat = test_forecast.flatten()



    # nRMSE

    rmse = np.sqrt(np.mean((actual_flat - pred_flat)**2))

    nrmse = 100 * rmse / np.mean(actual_flat)



    # Gap (3항, 3후보 오라클)

    sum_realized = 0.0; sum_oracle = 0.0

    for i in range(n_test_obs):

        a = actual_flat[i]; x = pred_flat[i]; dp = da_flat[i]; rp = rt_flat[i]

        pc = PENALTY_RATE * dp

        mismatch = a - x; surplus = max(mismatch, 0); shortage = max(-mismatch, 0)

        realized = scale * (dp * x + rp * surplus - pc * shortage)

        sum_realized += realized

        p0 = scale * rp * a

        pa = scale * dp * a

        surplus_1 = max(a - 1.0, 0); shortage_1 = max(1.0 - a, 0)

        p1 = scale * (dp * 1.0 + rp * surplus_1 - pc * shortage_1)

        oracle = max(p0, pa, p1)

        sum_oracle += oracle



    gap = 100 * (sum_oracle - sum_realized) / sum_oracle



    label = f"{W1}/{W2}"

    results.append((label, W1, W2, nrmse, gap))



    print(f"  {label}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}% "

          f"(paper: nRMSE={PAPER_NRMSE[widx]:.2f}%, Gap={PAPER_GAP[widx]:.2f}%)")

    print(f"  commit mean={pred_flat.mean():.4f}, commit=1.0 count={(pred_flat>0.999).sum()}/{n_test_obs}")





# =====================================================================

# 6. 기본 AR (baseline) — 3후보 오라클

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

ar_rmse = np.sqrt(np.mean((actual_flat - ar_pred_flat)**2))

ar_nrmse = 100 * ar_rmse / np.mean(actual_flat)



ar_sum_realized = 0.0; ar_sum_oracle = 0.0

for i in range(n_test_obs):

    a = actual_flat[i]; x = ar_pred_flat[i]; dp = da_flat[i]; rp = rt_flat[i]

    pc = PENALTY_RATE * dp

    mismatch = a - x; surplus = max(mismatch, 0); shortage = max(-mismatch, 0)

    realized = scale * (dp * x + rp * surplus - pc * shortage)

    ar_sum_realized += realized

    p0 = scale * rp * a

    pa = scale * dp * a

    surplus_1 = max(a - 1.0, 0); shortage_1 = max(1.0 - a, 0)

    p1 = scale * (dp * 1.0 + rp * surplus_1 - pc * shortage_1)

    oracle = max(p0, pa, p1)

    ar_sum_oracle += oracle

ar_gap = 100 * (ar_sum_oracle - ar_sum_realized) / ar_sum_oracle



print(f"\n=== 기본 AR baseline (3후보 오라클) ===")

print(f"  nRMSE={ar_nrmse:.2f}%, Gap={ar_gap:.2f}% (paper: nRMSE={PAPER_AR_NRMSE:.2f}%, Gap={PAPER_AR_GAP:.2f}%)")





# =====================================================================

# 7. 전체 결과 표 출력

# =====================================================================

print()

print("=" * 100)

print(f"{'Label':>6} {'W1':>4} {'W2':>4} {'nRMSE':>10} {'Gap':>10} {'Paper nRMSE':>12} {'Paper Gap':>12} {'Δ nRMSE':>10} {'Δ Gap':>10}")

print("-" * 100)

print(f"{'AR':>6} {'':>4} {'':>4} {ar_nrmse:>9.2f}% {ar_gap:>9.2f}% {PAPER_AR_NRMSE:>11.2f}% {PAPER_AR_GAP:>11.2f}% {ar_nrmse-PAPER_AR_NRMSE:>+9.2f}%p {ar_gap-PAPER_AR_GAP:>+9.2f}%p")

for label, W1, W2, nrmse, gap in results:

    idx = W_RATIOS.index((W1, W2))

    print(f"{label:>6} {W1:>4} {W2:>4} {nrmse:>9.2f}% {gap:>9.2f}% {PAPER_NRMSE[idx]:>11.2f}% {PAPER_GAP[idx]:>11.2f}% {nrmse-PAPER_NRMSE[idx]:>+9.2f}%p {gap-PAPER_GAP[idx]:>+9.2f}%p")

print("=" * 100)



# Best W1/W2 (closest to paper)

best_idx = min(range(len(results)), key=lambda i: abs(results[i][3] - PAPER_NRMSE[i]) + abs(results[i][4] - PAPER_GAP[i]))

best_label = results[best_idx][0]

print(f"\nBest match: W1/W2 = {best_label} (ΔnRMSE={results[best_idx][3] - PAPER_NRMSE[best_idx]:+.2f}%p, ΔGap={results[best_idx][4] - PAPER_GAP[best_idx]:+.2f}%p)")



# CSV 저장

import csv

csv_path = os.path.join(os.path.dirname(MERGED_FILE), "results", "simulation_output", "fig3_SPOplus_3term.csv")

os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["Label", "W1", "W2", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap", "Delta_nRMSE", "Delta_Gap"])

    writer.writerow(["AR", "", "", ar_nrmse, ar_gap, PAPER_AR_NRMSE, PAPER_AR_GAP,

                      ar_nrmse - PAPER_AR_NRMSE, ar_gap - PAPER_AR_GAP])

    for label, W1, W2, nrmse, gap in results:

        idx = W_RATIOS.index((W1, W2))

        writer.writerow([label, W1, W2, round(nrmse, 2), round(gap, 2),

                         PAPER_NRMSE[idx], PAPER_GAP[idx],

                         round(nrmse - PAPER_NRMSE[idx], 2), round(gap - PAPER_GAP[idx], 2)])

print(f"\n저장: {csv_path}")