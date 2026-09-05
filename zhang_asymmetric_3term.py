# -*- coding: utf-8 -*-



# =====================================================================

# zhang_asymmetric_3term.py

# 방법 3: Zhang et al. 방식 — ρ₊ ≠ ρ₋ 비대칭 시장가격 모델

#

# 3항 식 profit = DA·x + ρ₋·surplus − ρ₊·shortage

#   ρ₋ = RT (초과 시 실시간 판매)

#   ρ₊ = c·RT (c > 1, 부족 시 상향조정가)

#

# 기존 3항: PC = 0.5·DA → arbitrage(DA−PC>0) 항상 발생

# Zhang:   ρ₊ = c·RT, c>1 → RT>DA일 때 ρ₊>DA → arbitrage 해소

#

# c 값을 스윕(c=1.0, 1.2, 1.5, 2.0, 3.0)하여

#   (a) Fig.3 재현: 각 c마다 W1/W2 sweep

#   (b) arbitrage 해소 여부 확인

# =====================================================================



import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np

import pandas as pd

from scipy import sparse

from scipy.optimize import Bounds, LinearConstraint, milp, linprog

import csv





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



# W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 1), (1, 0)]

W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1),(2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]


C_RATES = [1.0, 1.2, 1.5, 2.0, 3.0]



PAPER_AR_NRMSE = 34.76

PAPER_AR_GAP = 15.04

PAPER_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95, 46.11,48.27, 49.21, 49.61,50.07]

PAPER_GAP = [13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36]



scale = CAPACITY_MW * DURATION_HOURS





# =====================================================================

# 1. 데이터 읽기 + 낮 시간대만 남기기

# =====================================================================

raw_table = pd.read_csv(MERGED_FILE)

raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])



is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)

daylight_table = raw_table[is_daylight].copy()

daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START





# =====================================================================

# 2. 이력/학습/테스트 구간

# =====================================================================

is_history = daylight_table["local_date"] == HISTORY_DATE

history_rows = daylight_table[is_history].copy().sort_values("hour_idx")



is_train = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)

train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])



is_test = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)

test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])



print(f"이력: {len(history_rows)}행, 학습: {len(train_rows)}행, 테스트: {len(test_rows)}행")





# =====================================================================

# 3. (날짜 x 12시간) 배열

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

# 4. AR 학습용 입력(X) - 논문 Eq.(3)

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

# 5. Gap 계산 함수 (Zhang 3항: DA*x + rho_-*surplus - rho_+*shortage)

#    oracle: 3후보 {commit=0, commit=actual, commit=1.0}

# =====================================================================

def compute_zhang_gap(pred_flat, c_rate):

    """

    Zhang 방식 3항 gap 계산.

    ρ₋ = RT, ρ₊ = c·RT

    """

    sum_realized = 0.0

    sum_oracle = 0.0

    for i in range(n_test_obs):

        a = actual_flat[i]

        x = pred_flat[i]

        dp = da_flat[i]

        rp = rt_flat[i]

        rho_plus = c_rate * rp  # shortage cost

        rho_minus = rp           # surplus price



        mismatch = a - x

        surplus = max(mismatch, 0)

        shortage = max(-mismatch, 0)



        realized = scale * (dp * x + rho_minus * surplus - rho_plus * shortage)

        sum_realized += realized



        # Oracle 3후보

        p0 = scale * rho_minus * a  # commit=0

        pa = scale * dp * a         # commit=actual

        s1 = max(a - 1.0, 0)

        y1 = max(1.0 - a, 0)

        p1 = scale * (dp * 1.0 + rho_minus * s1 - rho_plus * y1)  # commit=1

        oracle = max(p0, pa, p1)

        sum_oracle += oracle



    gap = 100 * (sum_oracle - sum_realized) / sum_oracle

    return gap





# =====================================================================

# 6. 기본 AR (baseline) — c_rate=1.0으로 gap 계산

#    (AR은 c와 무관하므로 한 번만 계산)

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



ar_nrmse_for_all_c = [ar_nrmse] * len(C_RATES)

ar_gap_for_all_c = [compute_zhang_gap(ar_pred_flat, c) for c in C_RATES]





# =====================================================================

# 7. c_rate별 W1/W2 sweep

# =====================================================================

all_results = []



for c_rate in C_RATES:

    print(f"\n{'='*60}")

    print(f"  c = {c_rate}  (rho_plus = {c_rate}*RT, rho_minus = RT)")

    print(f"{'='*60}")



    results_c = []



    for widx, (W1, W2) in enumerate(W_RATIOS):



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



            # 목적함수 계수

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



        # Predict test

        test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))

        prev_day = train_solar[-1]

        for d in range(n_test_days):

            feat = np.concatenate([[1.0], prev_day[::-1]])

            for h in range(HOURS_PER_DAY):

                raw = np.dot(coefficients_by_hour[h], feat)

                test_forecast[d, h] = np.clip(raw, 0, 1)

            prev_day = test_solar[d]



        pred_flat = test_forecast.flatten()

        rmse = np.sqrt(np.mean((actual_flat - pred_flat)**2))

        nrmse = 100 * rmse / np.mean(actual_flat)

        gap = compute_zhang_gap(pred_flat, c_rate)



        label = f"{W1}/{W2}"

        results_c.append((label, W1, W2, nrmse, gap))



        print(f"  {label}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}% "

              f"(paper: nRMSE={PAPER_NRMSE[widx]:.2f}%, Gap={PAPER_GAP[widx]:.2f}%)")



    all_results.append((c_rate, results_c))





# =====================================================================

# 8. 전체 결과 표

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



# AR baseline 요약

print(f"\nAR baseline:")

for i, c_rate in enumerate(C_RATES):

    print(f"  c={c_rate}: nRMSE={ar_nrmse_for_all_c[i]:.2f}%, Gap={ar_gap_for_all_c[i]:.2f}% "

          f"(paper: nRMSE={PAPER_AR_NRMSE:.2f}%, Gap={PAPER_AR_GAP:.2f}%)")





# =====================================================================

# 9. CSV 저장

# =====================================================================

csv_path = os.path.join(os.path.dirname(MERGED_FILE), "results", "simulation_output",

                        "fig3_zhang_asymmetric_3term.csv")

os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["c_rate", "Label", "W1", "W2", "nRMSE", "Gap",

                      "Paper_nRMSE", "Paper_Gap", "Delta_nRMSE", "Delta_Gap"])

    for c_rate, results_c in all_results:

        for label, W1, W2, nrmse, gap in results_c:

            idx = W_RATIOS.index((W1, W2))

            writer.writerow([c_rate, label, W1, W2, round(nrmse, 2), round(gap, 2),

                             PAPER_NRMSE[idx], PAPER_GAP[idx],

                             round(nrmse - PAPER_NRMSE[idx], 2), round(gap - PAPER_GAP[idx], 2)])

        writer.writerow([c_rate, "AR", "", "", ar_nrmse_for_all_c[C_RATES.index(c_rate)],

                         ar_gap_for_all_c[C_RATES.index(c_rate)],

                         PAPER_AR_NRMSE, PAPER_AR_GAP,

                         round(ar_nrmse_for_all_c[C_RATES.index(c_rate)] - PAPER_AR_NRMSE, 2),

                         round(ar_gap_for_all_c[C_RATES.index(c_rate)] - PAPER_AR_GAP, 2)])

        writer.writerow([])



print(f"\n저장: {csv_path}")





# =====================================================================

# 10. Best (c, W1/W2) 조합

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



if best_combo:

    c, lb, nr, gp, idx = best_combo

    print(f"\nBest match: c={c}, W1/W2={lb} "

          f"(nRMSE={nr:.2f}%, Gap={gp:.2f}%, "

          f"ΔnRMSE={nr-PAPER_NRMSE[idx]:+.2f}%p, ΔGap={gp-PAPER_GAP[idx]:+.2f}%p)")



