
# -*- coding: utf-8 -*-



# =====================================================================

# method2_pc_da_plus_rt.py

# 방법 2: PC = rate*(DA + RT) 벌금률 정의 변경

#

# 기존: PC = 0.5*DA -> PC < DA 항상 -> arbitrage

# 변경: PC = rate*(DA + RT) -> RT 충분히 크면 PC > DA -> arbitrage 해소

#

# rate를 스윕하여 최적 rate 탐색.

# =====================================================================



import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np

import pandas as pd

from scipy import sparse

from scipy.optimize import Bounds, LinearConstraint, milp, linprog

import csv





# =====================================================================

# 0. 설정

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



W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 1), (1, 0)]

PENALTY_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]



PAPER_AR_NRMSE = 34.76

PAPER_AR_GAP = 15.04

PAPER_NRMSE = [34.89, 35.14, 36.28, 44.95, 50.07]

PAPER_GAP = [13.91, 13.42, 12.71, 11.44, 11.36]



scale = CAPACITY_MW * DURATION_HOURS





# =====================================================================

# 1. 데이터

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





# =====================================================================

# 2. (날짜 x 12) 배열

# 3. AR 학습용 입력

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

# 4. Gap 계산 함수

#    realized = DA*x + RT*surplus - PC*shortage, PC = rate*(DA+RT)

#    oracle: 3후보 {0, actual, 1.0}

# =====================================================================

def compute_gap(pred_flat, penalty_rate):

    sum_realized = 0.0

    sum_oracle = 0.0

    for i in range(n_test_obs):

        a = actual_flat[i]; x = pred_flat[i]

        dp = da_flat[i]; rp = rt_flat[i]

        pc = penalty_rate * (dp + rp)



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

# 5. 기본 AR

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





# =====================================================================

# 6. penalty_rate별 W1/W2 sweep

# =====================================================================

all_results = []



for penalty_rate in PENALTY_RATES:

    print(f"\n{'='*60}")

    print(f"  rate = {penalty_rate}  (PC = rate*(DA+RT))")

    print(f"{'='*60}")



    results_c = []



    for widx, (W1, W2) in enumerate(W_RATIOS):



        coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))



        for hour in range(HOURS_PER_DAY):

            y_h = train_solar[:, hour]

            da_h = train_da_price[:, hour]

            rt_h = train_rt_price[:, hour]

            n_obs = n_ar_rows



            pc_h = penalty_rate * (da_h + rt_h)



            # Training oracle: 3후보

            oracle_profit_train = np.zeros(n_obs)

            for i in range(n_obs):

                a = y_h[i]; dp = da_h[i]; rp = rt_h[i]

                pc = penalty_rate * (dp + rp)

                p0 = scale * rp * a

                pa = scale * dp * a

                s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)

                p1 = scale * (dp * 1.0 + rp * s1 - pc * y1)

                oracle_profit_train[i] = max(p0, pa, p1)

            training_denom = oracle_profit_train.sum()



            surplus_cost = np.zeros(n_obs)

            shortage_cost = np.zeros(n_obs)

            for i in range(n_obs):

                pc = penalty_rate * (da_h[i] + rt_h[i])

                surplus_cost[i] = (-W1 * scale * rt_h[i] / training_denom) + (W2 / n_obs)

                shortage_cost[i] = (W1 * scale * pc / training_denom) + (W2 / n_obs)

        gap = compute_gap(pred_flat, penalty_rate)



        label = f"{W1}/{W2}"

        results_c.append((label, W1, W2, nrmse, gap))



        print(f"  {label}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}% "

              f"(paper: nRMSE={PAPER_NRMSE[widx]:.2f}%, Gap={PAPER_GAP[widx]:.2f}%)")



    all_results.append((penalty_rate, results_c))





# =====================================================================

# 7. 결과 표

# =====================================================================

print()

print("=" * 110)

print(f"{'rate':>5} {'Label':>6} {'W1':>4} {'W2':>4} {'nRMSE':>10} {'Gap':>10} {'Paper nRMSE':>12} {'Paper Gap':>12} {'d nRMSE':>10} {'d Gap':>10}")

print("-" * 110)



for penalty_rate, results_c in all_results:

    for label, W1, W2, nrmse, gap in results_c:

        idx = W_RATIOS.index((W1, W2))

        print(f"{penalty_rate:>5} {label:>6} {W1:>4} {W2:>4} {nrmse:>9.2f}% {gap:>9.2f}% "

              f"{PAPER_NRMSE[idx]:>11.2f}% {PAPER_GAP[idx]:>11.2f}% "

              f"{nrmse-PAPER_NRMSE[idx]:>+9.2f}%p {gap-PAPER_GAP[idx]:>+9.2f}%p")

print("=" * 110)



# AR baseline

print(f"\nAR baseline:")

for penalty_rate in PENALTY_RATES:

    arg = compute_gap(ar_pred_flat, penalty_rate)

    print(f"  rate={penalty_rate}: nRMSE={ar_nrmse:.2f}%, Gap={arg:.2f}% "

          f"(paper: nRMSE={PAPER_AR_NRMSE:.2f}%, Gap={PAPER_AR_GAP:.2f}%)")





# =====================================================================

# 8. CSV

# =====================================================================

csv_path = os.path.join(os.path.dirname(MERGED_FILE), "results", "simulation_output",

                        "fig3_method2_pc_da_plus_rt.csv")

os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["rate", "Label", "W1", "W2", "nRMSE", "Gap",

                      "Paper_nRMSE", "Paper_Gap", "Delta_nRMSE", "Delta_Gap"])

    for penalty_rate, results_c in all_results:

        for label, W1, W2, nrmse, gap in results_c:

            idx = W_RATIOS.index((W1, W2))

            writer.writerow([penalty_rate, label, W1, W2, round(nrmse, 2), round(gap, 2),

                             PAPER_NRMSE[idx], PAPER_GAP[idx],

                             round(nrmse - PAPER_NRMSE[idx], 2), round(gap - PAPER_GAP[idx], 2)])



print(f"\nsaved: {csv_path}")





# =====================================================================

# 9. Best

# =====================================================================

best_total_err = np.inf

best_combo = None

for penalty_rate, results_c in all_results:

    for label, W1, W2, nrmse, gap in results_c:

        idx = W_RATIOS.index((W1, W2))

        err = abs(nrmse - PAPER_NRMSE[idx]) + abs(gap - PAPER_GAP[idx])

        if err < best_total_err:

            best_total_err = err

            best_combo = (penalty_rate, label, nrmse, gap, idx)



if best_combo:

    r, lb, nr, gp, idx = best_combo

    print(f"\nBest: rate={r}, W1/W2={lb} "

          f"(nRMSE={nr:.2f}%, Gap={gp:.2f}%, "

          f"dnRMSE={nr-PAPER_NRMSE[idx]:+.2f}%p, dGap={gp-PAPER_GAP[idx]:+.2f}%p)")

