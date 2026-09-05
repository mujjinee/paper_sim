# -*- coding: utf-8 -*-

# =====================================================================
# model_proposed_ar_ver2.py
#
# 목적: model_proposed_ar.py("논문 제안 모형 AR", 논문 Eq.9 MILP)를
#       수정해서, 한 점(W1=1,W2=20, 벌금율=50%)만 계산하던 것을
#       Fig.3 / Fig.5 를 그리는 데 필요한 여러 점으로 확장한다.
#
#   - Fig.3용: 벌금비용률(PENALTY_RATE)을 50%로 고정하고, W1/W2 비율을
#              논문 Fig.3과 동일한 11개 지점(AR, 1/20, 1/10, 1/5, 1/2,
#              1/1, 2/1, 5/1, 10/1, 20/1, 1/0)으로 스윕한다.
#   - Fig.5용: W1/W2=1/1로 고정하고(논문 Fig.5와 동일 조건), 벌금비용률을
#              0%~100%(10%씩, 11개 지점)로 스윕한다.
#   - "AR" 지점(순수 기본 모형)은 model_baseline_ar.py 의 bounded LAD
#     회귀를 그대로 가져와 계산한다 (MILP 아님 - 벌금율과 무관하게 예측은
#     한 번만 하고, gap만 벌금율마다 다시 계산).
#
# 산출물: results/model_proposed_ar_ver2_fig3.csv, results/model_proposed_ar_ver2_fig5.csv
#         → fig3_fig5_그리기.py 가 이 두 CSV를 읽어서 그림을 그린다.
#
# 코딩 스타일: model_proposed_ar.py 와 마찬가지로 class는 쓰지 않는다.
#             다만 같은 MILP를 (penalty_rate, W1, W2) 조합 21번 반복해서
#             풀어야 하므로, 그 부분만 함수로 묶는다(method1_pc_max_da_rt.py
#             에서도 같은 이유로 def를 썼던 것과 동일한 예외).
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import csv                                   # 결과를 CSV로 저장하기 위한 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
import gurobipy as gp                        # Gurobi 최적화 라이브러리
from gurobipy import GRB                     # Gurobi 상수


# =====================================================================
# 0. 설정값
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 이 파이썬 파일이 있는 폴더 경로
MERGED_FILE = os.environ.get("MERGED_FILE", os.path.join(BASE_DIR, "merged_for_simulation_z03.csv"))

HOURS_PER_DAY = 12               # 하루 낮 시간대 개수 (Sydney 현지시간 9시~20시)
LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_START = pd.Timestamp(os.environ.get("TRAIN_START", "2013-08-25"))   # 학습 시작일
TRAIN_END = pd.Timestamp(os.environ.get("TRAIN_END",   "2013-11-22"))     # 학습 마지막일
TEST_START = pd.Timestamp(os.environ.get("TEST_START",  "2013-11-23"))    # 테스트 시작일
TEST_END = pd.Timestamp(os.environ.get("TEST_END",    "2013-12-22"))      # 테스트 마지막일
HISTORY_DATE = TRAIN_START - pd.Timedelta(days=1)   # 학습 첫날 직전 하루를 자동으로 계산

CAPACITY_MW = 30.0                # 태양광 패널 설비 최대 용량 (논문 가정)
DURATION_HOURS = 1.0              # 한 시간대의 길이(시간)

# --- Fig.3 설정: 벌금비용률 고정, W1/W2 11개점 스윕 ---
FIG3_PENALTY_RATE = 0.5
FIG3_LABELS = ["AR", "1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]
FIG3_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG3_PAPER_NRMSE = [34.76, 34.89, 35.14, 36.28, 41.09, 44.95, 46.11, 48.27, 49.21, 49.61, 50.07]
FIG3_PAPER_GAP   = [15.04, 13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36]

# --- Fig.5 설정: W1/W2 고정, 벌금비용률 0~100% 스윕 ---
FIG5_W1, FIG5_W2 = 1.0, 1.0
FIG5_PENALTY_RATES = [round(0.1 * i, 1) for i in range(11)]   # 0.0, 0.1, ..., 1.0


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기 (model_proposed_ar.py 1절과 동일)
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START


# =====================================================================
# 2. 이력(history) / 학습(train) / 테스트(test) 구간으로 자르기 (2절과 동일)
# =====================================================================
is_history_date = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history_date].copy().sort_values("hour_idx")

is_train_date = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train_date].copy().sort_values(["local_date", "hour_idx"])

is_test_date = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test_date].copy().sort_values(["local_date", "hour_idx"])

print("이력 날짜:", HISTORY_DATE.date(), "행 수:", len(history_rows))
print("학습 구간:", TRAIN_START.date(), "~", TRAIN_END.date(), "행 수:", len(train_rows))
print("테스트 구간:", TEST_START.date(), "~", TEST_END.date(), "행 수:", len(test_rows))


# =====================================================================
# 3. (날짜 x 12시간) 모양의 숫자 배열로 바꾸기 (3절과 동일)
# =====================================================================
history_solar = np.zeros((1, HOURS_PER_DAY))
row_counter = 0
for _, one_row in history_rows.iterrows():
    hour_position = row_counter % HOURS_PER_DAY
    history_solar[0, hour_position] = one_row["solar_power"]
    row_counter = row_counter + 1

train_dates_sorted = sorted(train_rows["local_date"].unique())
n_train_days = len(train_dates_sorted)
train_solar = np.zeros((n_train_days, HOURS_PER_DAY))
train_da_price = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt_price = np.zeros((n_train_days, HOURS_PER_DAY))
row_counter = 0
for _, one_row in train_rows.iterrows():
    day_position = row_counter // HOURS_PER_DAY
    hour_position = row_counter % HOURS_PER_DAY
    train_solar[day_position, hour_position] = one_row["solar_power"]
    train_da_price[day_position, hour_position] = one_row["da_price"]
    train_rt_price[day_position, hour_position] = one_row["rt_price"]
    row_counter = row_counter + 1

test_dates_sorted = sorted(test_rows["local_date"].unique())
n_test_days = len(test_dates_sorted)
test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))
row_counter = 0
for _, one_row in test_rows.iterrows():
    day_position = row_counter // HOURS_PER_DAY
    hour_position = row_counter % HOURS_PER_DAY
    test_solar[day_position, hour_position] = one_row["solar_power"]
    test_da_price[day_position, hour_position] = one_row["da_price"]
    test_rt_price[day_position, hour_position] = one_row["rt_price"]
    row_counter = row_counter + 1


# =====================================================================
# 4. AR 학습용 입력(X) 만들기 - "직전 하루"의 12시간을 입력으로 씀 (4절과 동일)
# =====================================================================
history_and_train_solar = np.vstack([history_solar, train_solar])

n_ar_rows = n_train_days
ar_intercept_column = np.ones((n_ar_rows, 1))
ar_lag_features = np.zeros((n_ar_rows, HOURS_PER_DAY))
for day_index in range(n_ar_rows):
    previous_day_values = history_and_train_solar[day_index]
    ar_lag_features[day_index] = previous_day_values[::-1]

ar_design_matrix = np.hstack([ar_intercept_column, ar_lag_features])
n_features = ar_design_matrix.shape[1]

actual_flat = test_solar.flatten()
da_flat = test_da_price.flatten()
rt_flat = test_rt_price.flatten()


# =====================================================================
# 5. nRMSE / optimality gap 계산 함수
#    - PC = penalty_rate * DA (논문 Eq.1a 3항 그대로, model_proposed_ar.py와 동일)
#    - 오라클: {0, 실제발전량, 설비최대(1.0)} 3후보 (수정 후 보정 오라클)
# =====================================================================
def compute_nrmse(pred_flat):
    rmse_value = np.sqrt(np.mean((actual_flat - pred_flat) ** 2))
    return 100.0 * rmse_value / np.mean(actual_flat)


def compute_gap(pred_flat, penalty_rate):
    sum_realized = 0.0
    sum_oracle = 0.0
    for i in range(len(actual_flat)):
        actual_i = actual_flat[i]; commitment_i = pred_flat[i]
        da_i = da_flat[i]; rt_i = rt_flat[i]
        penalty_cost_i = penalty_rate * da_i

        mismatch_i = actual_i - commitment_i
        surplus_i = max(mismatch_i, 0.0)
        shortage_i = max(-mismatch_i, 0.0)
        realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
            da_i * commitment_i + rt_i * surplus_i - penalty_cost_i * shortage_i
        )
        sum_realized += realized_profit_i

        profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        if actual_i <= 1.0:
            profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        else:
            profit_if_commit_actual = -np.inf
        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)
        profit_if_commit_full = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - penalty_cost_i * shortage_if_full
        )
        oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual, profit_if_commit_full)
        sum_oracle += oracle_profit_i

    return 100.0 * (sum_oracle - sum_realized) / sum_oracle


def predict_test(coefficients_by_hour):
    # 테스트 100일을 하루씩 순서대로 예측 (rolling one-day-ahead) - 6절과 동일
    test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))
    previous_day_actual = train_solar[-1]
    for day_index in range(n_test_days):
        feature_vector = np.concatenate([[1.0], previous_day_actual[::-1]])
        for hour in range(HOURS_PER_DAY):
            raw_prediction = np.dot(coefficients_by_hour[hour], feature_vector)
            test_forecast[day_index, hour] = min(max(raw_prediction, 0.0), 1.0)
        previous_day_actual = test_solar[day_index]
    return test_forecast.flatten()


# =====================================================================
# 6. AR 기본 모형 (model_baseline_ar.py 5절의 bounded LAD 회귀 그대로) -
#    벌금율/W1/W2와 무관하게 예측은 딱 한 번만 하면 됨
# =====================================================================
print("\n" + "=" * 60)
print("  AR 기본 모형 (bounded LAD) 계산")
print("=" * 60)

ar_coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))
for hour in range(HOURS_PER_DAY):
    y_this_hour = train_solar[:, hour]

    gmodel = gp.Model(f"baseline_ar_hour_{hour}")
    gmodel.Params.OutputFlag = 0

    beta_var = gmodel.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
    u_var = gmodel.addMVar(n_ar_rows, lb=0.0, name="u")

    fitted_expr = ar_design_matrix @ beta_var
    gmodel.addConstr(fitted_expr - y_this_hour <= u_var, name="resid_upper")
    gmodel.addConstr(y_this_hour - fitted_expr <= u_var, name="resid_lower")
    gmodel.setObjective(u_var.sum() / n_ar_rows, GRB.MINIMIZE)
    gmodel.optimize()

    ar_coefficients_by_hour[hour] = beta_var.X
    print(f"  시간대 {hour} AR 회귀 완료 (성공 여부: {gmodel.Status == GRB.OPTIMAL})")

ar_pred_flat = predict_test(ar_coefficients_by_hour)
ar_nrmse = compute_nrmse(ar_pred_flat)   # 벌금율과 무관 - 한 번만 계산
print(f"  AR nRMSE = {ar_nrmse:.4f}%  (벌금율에 따라 Gap만 달라짐)")


# =====================================================================
# 7. (penalty_rate, W1, W2) 하나를 받아 제안모형 MILP를 풀고 계수를
#    반환하는 함수 - model_proposed_ar.py 5절 로직 그대로, 상수만 인자로 뺌
# =====================================================================
def solve_proposed_milp(penalty_rate, W1, W2):
    coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))

    for hour in range(HOURS_PER_DAY):
        y_this_hour = train_solar[:, hour]
        da_this_hour = train_da_price[:, hour]
        rt_this_hour = train_rt_price[:, hour]

        # ---- 학습용(training) 오라클 이익: {0, 실제발전량, 설비최대(1.0)} 3후보 ----
        oracle_profit_train = np.zeros(n_ar_rows)
        for i in range(n_ar_rows):
            actual_i = y_this_hour[i]; da_i = da_this_hour[i]; rt_i = rt_this_hour[i]
            penalty_i = penalty_rate * da_i

            profit_commit_0 = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
            profit_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
            surplus_if_full = max(actual_i - 1.0, 0.0)
            shortage_if_full = max(1.0 - actual_i, 0.0)
            profit_commit_1 = CAPACITY_MW * DURATION_HOURS * (
                da_i * 1.0 + rt_i * surplus_if_full - penalty_i * shortage_if_full
            )
            oracle_profit_train[i] = max(profit_commit_0, profit_commit_actual, profit_commit_1)

        # ---- 목적함수 계수 계산 (잉여/부족 각각에 대한 비용) - 논문 Eq.(9a) 그대로 ----
        surplus_cost = np.zeros(n_ar_rows)
        shortage_cost = np.zeros(n_ar_rows)
        for i in range(n_ar_rows):
            penalty_i = penalty_rate * da_this_hour[i]
            surplus_cost[i] = (-W1 * rt_this_hour[i]) + W2
            shortage_cost[i] = (W1 * penalty_i) + W2

        binary_row_list = []
        for i in range(n_ar_rows):
            if surplus_cost[i] + shortage_cost[i] < 0.0:
                binary_row_list.append(i)
        binary_rows = np.array(binary_row_list, dtype=int)
        n_binary = len(binary_rows)

        # ---- Gurobi로 MILP 구성 및 풀기 ----
        gmodel = gp.Model(f"proposed_ar_hour_{hour}")
        gmodel.Params.OutputFlag = 0
        gmodel.Params.MIPGap = 1e-9

        beta_var = gmodel.addMVar(n_features, lb=-GRB.INFINITY, name="beta")
        x_var = gmodel.addMVar(n_ar_rows, lb=0.0, ub=1.0, name="x")
        yplus_var = gmodel.addMVar(n_ar_rows, lb=0.0, ub=1.0, name="y_plus")
        yminus_var = gmodel.addMVar(n_ar_rows, lb=0.0, ub=1.0, name="y_minus")

        gmodel.addConstr(x_var - ar_design_matrix @ beta_var == 0.0, name="commitment_eq")
        gmodel.addConstr(x_var + yplus_var - yminus_var == y_this_hour, name="mismatch_eq")

        if n_binary > 0:
            z_var = gmodel.addMVar(n_binary, vtype=GRB.BINARY, name="z")
            gmodel.addConstr(yplus_var[binary_rows] + z_var <= 1.0, name="complementarity_plus")
            gmodel.addConstr(yminus_var[binary_rows] - z_var <= 0.0, name="complementarity_minus")

        objective_expr = (
            (-W1 * da_this_hour) @ x_var
            + surplus_cost @ yplus_var
            + shortage_cost @ yminus_var
        )
        gmodel.setObjective(objective_expr, GRB.MINIMIZE)
        gmodel.optimize()

        coefficients_by_hour[hour] = beta_var.X

    return coefficients_by_hour


# =====================================================================
# 8. Fig.3 데이터 생성 — 벌금율 50% 고정, W1/W2 11개점(AR 포함) 스윕
# =====================================================================
print("\n" + "=" * 60)
print(f"  Fig.3 데이터 생성 (벌금율={FIG3_PENALTY_RATE} 고정)")
print("=" * 60)

fig3_ar_gap = compute_gap(ar_pred_flat, FIG3_PENALTY_RATE)
fig3_nrmse_list = [ar_nrmse]
fig3_gap_list = [fig3_ar_gap]
print(f"  AR: nRMSE={ar_nrmse:.2f}%, Gap={fig3_ar_gap:.2f}%  "
      f"(논문: nRMSE={FIG3_PAPER_NRMSE[0]:.2f}%, Gap={FIG3_PAPER_GAP[0]:.2f}%)")

for idx, (W1, W2) in enumerate(FIG3_W_RATIOS):
    coeffs = solve_proposed_milp(FIG3_PENALTY_RATE, W1, W2)
    pred_flat = predict_test(coeffs)
    nrmse = compute_nrmse(pred_flat)
    gap = compute_gap(pred_flat, FIG3_PENALTY_RATE)
    fig3_nrmse_list.append(nrmse)
    fig3_gap_list.append(gap)
    print(f"  {W1}/{W2}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}%  "
          f"(논문: nRMSE={FIG3_PAPER_NRMSE[idx + 1]:.2f}%, Gap={FIG3_PAPER_GAP[idx + 1]:.2f}%)")


# =====================================================================
# 9. Fig.5 데이터 생성 — W1/W2=1/1 고정, 벌금율 0~100% 스윕
# =====================================================================
print("\n" + "=" * 60)
print(f"  Fig.5 데이터 생성 (W1/W2={FIG5_W1:.0f}/{FIG5_W2:.0f} 고정)")
print("=" * 60)

fig5_ar_nrmse_list = []
fig5_ar_gap_list = []
fig5_prop_nrmse_list = []
fig5_prop_gap_list = []

for rate in FIG5_PENALTY_RATES:
    ar_gap_r = compute_gap(ar_pred_flat, rate)
    fig5_ar_nrmse_list.append(ar_nrmse)
    fig5_ar_gap_list.append(ar_gap_r)

    coeffs = solve_proposed_milp(rate, FIG5_W1, FIG5_W2)
    pred_flat = predict_test(coeffs)
    prop_nrmse_r = compute_nrmse(pred_flat)
    prop_gap_r = compute_gap(pred_flat, rate)
    fig5_prop_nrmse_list.append(prop_nrmse_r)
    fig5_prop_gap_list.append(prop_gap_r)

    print(f"  rate={rate:.1f}: AR Gap={ar_gap_r:.2f}%, "
          f"제안 nRMSE={prop_nrmse_r:.2f}%, Gap={prop_gap_r:.2f}%")


# =====================================================================
# 10. 결과 CSV 저장 — fig3_fig5_그리기.py 가 이 두 파일을 읽는다
# =====================================================================
out_dir = os.path.join(BASE_DIR, "results")
os.makedirs(out_dir, exist_ok=True)

fig3_csv_path = os.path.join(out_dir, "model_proposed_ar_ver2_fig3.csv")
with open(fig3_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"])
    for label, nrmse, gap, p_nrmse, p_gap in zip(
            FIG3_LABELS, fig3_nrmse_list, fig3_gap_list, FIG3_PAPER_NRMSE, FIG3_PAPER_GAP):
        writer.writerow([label, round(nrmse, 4), round(gap, 4), p_nrmse, p_gap])
print(f"\nsaved: {fig3_csv_path}")

fig5_csv_path = os.path.join(out_dir, "model_proposed_ar_ver2_fig5.csv")
with open(fig5_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Rate", "AR_nRMSE", "AR_Gap", "Proposed_nRMSE", "Proposed_Gap"])
    for rate, an, ag, pn, pg in zip(
            FIG5_PENALTY_RATES, fig5_ar_nrmse_list, fig5_ar_gap_list, fig5_prop_nrmse_list, fig5_prop_gap_list):
        writer.writerow([rate, round(an, 4), round(ag, 4), round(pn, 4), round(pg, 4)])
print(f"saved: {fig5_csv_path}")

print("\n완료.")
