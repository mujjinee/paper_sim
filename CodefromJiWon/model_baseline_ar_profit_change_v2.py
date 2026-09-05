# -*- coding: utf-8 -*-

# =====================================================================
# model_baseline_ar_profit_change_v2.py
#
# 목적: model_baseline_ar_profit_change.py("기본 모형 AR", 4항 이익함수:
#       부족분 = 벌금(rate*DA) + RT가격)를 수정해서, 벌금비용률 50%
#       한 점만 계산하던 것을 Fig.3 / Fig.5 에 필요한 값으로 확장한다.
#
#   - AR은 제안 모형(MILP)과 달리 W1/W2 라는 손잡이가 없다 - 회귀 자체가
#     벌금율/W1/W2와 무관하게 딱 한 번만 풀린다. 그래서:
#       · Fig.3용: 벌금비용률을 논문과 동일하게 50%로 고정한 "AR" 한 점
#         (nRMSE, Gap)만 있으면 된다 - 제안 모형 스윕의 맨 왼쪽 기준점.
#       · Fig.5용: W1/W2는 AR과 무관하므로, 벌금비용률만 0%~100%
#         (10%씩, 11개 지점)로 스윕한다. nRMSE는 회귀가 고정이라
#         모든 지점에서 동일하고, Gap만 벌금율에 따라 달라진다.
#
#   - 예측(회귀) 자체는 model_baseline_ar_profit_change.py 1~7절과 완전히
#     동일 - 딱 한 번만 계산해서 재사용한다.
#
# 산출물: results/model_baseline_ar_profit_change_v2_fig3.csv,
#         results/model_baseline_ar_profit_change_v2_fig5.csv 로 저장하고,
#         동시에 Fig.3용 1행 / Fig.5용 11행 표를 print도 한다.
#         → model_baseline_ar_profit_change_v2_그리기.py 가 이 두 CSV를
#           읽어서 그림을 그린다.
#
# 코딩 스타일: model_baseline_ar_profit_change.py 와 마찬가지로 class는
#             쓰지 않는다. 다만 같은 gap 계산을 벌금율 11번 반복해야
#             하므로, 그 부분만 함수로 묶는다(model_proposed_ar_ver2.py
#             에서도 같은 이유로 def를 썼던 것과 동일한 예외).
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import csv                                   # 결과를 CSV로 저장하기 위한 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
import gurobipy as gp                        # Gurobi 최적화 라이브러리
from gurobipy import GRB                     # Gurobi 상수(GRB.MINIMIZE 등)


# =====================================================================
# 0. 설정값 (여기 숫자만 바꾸면 동작이 바뀜)
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

# --- Fig.3 설정: 벌금비용률 고정(논문과 동일 50%), AR은 이 한 점만 필요 ---
FIG3_PENALTY_RATE = 0.5
FIG3_PAPER_NRMSE = 34.76           # 논문 Table 3, 기본 모형 AR 의 nRMSE(%) - 비교용
FIG3_PAPER_GAP = 15.04             # 논문 Table 3, 기본 모형 AR 의 optimality gap(%) - 비교용

# --- Fig.5 설정: 벌금비용률 0~100% 스윕 (W1/W2는 AR과 무관해서 없음) ---
FIG5_PENALTY_RATES = [round(0.1 * i, 1) for i in range(11)]   # 0.0, 0.1, ..., 1.0


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
#    (model_baseline_ar_profit_change.py 1절과 동일)
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
row_counter = 0
for _, one_row in train_rows.iterrows():
    day_position = row_counter // HOURS_PER_DAY
    hour_position = row_counter % HOURS_PER_DAY
    train_solar[day_position, hour_position] = one_row["solar_power"]
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
# 4. AR 학습용 입력(X), 정답(y) 만들기 - 논문 Eq.(3): "직전 하루"의
#    12시간을 입력으로 씀 (4절과 동일)
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


# =====================================================================
# 5. 시간대(h=0~11) 마다 따로 "bounded LAD" 회귀를 풀어서 계수를 구함
#    (5절과 동일 - 벌금율/W1/W2와 무관하게 한 번만 풀면 됨)
# =====================================================================
coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))

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

    coefficients_by_hour[hour] = beta_var.X
    print(f"  시간대 {hour} 회귀 완료 (성공 여부: {gmodel.Status == GRB.OPTIMAL})")


# =====================================================================
# 6. 테스트 100일을 하루씩 순서대로 예측 (rolling one-day-ahead) - 6절과 동일
# =====================================================================
test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))
previous_day_actual = train_solar[-1]

for day_index in range(n_test_days):
    feature_vector = np.concatenate([[1.0], previous_day_actual[::-1]])
    for hour in range(HOURS_PER_DAY):
        raw_prediction = np.dot(coefficients_by_hour[hour], feature_vector)
        test_forecast[day_index, hour] = min(max(raw_prediction, 0.0), 1.0)
    previous_day_actual = test_solar[day_index]

actual_flat = test_solar.flatten()
predicted_flat = test_forecast.flatten()
da_flat = test_da_price.flatten()
rt_flat = test_rt_price.flatten()


# =====================================================================
# 7. nRMSE 계산 (Eq. 11-12) - 7절과 동일. AR 예측은 고정이라 벌금율과
#    무관하게 딱 한 번만 계산하면 되고, Fig.3/Fig.5 어디서든 재사용한다.
# =====================================================================
rmse_value = np.sqrt(np.mean((actual_flat - predicted_flat) ** 2))
nrmse_percent = 100.0 * rmse_value / np.mean(actual_flat)


# =====================================================================
# 8. optimality gap을 벌금비용률(rate)에 대한 함수로 만들기
#    - 논문 Eq.1a 3항이 아니라 model_baseline_ar_profit_change.py 8절과
#      동일한 "4항"(부족분 = 벌금(rate*DA) + RT가격) 그대로 사용
#    - AR 예측(predicted_flat)은 벌금율과 무관하게 고정 - rate가 바뀌어도
#      회귀를 다시 풀지 않고, gap 계산에만 rate를 대입한다
# =====================================================================
def compute_gap(rate):
    sum_of_realized_profit = 0.0
    sum_of_oracle_profit = 0.0

    for i in range(len(actual_flat)):
        actual_i = actual_flat[i]
        commitment_i = predicted_flat[i]
        da_i = da_flat[i]
        rt_i = rt_flat[i]
        penalty_cost_i = rate * da_i

        # ---- 실제(AR 예측) 커밋에 대한 이익 계산 (4항) ----
        mismatch_i = actual_i - commitment_i
        surplus_i = max(mismatch_i, 0.0)
        shortage_i = max(-mismatch_i, 0.0)
        realized_profit_i = CAPACITY_MW * DURATION_HOURS * (
            da_i * commitment_i + rt_i * surplus_i - (penalty_cost_i + rt_i) * shortage_i
        )
        sum_of_realized_profit += realized_profit_i

        # ---- 오라클(사후 최적) 이익 계산: {0, 실제발전량, 설비최대(1.0)} 3후보 ----
        profit_if_commit_zero = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
        if actual_i <= 1.0:
            profit_if_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
        else:
            profit_if_commit_actual = -np.inf
        surplus_if_full = max(actual_i - 1.0, 0.0)
        shortage_if_full = max(1.0 - actual_i, 0.0)
        profit_if_commit_full = CAPACITY_MW * DURATION_HOURS * (
            da_i * 1.0 + rt_i * surplus_if_full - (penalty_cost_i + rt_i) * shortage_if_full
        )
        oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual, profit_if_commit_full)
        sum_of_oracle_profit += oracle_profit_i

    return 100.0 * (sum_of_oracle_profit - sum_of_realized_profit) / sum_of_oracle_profit


# =====================================================================
# 9. Fig.3 값 출력 - 벌금율 50% 고정, AR은 한 점(nRMSE, Gap)뿐
# =====================================================================
print("\n" + "=" * 60)
print(f"  Fig.3용 값 (벌금율={FIG3_PENALTY_RATE:.0%} 고정, AR은 W1/W2 무관 - 한 점)")
print("=" * 60)

fig3_gap = compute_gap(FIG3_PENALTY_RATE)
print(f"  AR: nRMSE={nrmse_percent:.2f}%, Gap={fig3_gap:.2f}%  "
      f"(논문: nRMSE={FIG3_PAPER_NRMSE:.2f}%, Gap={FIG3_PAPER_GAP:.2f}%)")

print()
print("| Label | nRMSE | Gap | Paper_nRMSE | Paper_Gap |")
print("|---|---:|---:|---:|---:|")
print(f"| AR | {nrmse_percent:.4f}% | {fig3_gap:.4f}% | {FIG3_PAPER_NRMSE:.2f}% | {FIG3_PAPER_GAP:.2f}% |")

out_dir = os.path.join(BASE_DIR, "results")
os.makedirs(out_dir, exist_ok=True)

fig3_csv_path = os.path.join(out_dir, "model_baseline_ar_profit_change_v2_fig3.csv")
with open(fig3_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"])
    writer.writerow(["AR", round(nrmse_percent, 4), round(fig3_gap, 4), FIG3_PAPER_NRMSE, FIG3_PAPER_GAP])
print(f"\nsaved: {fig3_csv_path}")


# =====================================================================
# 10. Fig.5 값 출력 - 벌금율 0~100% 스윕 (W1/W2는 AR과 무관해서 없음)
# =====================================================================
print("\n" + "=" * 60)
print("  Fig.5용 값 (벌금율 0~100% 스윕, AR nRMSE는 회귀가 고정이라 불변)")
print("=" * 60)

fig5_gap_list = []
for rate in FIG5_PENALTY_RATES:
    gap_r = compute_gap(rate)
    fig5_gap_list.append(gap_r)
    print(f"  rate={rate:.1f}: nRMSE={nrmse_percent:.2f}%, Gap={gap_r:.2f}%")

print()
print("| Rate | AR_nRMSE | AR_Gap |")
print("|---|---:|---:|")
for rate, gap_r in zip(FIG5_PENALTY_RATES, fig5_gap_list):
    print(f"| {rate:.1f} | {nrmse_percent:.4f}% | {gap_r:.4f}% |")

fig5_csv_path = os.path.join(out_dir, "model_baseline_ar_profit_change_v2_fig5.csv")
with open(fig5_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Rate", "AR_nRMSE", "AR_Gap"])
    for rate, gap_r in zip(FIG5_PENALTY_RATES, fig5_gap_list):
        writer.writerow([rate, round(nrmse_percent, 4), round(gap_r, 4)])
print(f"saved: {fig5_csv_path}")

print("\n완료.")
