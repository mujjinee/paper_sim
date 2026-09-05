# -*- coding: utf-8 -*-

# =====================================================================
# model_proposed_ar_profit_change_v3.py
#
# 목적: model_proposed_ar_profit_change_v2.py(데이터 계산 + CSV 저장)와
#       model_proposed_ar_profit_change_v2_그리기.py(그림 그리기)를 한
#       파일로 합치고, Fig.3뿐 아니라 Fig.5(벌금율 스윕)까지 한 번에
#       계산+그림까지 끝내도록 확장한다.
#
#   - Fig.3: 벌금율 50% 고정, W1/W2 10개점(AR 포함 11개) 스윕 - 논문
#            Table 3/Fig.3(PDF 8쪽)과 동일한 dual y축 그림.
#   - Fig.5: W1=W2=1 고정, 벌금율 0%~100%(11개점) 스윕 - 논문 Fig.5(a)/(b)
#            와 동일하게 nRMSE/Gap을 좌우 두 그래프로 분리.
#   - 4항 이익함수(부족분 = 벌금(rate*DA) + RT가격)를 Fig.3/Fig.5 양쪽에
#     그대로 사용(model_proposed_ar_profit_change.py 5절과 동일 공식).
#
# 산출물: results/model_proposed_ar_profit_change_v3_fig3.csv,
#         results/model_proposed_ar_profit_change_v3_fig5.csv,
#         results/model_proposed_ar_profit_change_v3_fig3.png,
#         results/model_proposed_ar_profit_change_v3_fig5.png
#
# 코딩 스타일: model_proposed_ar_ver2.py 와 마찬가지로 class는 쓰지
#             않는다. 같은 MILP를 (penalty_rate, W1, W2) 조합만큼
#             반복해야 하므로, 그 부분만 함수로 묶는다.
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import csv                                   # 결과를 CSV로 저장하기 위한 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
import gurobipy as gp                        # Gurobi 최적화 라이브러리
from gurobipy import GRB                     # Gurobi 상수
import matplotlib                            # 그래프를 그리는 라이브러리
matplotlib.use("Agg")                         # 화면 없이 파일로만 저장하는 백엔드 지정
import matplotlib.pyplot as plt               # 실제 그리기 함수들이 들어있는 서브모듈
import matplotlib.font_manager as fm          # 한글 폰트를 찾아 쓰기 위한 서브모듈


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
MERGED_FILE = os.environ.get("MERGED_FILE", os.path.join(BASE_DIR, "merged_for_simulation_z03.csv"))

HOURS_PER_DAY = 12
LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21

TRAIN_START = pd.Timestamp(os.environ.get("TRAIN_START", "2013-08-25"))
TRAIN_END = pd.Timestamp(os.environ.get("TRAIN_END",   "2013-11-22"))
TEST_START = pd.Timestamp(os.environ.get("TEST_START",  "2013-11-23"))
TEST_END = pd.Timestamp(os.environ.get("TEST_END",    "2013-12-22"))
HISTORY_DATE = TRAIN_START - pd.Timedelta(days=1)

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0

# --- Fig.3 설정: 벌금율 50% 고정, W1/W2 10개점 스윕 (AR 포함 11개 지점) ---
FIG3_PENALTY_RATE = 0.5
FIG3_LABELS = ["AR", "1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]
FIG3_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG3_PAPER_AR_NRMSE = 34.76
FIG3_PAPER_AR_GAP = 15.04
FIG3_PAPER_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95, 46.11, 48.27, 49.21, 49.61, 50.07]
FIG3_PAPER_GAP   = [13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36]

# --- Fig.5 설정: W1=W2=1 고정, 벌금율 0~100% 스윕 ---
FIG5_W1, FIG5_W2 = 1.0, 1.0
FIG5_PENALTY_RATES = [round(0.1 * i, 1) for i in range(11)]   # 0.0, 0.1, ..., 1.0

# 논문 Fig.5(a)/(b)를 육안으로 읽은 참고값(50%만 Table 3 실측치, 나머지는
# 근사치) - fig3_fig5_그리기.py / temp_method1_rate09_fig3_fig5.py 와 동일
FIG5_PAPER_AR_NRMSE   = [34.76] * 11
FIG5_PAPER_AR_GAP     = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]
FIG5_PAPER_PROP_NRMSE = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
FIG5_PAPER_PROP_GAP   = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]


# =====================================================================
# 2. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START


# =====================================================================
# 3. 이력(history) / 학습(train) / 테스트(test) 구간으로 자르기
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
# 4. (날짜 x 12시간) 모양의 숫자 배열로 바꾸기
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
# 5. AR 학습용 입력(X) 만들기 - 논문 Eq.(3): "직전 하루"의 12시간을 입력으로 씀
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
# 6. nRMSE / optimality gap 계산 함수
#    - 4항: profit = DA*x + RT*surplus - (PC+RT)*shortage, PC=penalty_rate*DA
#    - 오라클: {0, 실제발전량, 설비최대(1.0)} 3후보
#    - penalty_rate를 인자로 받아 Fig.3(고정 0.5)과 Fig.5(스윕) 양쪽에서 재사용
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
            da_i * commitment_i + rt_i * surplus_i - (penalty_cost_i + rt_i) * shortage_i
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
            da_i * 1.0 + rt_i * surplus_if_full - (penalty_cost_i + rt_i) * shortage_if_full
        )
        oracle_profit_i = max(profit_if_commit_zero, profit_if_commit_actual, profit_if_commit_full)
        sum_oracle += oracle_profit_i

    return 100.0 * (sum_oracle - sum_realized) / sum_oracle


def predict_test(coefficients_by_hour):
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
# 7. AR 기본 모형 (bounded LAD) - 벌금율/W1/W2와 무관하게 딱 한 번만 풀면 됨
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


# =====================================================================
# 8. (penalty_rate, W1, W2) 하나를 받아 제안모형 MILP를 풀고 계수를 반환
#    하는 함수 - model_proposed_ar_profit_change.py 5절 로직 그대로,
#    상수만 인자로 뺌. 4항: shortage_cost = W1*(penalty+RT) + W2
# =====================================================================
def solve_proposed_milp(penalty_rate, W1, W2):
    coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))

    for hour in range(HOURS_PER_DAY):
        y_this_hour = train_solar[:, hour]
        da_this_hour = train_da_price[:, hour]
        rt_this_hour = train_rt_price[:, hour]

        # ---- 학습용(training) 오라클 이익: {0, 실제발전량, 설비최대(1.0)} 3후보 (4항) ----
        oracle_profit_train = np.zeros(n_ar_rows)
        for i in range(n_ar_rows):
            actual_i = y_this_hour[i]; da_i = da_this_hour[i]; rt_i = rt_this_hour[i]
            penalty_i = penalty_rate * da_i

            profit_commit_0 = CAPACITY_MW * DURATION_HOURS * (rt_i * actual_i)
            profit_commit_actual = CAPACITY_MW * DURATION_HOURS * (da_i * actual_i)
            surplus_if_full = max(actual_i - 1.0, 0.0)
            shortage_if_full = max(1.0 - actual_i, 0.0)
            profit_commit_1 = CAPACITY_MW * DURATION_HOURS * (
                da_i * 1.0 + rt_i * surplus_if_full - (penalty_i + rt_i) * shortage_if_full
            )
            oracle_profit_train[i] = max(profit_commit_0, profit_commit_actual, profit_commit_1)

        # ---- 목적함수 계수 계산 (잉여/부족 각각에 대한 비용) - 4항 ----
        surplus_cost = np.zeros(n_ar_rows)
        shortage_cost = np.zeros(n_ar_rows)
        for i in range(n_ar_rows):
            penalty_i = penalty_rate * da_this_hour[i]
            surplus_cost[i] = (-W1 * rt_this_hour[i]) + W2
            shortage_cost[i] = (W1 * (penalty_i + rt_this_hour[i])) + W2

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
# 9. Fig.3 데이터 생성 — 벌금율 50% 고정, W1/W2 10개점(AR 포함 11개) 스윕
# =====================================================================
print("\n" + "=" * 60)
print(f"  Fig.3 데이터 생성 (벌금율={FIG3_PENALTY_RATE} 고정, 4항 이익함수)")
print("=" * 60)

fig3_ar_gap = compute_gap(ar_pred_flat, FIG3_PENALTY_RATE)
fig3_nrmse_list = [ar_nrmse]
fig3_gap_list = [fig3_ar_gap]
print(f"  AR: nRMSE={ar_nrmse:.2f}%, Gap={fig3_ar_gap:.2f}%  "
      f"(논문: nRMSE={FIG3_PAPER_AR_NRMSE:.2f}%, Gap={FIG3_PAPER_AR_GAP:.2f}%)")

for idx, (W1, W2) in enumerate(FIG3_W_RATIOS):
    coeffs = solve_proposed_milp(FIG3_PENALTY_RATE, W1, W2)
    pred_flat = predict_test(coeffs)
    nrmse = compute_nrmse(pred_flat)
    gap = compute_gap(pred_flat, FIG3_PENALTY_RATE)
    fig3_nrmse_list.append(nrmse)
    fig3_gap_list.append(gap)
    print(f"  {W1}/{W2}: nRMSE={nrmse:.2f}%, Gap={gap:.2f}%  "
          f"(논문: nRMSE={FIG3_PAPER_NRMSE[idx]:.2f}%, Gap={FIG3_PAPER_GAP[idx]:.2f}%)")


# =====================================================================
# 10. Fig.5 데이터 생성 — W1=W2=1 고정, 벌금율 0~100% 스윕
# =====================================================================
print("\n" + "=" * 60)
print(f"  Fig.5 데이터 생성 (W1/W2={FIG5_W1:.0f}/{FIG5_W2:.0f} 고정, 4항 이익함수)")
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
# 11. 결과 CSV 저장
# =====================================================================
out_dir = os.path.join(BASE_DIR, "results")
os.makedirs(out_dir, exist_ok=True)

fig3_paper_nrmse_all = [FIG3_PAPER_AR_NRMSE] + FIG3_PAPER_NRMSE
fig3_paper_gap_all = [FIG3_PAPER_AR_GAP] + FIG3_PAPER_GAP

fig3_csv_path = os.path.join(out_dir, "model_proposed_ar_profit_change_v3_fig3.csv")
with open(fig3_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "nRMSE", "Gap", "Paper_nRMSE", "Paper_Gap"])
    for label, nrmse, gap, p_nrmse, p_gap in zip(
            FIG3_LABELS, fig3_nrmse_list, fig3_gap_list, fig3_paper_nrmse_all, fig3_paper_gap_all):
        writer.writerow([label, round(nrmse, 4), round(gap, 4), p_nrmse, p_gap])
print(f"\nsaved: {fig3_csv_path}")

fig5_csv_path = os.path.join(out_dir, "model_proposed_ar_profit_change_v3_fig5.csv")
with open(fig5_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Rate", "AR_nRMSE", "AR_Gap", "Proposed_nRMSE", "Proposed_Gap"])
    for rate, an, ag, pn, pg in zip(
            FIG5_PENALTY_RATES, fig5_ar_nrmse_list, fig5_ar_gap_list, fig5_prop_nrmse_list, fig5_prop_gap_list):
        writer.writerow([rate, round(an, 4), round(ag, 4), round(pn, 4), round(pg, 4)])
print(f"saved: {fig5_csv_path}")


# =====================================================================
# 12. Fig.3 그리기 — 하나의 그래프에 nRMSE(왼쪽 y축) + Gap(오른쪽 y축)
#     (논문 원본 PDF 8쪽과 동일한 dual y축 구성)
# =====================================================================
x3_positions = list(range(len(FIG3_LABELS)))

figure3, axis_left = plt.subplots(figsize=(10, 6))
figure3.suptitle("Fig.3 재현 — AR + 제안 모형(4항 이익함수), 벌금율 50%, W1/W2 스윕 (dual y축)", fontsize=13)

axis_right = axis_left.twinx()

line1 = axis_left.plot(x3_positions, fig3_paper_nrmse_all, marker="o", color="#2a78d6",
                        linestyle="-", label="논문 nRMSE")
line2 = axis_left.plot(x3_positions, fig3_nrmse_list, marker="o", color="#0b3d78",
                        linestyle="-", label="재현 nRMSE")

line3 = axis_right.plot(x3_positions, fig3_paper_gap_all, marker="s", color="#eb9834",
                         linestyle="--", label="논문 Gap")
line4 = axis_right.plot(x3_positions, fig3_gap_list, marker="s", color="#b34700",
                         linestyle="--", label="재현 Gap")

axis_left.set_xticks(x3_positions)
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
fig3_png_path = os.path.join(out_dir, "model_proposed_ar_profit_change_v3_fig3.png")
figure3.savefig(fig3_png_path, dpi=150)
print(f"\nsaved: {fig3_png_path}")


# =====================================================================
# 13. Fig.5 그리기 — nRMSE / Gap 을 좌우 두 그래프로 나눠서 (AR + 제안모형)
# =====================================================================
x5_labels = [f"{int(round(r * 100))}%" for r in FIG5_PENALTY_RATES]
x5_positions = list(range(len(x5_labels)))

figure5, (axis_nrmse, axis_gap) = plt.subplots(1, 2, figsize=(13, 5))
figure5.suptitle("Fig.5 재현 — AR + 제안 모형(4항 이익함수), W1/W2=1/1, 벌금비용률 스윕", fontsize=14)

axis_nrmse.plot(x5_positions, FIG5_PAPER_AR_NRMSE, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 AR")
axis_nrmse.plot(x5_positions, FIG5_PAPER_PROP_NRMSE, linestyle="--", color="#eb6834", alpha=0.6, label="논문 제안모형")
axis_nrmse.plot(x5_positions, fig5_ar_nrmse_list, marker="o", color="#2a78d6", label="재현 AR")
axis_nrmse.plot(x5_positions, fig5_prop_nrmse_list, marker="o", color="#eb6834", label="재현 제안모형")
axis_nrmse.set_xticks(x5_positions)
axis_nrmse.set_xticklabels(x5_labels)
axis_nrmse.set_xlabel("벌금비용률(penalty cost rate)")
axis_nrmse.set_ylabel("nRMSE (%)")
axis_nrmse.set_title("nRMSE 비교")
axis_nrmse.grid(True, alpha=0.3)
axis_nrmse.legend(fontsize=8)

axis_gap.plot(x5_positions, FIG5_PAPER_AR_GAP, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 AR")
axis_gap.plot(x5_positions, FIG5_PAPER_PROP_GAP, linestyle="--", color="#eb6834", alpha=0.6, label="논문 제안모형")
axis_gap.plot(x5_positions, fig5_ar_gap_list, marker="o", color="#2a78d6", label="재현 AR")
axis_gap.plot(x5_positions, fig5_prop_gap_list, marker="o", color="#eb6834", label="재현 제안모형")
axis_gap.set_xticks(x5_positions)
axis_gap.set_xticklabels(x5_labels)
axis_gap.set_xlabel("벌금비용률(penalty cost rate)")
axis_gap.set_ylabel("Optimality Gap (%)")
axis_gap.set_title("Optimality Gap 비교")
axis_gap.grid(True, alpha=0.3)
axis_gap.legend(fontsize=8)

figure5.tight_layout()
fig5_png_path = os.path.join(out_dir, "model_proposed_ar_profit_change_v3_fig5.png")
figure5.savefig(fig5_png_path, dpi=150)
print(f"saved: {fig5_png_path}")

print("\n완료.")
