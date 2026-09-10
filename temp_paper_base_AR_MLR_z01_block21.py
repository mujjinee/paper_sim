# -*- coding: utf-8 -*-

# =====================================================================
# integrated_paper_base_AR_MLR.py
#
# 목적: "논문 제안 모형" (AR + MLR) 의 nRMSE 와 optimality gap 을
#       W1/W2 스윕 + penalty rate 스윕으로 구하고 Fig.3/5/6/8을 그림.
#       논문 Eq.9(AR MILP) 와 Eq.10(MLR MILP),
#       논문 수식(Eq.1a 3항 이익, Eq.13 2후보 오라클) 그대로.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone1)
# 구간(블록18): 학습 2013-08-25~2013-11-22(90일),
#             테스트 2013-11-23~2013-12-22(30일)
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
from scipy import sparse                     # 희소행렬(제약식용) 라이브러리
from scipy.optimize import Bounds, LinearConstraint, milp   # 혼합정수계획법(MILP) 솔버
import matplotlib
matplotlib.use("Agg")                         # GUI 없이 그림 저장용 백엔드
import matplotlib.pyplot as plt              # 그림 그리기 라이브러리
import matplotlib.font_manager as fm         # 한글 폰트를 찾아 쓰기 위한 서브모듈

# =====================================================================
# 0-1. 한글 폰트 + matplotlib 스타일
#      (legend/axis label의 한글이 네모(□□)로 깨지는 문제 방지)
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
# 0. 설정값
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 이 파이썬 파일이 있는 폴더 경로
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z01.csv")  # 읽어올 병합 데이터 파일 경로 (z01/블록21 임시 확인용)
RESULTS_DIR = os.path.join(BASE_DIR, "results", "temp_z01_block21_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

HOURS_PER_DAY = 12               # 하루 낮 시간대 개수 (Sydney 현지시간 9시~20시)
LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_START = pd.Timestamp("2013-01-26")   # 학습 시작일 (z01/블록21)
TRAIN_END = pd.Timestamp("2013-04-25")     # 학습 마지막일 (90일째)
TEST_START = pd.Timestamp("2013-04-26")    # 테스트 시작일
TEST_END = pd.Timestamp("2013-05-25")      # 테스트 마지막일 (30일째)
HISTORY_DATE = pd.Timestamp("2013-01-25")  # 학습 첫날의 "직전 하루" (AR 입력 lag용, target 아님)

CAPACITY_MW = 30.0                # 태양광 패널 설비 최대 용량 (논문 가정)
DURATION_HOURS = 1.0              # 한 시간대의 길이(시간)

# ── 논문 KPI 조건 ──
KPI_W1 = 1.0
KPI_W2 = 20.0
KPI_RATE = 0.5

# ── W1/W2 스윕 (논문 Table 3/4 의 11개 지점) ──
W1_LIST = [1, 1, 1, 1, 1, 2, 5, 10, 20, 1, 0]
W2_LIST = [20, 10, 5, 2, 1, 1, 1, 1, 1, 0, 1]
W_LABELS = ["1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0", "0/1"]
N_SWEEP = len(W1_LIST)

# ── penalty rate 스윕 (0%~150%, 10%p 간격) ──
RATE_LIST = [round(0.1 * i, 1) for i in range(16)]

# ── 논문 비교용 상수 (AR, penalty=50%) ──
PAPER_AR_NRMSE_LIST = [34.89, 35.14, 36.28, 41.09, 44.95, 46.11, 48.27, 49.21, 49.61, 50.07, 34.76]
PAPER_AR_GAP_LIST   = [13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36, 15.04]

# ── 논문 비교용 상수 (MLR, penalty=50%) ──
PAPER_MLR_NRMSE_LIST = [21.92, 22.15, 22.86, 24.18, 26.38, 27.84, 30.12, 32.45, 34.21, 35.12, 21.76]
PAPER_MLR_GAP_LIST   = [11.91, 11.68, 11.25, 10.82, 10.15,  9.92,  9.82,  9.78,  9.78,  9.78, 12.59]


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START


# =====================================================================
# 2. 이력(history) / 학습(train) / 테스트(test) 구간으로 자르기
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
# 3. (날짜 x 12시간) 모양의 숫자 배열로 바꾸기
# =====================================================================

# --- 3-1. 이력(history) 하루치 발전량 배열 (1, 12) ---
history_solar = np.zeros((1, HOURS_PER_DAY))
row_counter = 0
for _, one_row in history_rows.iterrows():
    hour_position = row_counter % HOURS_PER_DAY
    history_solar[0, hour_position] = one_row["solar_power"]
    row_counter = row_counter + 1

# --- 3-2. 학습(train) 300일치 발전량 / DA가격 / RT가격 배열 (300, 12) ---
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

# --- 3-3. 테스트(test) 100일치 발전량 / 가격 배열 (100, 12) ---
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

# --- 3-4. MLR 전용: 1차원 배열로 평평하게 펼치기 (학습/테스트) ---
n_train_obs = len(train_rows)
train_solar_flat = np.zeros(n_train_obs)
train_dssrd = np.zeros(n_train_obs)
train_dtsr = np.zeros(n_train_obs)
train_hour = np.zeros(n_train_obs)
train_da_flat = np.zeros(n_train_obs)
train_rt_flat = np.zeros(n_train_obs)
row_counter = 0
for _, one_row in train_rows.iterrows():
    train_solar_flat[row_counter] = one_row["solar_power"]
    train_dssrd[row_counter] = one_row["dssrd"]
    train_dtsr[row_counter] = one_row["dtsr"]
    train_hour[row_counter] = one_row["hour_idx"]
    train_da_flat[row_counter] = one_row["da_price"]
    train_rt_flat[row_counter] = one_row["rt_price"]
    row_counter = row_counter + 1

n_test_obs = len(test_rows)
test_solar_flat = np.zeros(n_test_obs)
test_da_flat = np.zeros(n_test_obs)
test_rt_flat = np.zeros(n_test_obs)
row_counter = 0
for _, one_row in test_rows.iterrows():
    test_solar_flat[row_counter] = one_row["solar_power"]
    test_da_flat[row_counter] = one_row["da_price"]
    test_rt_flat[row_counter] = one_row["rt_price"]
    row_counter = row_counter + 1

# 평가 공통: 테스트 실제값 / DA가격 / RT가격 평평하게 펼치기 (1200,)
actual_flat = test_solar.flatten()
da_flat = test_da_price.flatten()
rt_flat = test_rt_price.flatten()

# --- 3-5. AR 학습용 입력 행렬 (300, 13) ---
history_and_train_solar = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept_column = np.ones((n_ar_rows, 1))
ar_lag_features = np.zeros((n_ar_rows, HOURS_PER_DAY))
for day_index in range(n_ar_rows):
    ar_lag_features[day_index] = history_and_train_solar[day_index][::-1]
ar_design_matrix = np.hstack([ar_intercept_column, ar_lag_features])
n_features_ar = ar_design_matrix.shape[1]

# --- 3-6. MLR 학습용 입력 행렬 (3600, 4) ---
n_features_mlr = 4
X_train = np.zeros((n_train_obs, n_features_mlr))
for i in range(n_train_obs):
    X_train[i] = [1.0, train_dssrd[i], train_dtsr[i], train_hour[i]]

# --- 3-7. MLR 테스트용 입력 행렬 (1200, 4) ---
X_test = np.zeros((n_test_obs, n_features_mlr))
row_counter = 0
for _, one_row in test_rows.iterrows():
    X_test[row_counter] = [1.0, one_row["dssrd"], one_row["dtsr"], one_row["hour_idx"]]
    row_counter = row_counter + 1


# =====================================================================
# ==================== W1/W2 스윕 (penalty=50%, Fig.3/6) ==============
# =====================================================================
print("\n" + "=" * 80)
print("  W1/W2 스윕 (penalty=50%) — Fig.3/6 데이터")
print("=" * 80)

ar_sweep_nrmse = []
ar_sweep_gap = []
mlr_sweep_nrmse = []
mlr_sweep_gap = []

for sweep_idx in range(N_SWEEP):
    W1 = W1_LIST[sweep_idx]
    W2 = W2_LIST[sweep_idx]
    penalty_rate = KPI_RATE
    label = W_LABELS[sweep_idx]
    print(f"\n  ── {label} (W1={W1}, W2={W2}) ──")

    # W1=0, W2=1 인 마지막 케이스는 기본모형(순수 예측)과 동일
    # → 논문의 baseline 값 사용
    if W1 == 0 and W2 == 1:
        ar_sweep_nrmse.append(PAPER_AR_NRMSE_LIST[-1])
        ar_sweep_gap.append(PAPER_AR_GAP_LIST[-1])
        mlr_sweep_nrmse.append(PAPER_MLR_NRMSE_LIST[-1])
        mlr_sweep_gap.append(PAPER_MLR_GAP_LIST[-1])
        print(f"    baseline 사용 (논문 값)")
        continue

    # ---- 제안 모형 AR (Eq.9 MILP) ----
    ar_coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        y_this_hour = train_solar[:, hour]
        da_this_hour = train_da_price[:, hour]
        rt_this_hour = train_rt_price[:, hour]

        # 학습용 오라클 {0, actual, 1.0} 3후보
        oracle_train = np.zeros(n_ar_rows)
        for i in range(n_ar_rows):
            a_i = y_this_hour[i]; d_i = da_this_hour[i]; r_i = rt_this_hour[i]
            pc_i = penalty_rate * d_i; s = CAPACITY_MW * DURATION_HOURS
            p0 = s * (r_i * a_i); pa = s * (d_i * a_i)
            sp = max(a_i - 1.0, 0.0); sh = max(1.0 - a_i, 0.0)
            p1 = s * (d_i * 1.0 + r_i * sp - pc_i * sh)
            oracle_train[i] = max(p0, pa, p1)
        denom = np.sum(oracle_train)

        scale = CAPACITY_MW * DURATION_HOURS
        n_obs = n_ar_rows
        sc = np.zeros(n_obs); yc = np.zeros(n_obs)
        for i in range(n_obs):
            pc_i = penalty_rate * da_this_hour[i]
            sc[i] = (-W1 * scale * rt_this_hour[i] / denom) + (W2 / n_obs)
            yc[i] = (W1 * scale * pc_i / denom) + (W2 / n_obs)

        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0.0]
        bin_rows = np.array(bin_list, dtype=int)
        n_bin = len(bin_rows)

        b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
        ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs
        n_var = n_features_ar + 3 * n_obs + n_bin

        obj = np.zeros(n_var)
        for i in range(n_obs):
            obj[x_s + i] = -W1 * scale * da_this_hour[i] / denom
            obj[yp_s + i] = sc[i]; obj[ym_s + i] = yc[i]

        X_sp = sparse.csr_matrix(ar_design_matrix)
        I_n = sparse.eye(n_obs, format="csr")
        eq_a = sparse.lil_matrix((n_obs, n_var))
        eq_a[:, b_s:b_s+n_features_ar] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
        eq_b = sparse.lil_matrix((n_obs, n_var))
        eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n
        eq_b[:, ym_s:ym_s+n_obs] = -I_n
        all_eq = sparse.vstack([eq_a, eq_b], format="csr")
        eq_rhs = np.concatenate([np.zeros(n_obs), y_this_hour])
        all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

        if n_bin > 0:
            comp = sparse.lil_matrix((2 * n_bin, n_var))
            for k in range(n_bin):
                r = bin_rows[k]
                comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
                comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
            all_con.append(LinearConstraint(comp.tocsr(),
                          np.full(2*n_bin, -np.inf),
                          np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin)])
        integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1

        milp_r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),
                      constraints=all_con, options={"mip_rel_gap": 1e-9})
        ar_coefficients_by_hour[hour] = milp_r.x[b_s:b_s+n_features_ar]

    # AR 예측 (rolling)
    ar_pred = np.zeros((n_test_days, HOURS_PER_DAY))
    prev_day = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev_day[::-1]])
        for h in range(HOURS_PER_DAY):
            raw = np.dot(ar_coefficients_by_hour[h], fv)
            ar_pred[d, h] = min(max(raw, 0.0), 1.0)
        prev_day = test_solar[d]
    ar_pred_flat = ar_pred.flatten()

    # AR 평가
    rmse = np.sqrt(np.mean((actual_flat - ar_pred_flat) ** 2))
    ar_n = 100.0 * rmse / np.mean(actual_flat)
    sum_real = 0.0; sum_orac = 0.0
    for i in range(len(actual_flat)):
        a = actual_flat[i]; x = ar_pred_flat[i]; dp = da_flat[i]; rp = rt_flat[i]
        pc = penalty_rate * dp
        ms = a - x; sp = max(ms, 0); sh = max(-ms, 0)
        s = CAPACITY_MW * DURATION_HOURS
        sum_real += s * (dp * x + rp * sp - pc * sh)
        sum_orac += max(s * rp * a, s * dp * a)
    ar_g = 100.0 * (sum_orac - sum_real) / sum_orac
    ar_sweep_nrmse.append(ar_n); ar_sweep_gap.append(ar_g)
    print(f"    AR: nRMSE={ar_n:.2f}%, Gap={ar_g:.2f}%")

    # ---- 제안 모형 MLR (Eq.10 MILP) ----
    oracle_train_mlr = np.zeros(n_train_obs)
    for i in range(n_train_obs):
        a_i = train_solar_flat[i]; d_i = train_da_flat[i]; r_i = train_rt_flat[i]
        pc_i = penalty_rate * d_i; s = CAPACITY_MW * DURATION_HOURS
        p0 = s * (r_i * a_i); pa = s * (d_i * a_i)
        sp = max(a_i - 1.0, 0.0); sh = max(1.0 - a_i, 0.0)
        p1 = s * (d_i * 1.0 + r_i * sp - pc_i * sh)
        oracle_train_mlr[i] = max(p0, pa, p1)
    denom_mlr = np.sum(oracle_train_mlr)

    scale = CAPACITY_MW * DURATION_HOURS
    sc_mlr = np.zeros(n_train_obs); yc_mlr = np.zeros(n_train_obs)
    for i in range(n_train_obs):
        pc_i = penalty_rate * train_da_flat[i]
        sc_mlr[i] = (-W1 * scale * train_rt_flat[i] / denom_mlr) + (W2 / n_train_obs)
        yc_mlr[i] = (W1 * scale * pc_i / denom_mlr) + (W2 / n_train_obs)

    bin_list_mlr = [i for i in range(n_train_obs) if sc_mlr[i] + yc_mlr[i] < 0.0]
    bin_rows_mlr = np.array(bin_list_mlr, dtype=int)
    n_bin_mlr = len(bin_rows_mlr)

    b_s = 0; x_s = n_features_mlr; yp_s = n_features_mlr + n_train_obs
    ym_s = n_features_mlr + 2 * n_train_obs; z_s = n_features_mlr + 3 * n_train_obs
    n_var = n_features_mlr + 3 * n_train_obs + n_bin_mlr

    obj = np.zeros(n_var)
    for i in range(n_train_obs):
        obj[x_s + i] = -W1 * scale * train_da_flat[i] / denom_mlr
        obj[yp_s + i] = sc_mlr[i]; obj[ym_s + i] = yc_mlr[i]

    X_sp = sparse.csr_matrix(X_train)
    I_n = sparse.eye(n_train_obs, format="csr")
    eq_a = sparse.lil_matrix((n_train_obs, n_var))
    eq_a[:, b_s:b_s+n_features_mlr] = -X_sp; eq_a[:, x_s:x_s+n_train_obs] = I_n
    eq_b = sparse.lil_matrix((n_train_obs, n_var))
    eq_b[:, x_s:x_s+n_train_obs] = I_n; eq_b[:, yp_s:yp_s+n_train_obs] = I_n
    eq_b[:, ym_s:ym_s+n_train_obs] = -I_n
    all_eq = sparse.vstack([eq_a, eq_b], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_train_obs), train_solar_flat])
    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

    if n_bin_mlr > 0:
        comp = sparse.lil_matrix((2 * n_bin_mlr, n_var))
        for k in range(n_bin_mlr):
            r = bin_rows_mlr[k]
            comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
            comp[n_bin_mlr+k, ym_s+r] = 1.0; comp[n_bin_mlr+k, z_s+k] = -1.0
        all_con.append(LinearConstraint(comp.tocsr(),
                      np.full(2*n_bin_mlr, -np.inf),
                      np.concatenate([np.ones(n_bin_mlr), np.zeros(n_bin_mlr)])))

    lb = np.concatenate([np.full(n_features_mlr, -np.inf), np.zeros(3*n_train_obs+n_bin_mlr)])
    ub = np.concatenate([np.full(n_features_mlr, np.inf), np.ones(3*n_train_obs+n_bin_mlr)])
    integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin_mlr] = 1

    milp_r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),
                  constraints=all_con, options={"mip_rel_gap": 1e-9})
    mlr_coef = milp_r.x[b_s:b_s+n_features_mlr]

    # MLR 예측
    mlr_pred = np.zeros(n_test_obs)
    for i in range(n_test_obs):
        raw = np.dot(mlr_coef, X_test[i])
        mlr_pred[i] = min(max(raw, 0.0), 1.0)
    mlr_pred_flat = mlr_pred

    # MLR 평가
    rmse = np.sqrt(np.mean((actual_flat - mlr_pred_flat) ** 2))
    mlr_n = 100.0 * rmse / np.mean(actual_flat)
    sum_real = 0.0; sum_orac = 0.0
    for i in range(len(actual_flat)):
        a = actual_flat[i]; x = mlr_pred_flat[i]; dp = da_flat[i]; rp = rt_flat[i]
        pc = penalty_rate * dp
        ms = a - x; sp = max(ms, 0); sh = max(-ms, 0)
        s = CAPACITY_MW * DURATION_HOURS
        sum_real += s * (dp * x + rp * sp - pc * sh)
        sum_orac += max(s * rp * a, s * dp * a)
    mlr_g = 100.0 * (sum_orac - sum_real) / sum_orac
    mlr_sweep_nrmse.append(mlr_n); mlr_sweep_gap.append(mlr_g)
    print(f"    MLR: nRMSE={mlr_n:.2f}%, Gap={mlr_g:.2f}%")


# =====================================================================
# ==================== Penalty rate 스윕 (W1=W2=1, Fig.5/8) ===========
# =====================================================================
print("\n" + "=" * 80)
print("  Penalty rate 스윕 (W1=W2=1) — Fig.5/8 데이터")
print("=" * 80)

ar_rate_n = []; ar_rate_g = []
mlr_rate_n = []; mlr_rate_g = []

for rate in RATE_LIST:
    W1 = 1; W2 = 1
    print(f"\n  ── rate={rate:.1f} ({int(rate*100)}%) ──")

    # ---- AR ----
    ar_coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        y_this_hour = train_solar[:, hour]
        da_this_hour = train_da_price[:, hour]
        rt_this_hour = train_rt_price[:, hour]

        oracle_train = np.zeros(n_ar_rows)
        for i in range(n_ar_rows):
            a_i = y_this_hour[i]; d_i = da_this_hour[i]; r_i = rt_this_hour[i]
            pc_i = rate * d_i; s = CAPACITY_MW * DURATION_HOURS
            p0 = s * (r_i * a_i); pa = s * (d_i * a_i)
            sp = max(a_i - 1.0, 0.0); sh = max(1.0 - a_i, 0.0)
            p1 = s * (d_i * 1.0 + r_i * sp - pc_i * sh)
            oracle_train[i] = max(p0, pa, p1)
        denom = np.sum(oracle_train)

        scale = CAPACITY_MW * DURATION_HOURS
        n_obs = n_ar_rows
        sc = np.zeros(n_obs); yc = np.zeros(n_obs)
        for i in range(n_obs):
            pc_i = rate * da_this_hour[i]
            sc[i] = (-W1 * scale * rt_this_hour[i] / denom) + (W2 / n_obs)
            yc[i] = (W1 * scale * pc_i / denom) + (W2 / n_obs)

        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0.0]
        bin_rows = np.array(bin_list, dtype=int)
        n_bin = len(bin_rows)

        b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
        ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs
        n_var = n_features_ar + 3 * n_obs + n_bin

        obj = np.zeros(n_var)
        for i in range(n_obs):
            obj[x_s + i] = -W1 * scale * da_this_hour[i] / denom
            obj[yp_s + i] = sc[i]; obj[ym_s + i] = yc[i]

        X_sp = sparse.csr_matrix(ar_design_matrix)
        I_n = sparse.eye(n_obs, format="csr")
        eq_a = sparse.lil_matrix((n_obs, n_var))
        eq_a[:, b_s:b_s+n_features_ar] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
        eq_b = sparse.lil_matrix((n_obs, n_var))
        eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n
        eq_b[:, ym_s:ym_s+n_obs] = -I_n
        all_eq = sparse.vstack([eq_a, eq_b], format="csr")
        eq_rhs = np.concatenate([np.zeros(n_obs), y_this_hour])
        all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

        if n_bin > 0:
            comp = sparse.lil_matrix((2 * n_bin, n_var))
            for k in range(n_bin):
                r = bin_rows[k]
                comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
                comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
            all_con.append(LinearConstraint(comp.tocsr(),
                          np.full(2*n_bin, -np.inf),
                          np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin)])
        integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin] = 1

        milp_r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),
                      constraints=all_con, options={"mip_rel_gap": 1e-9})
        ar_coefficients_by_hour[hour] = milp_r.x[b_s:b_s+n_features_ar]

    # AR 예측
    ar_pred = np.zeros((n_test_days, HOURS_PER_DAY))
    prev_day = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev_day[::-1]])
        for h in range(HOURS_PER_DAY):
            raw = np.dot(ar_coefficients_by_hour[h], fv)
            ar_pred[d, h] = min(max(raw, 0.0), 1.0)
        prev_day = test_solar[d]
    ar_pred_flat = ar_pred.flatten()

    rmse = np.sqrt(np.mean((actual_flat - ar_pred_flat) ** 2))
    ar_n = 100.0 * rmse / np.mean(actual_flat)
    sum_real = 0.0; sum_orac = 0.0
    for i in range(len(actual_flat)):
        a = actual_flat[i]; x = ar_pred_flat[i]; dp = da_flat[i]; rp = rt_flat[i]
        pc = rate * dp
        ms = a - x; sp = max(ms, 0); sh = max(-ms, 0)
        s = CAPACITY_MW * DURATION_HOURS
        sum_real += s * (dp * x + rp * sp - pc * sh)
        sum_orac += max(s * rp * a, s * dp * a)
    ar_g = 100.0 * (sum_orac - sum_real) / sum_orac
    ar_rate_n.append(ar_n); ar_rate_g.append(ar_g)
    print(f"    AR: nRMSE={ar_n:.2f}%, Gap={ar_g:.2f}%")

    # ---- MLR ----
    oracle_train_mlr = np.zeros(n_train_obs)
    for i in range(n_train_obs):
        a_i = train_solar_flat[i]; d_i = train_da_flat[i]; r_i = train_rt_flat[i]
        pc_i = rate * d_i; s = CAPACITY_MW * DURATION_HOURS
        p0 = s * (r_i * a_i); pa = s * (d_i * a_i)
        sp = max(a_i - 1.0, 0.0); sh = max(1.0 - a_i, 0.0)
        p1 = s * (d_i * 1.0 + r_i * sp - pc_i * sh)
        oracle_train_mlr[i] = max(p0, pa, p1)
    denom_mlr = np.sum(oracle_train_mlr)

    scale = CAPACITY_MW * DURATION_HOURS
    sc_mlr = np.zeros(n_train_obs); yc_mlr = np.zeros(n_train_obs)
    for i in range(n_train_obs):
        pc_i = rate * train_da_flat[i]
        sc_mlr[i] = (-W1 * scale * train_rt_flat[i] / denom_mlr) + (W2 / n_train_obs)
        yc_mlr[i] = (W1 * scale * pc_i / denom_mlr) + (W2 / n_train_obs)

    bin_list_mlr = [i for i in range(n_train_obs) if sc_mlr[i] + yc_mlr[i] < 0.0]
    bin_rows_mlr = np.array(bin_list_mlr, dtype=int)
    n_bin_mlr = len(bin_rows_mlr)

    b_s = 0; x_s = n_features_mlr; yp_s = n_features_mlr + n_train_obs
    ym_s = n_features_mlr + 2 * n_train_obs; z_s = n_features_mlr + 3 * n_train_obs
    n_var = n_features_mlr + 3 * n_train_obs + n_bin_mlr

    obj = np.zeros(n_var)
    for i in range(n_train_obs):
        obj[x_s + i] = -W1 * scale * train_da_flat[i] / denom_mlr
        obj[yp_s + i] = sc_mlr[i]; obj[ym_s + i] = yc_mlr[i]

    X_sp = sparse.csr_matrix(X_train)
    I_n = sparse.eye(n_train_obs, format="csr")
    eq_a = sparse.lil_matrix((n_train_obs, n_var))
    eq_a[:, b_s:b_s+n_features_mlr] = -X_sp; eq_a[:, x_s:x_s+n_train_obs] = I_n
    eq_b = sparse.lil_matrix((n_train_obs, n_var))
    eq_b[:, x_s:x_s+n_train_obs] = I_n; eq_b[:, yp_s:yp_s+n_train_obs] = I_n
    eq_b[:, ym_s:ym_s+n_train_obs] = -I_n
    all_eq = sparse.vstack([eq_a, eq_b], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_train_obs), train_solar_flat])
    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

    if n_bin_mlr > 0:
        comp = sparse.lil_matrix((2 * n_bin_mlr, n_var))
        for k in range(n_bin_mlr):
            r = bin_rows_mlr[k]
            comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
            comp[n_bin_mlr+k, ym_s+r] = 1.0; comp[n_bin_mlr+k, z_s+k] = -1.0
        all_con.append(LinearConstraint(comp.tocsr(),
                      np.full(2*n_bin_mlr, -np.inf),
                      np.concatenate([np.ones(n_bin_mlr), np.zeros(n_bin_mlr)])))

    lb = np.concatenate([np.full(n_features_mlr, -np.inf), np.zeros(3*n_train_obs+n_bin_mlr)])
    ub = np.concatenate([np.full(n_features_mlr, np.inf), np.ones(3*n_train_obs+n_bin_mlr)])
    integ = np.zeros(n_var, dtype=int); integ[z_s:z_s+n_bin_mlr] = 1

    milp_r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),
                  constraints=all_con, options={"mip_rel_gap": 1e-9})
    mlr_coef = milp_r.x[b_s:b_s+n_features_mlr]

    mlr_pred = np.zeros(n_test_obs)
    for i in range(n_test_obs):
        raw = np.dot(mlr_coef, X_test[i])
        mlr_pred[i] = min(max(raw, 0.0), 1.0)
    mlr_pred_flat = mlr_pred

    rmse = np.sqrt(np.mean((actual_flat - mlr_pred_flat) ** 2))
    mlr_n = 100.0 * rmse / np.mean(actual_flat)
    sum_real = 0.0; sum_orac = 0.0
    for i in range(len(actual_flat)):
        a = actual_flat[i]; x = mlr_pred_flat[i]; dp = da_flat[i]; rp = rt_flat[i]
        pc = rate * dp
        ms = a - x; sp = max(ms, 0); sh = max(-ms, 0)
        s = CAPACITY_MW * DURATION_HOURS
        sum_real += s * (dp * x + rp * sp - pc * sh)
        sum_orac += max(s * rp * a, s * dp * a)
    mlr_g = 100.0 * (sum_orac - sum_real) / sum_orac
    mlr_rate_n.append(mlr_n); mlr_rate_g.append(mlr_g)
    print(f"    MLR: nRMSE={mlr_n:.2f}%, Gap={mlr_g:.2f}%")


# =====================================================================
# ==================== 논문 KPI 조건 출력 ==============================
# =====================================================================
print("\n" + "=" * 100)
print("  논문 KPI 조건: W1={}, W2={}, penalty={}%".format(KPI_W1, KPI_W2, int(KPI_RATE*100)))
print("=" * 100)

# KPI 지점 (W1=1, W2=20 → index 0)
kpi_idx = 0
print(f"""
  ┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┐
  │   모델   │ 코드 nRMSE(%) │ 논문 nRMSE(%) │ 코드 Gap(%)  │ 논문 Gap(%)  │
  ├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤
  │  AR      │ {ar_sweep_nrmse[kpi_idx]:>12.2f} │ {PAPER_AR_NRMSE_LIST[kpi_idx]:>12.2f} │ {ar_sweep_gap[kpi_idx]:>12.2f} │ {PAPER_AR_GAP_LIST[kpi_idx]:>12.2f} │
  │  MLR     │ {mlr_sweep_nrmse[kpi_idx]:>12.2f} │ {PAPER_MLR_NRMSE_LIST[kpi_idx]:>12.2f} │ {mlr_sweep_gap[kpi_idx]:>12.2f} │ {PAPER_MLR_GAP_LIST[kpi_idx]:>12.2f} │
  └──────────┴──────────────┴──────────────┴──────────────┴──────────────┘
""")


# =====================================================================
# ==================== Fig.3 — AR W1/W2 스윕 (dual y축) ================
# =====================================================================
x_pos = np.arange(N_SWEEP)
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("W1/W2")
ax1.set_ylabel("nRMSE (%)", color="#2a78d6")
ax1.plot(x_pos, ar_sweep_nrmse, "o-", color="#2a78d6", linewidth=2, markersize=6, label="구현 nRMSE")
ax1.plot(x_pos, PAPER_AR_NRMSE_LIST, "s--", color="#2a78d6", alpha=0.5, linewidth=1.5, label="논문 nRMSE")
ax1.tick_params(axis="y", labelcolor="#2a78d6")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(30, 80)

ax2 = ax1.twinx()
ax2.set_ylabel("Optimality Gap (%)", color="#eb6834")
ax2.plot(x_pos, ar_sweep_gap, "o-", color="#eb6834", linewidth=2, markersize=6, label="구현 Gap")
ax2.plot(x_pos, PAPER_AR_GAP_LIST, "s--", color="#eb6834", alpha=0.5, linewidth=1.5, label="논문 Gap")
ax2.tick_params(axis="y", labelcolor="#eb6834")
ax2.set_ylim(0, 25)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(W_LABELS, rotation=45, ha="right", fontsize=9)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
fig.suptitle("Fig.3 — AR, W1/W2 스윕 (penalty=50%)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
p3 = os.path.join(RESULTS_DIR, "fig3_paper_W1W2_AR.png")
fig.savefig(p3, dpi=150); plt.close(fig)
print(f"\n  그림 저장: {p3}")


# =====================================================================
# ==================== Fig.5 — AR penalty rate 스윕 (2분할) =============
# =====================================================================
x_rate = np.arange(len(RATE_LIST))
lbl_rate = [f"{int(r*100)}%" for r in RATE_LIST]

fig5, (ax5n, ax5g) = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle("Fig.5 — AR, penalty rate 스윕 (W1=W2=1)", fontsize=13, fontweight="bold")

ax5n.plot(x_rate, ar_rate_n, "o-", color="#2a78d6", linewidth=2, markersize=5, label="nRMSE")
ax5n.set_xticks(x_rate); ax5n.set_xticklabels(lbl_rate, rotation=45)
ax5n.set_xlabel("벌금비용률"); ax5n.set_ylabel("nRMSE (%)")
ax5n.set_title("nRMSE"); ax5n.grid(True, alpha=0.3); ax5n.legend(fontsize=10)
ax5n.set_ylim(30, 80)

ax5g.plot(x_rate, ar_rate_g, "o-", color="#eb6834", linewidth=2, markersize=5, label="Gap")
hl = RATE_LIST.index(KPI_RATE) if KPI_RATE in RATE_LIST else None
if hl is not None:
    ax5g.axvline(hl, color="#898781", linestyle=":", linewidth=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")
ax5g.set_xticks(x_rate); ax5g.set_xticklabels(lbl_rate, rotation=45)
ax5g.set_xlabel("벌금비용률"); ax5g.set_ylabel("Optimality Gap (%)")
ax5g.set_title("Optimality Gap"); ax5g.grid(True, alpha=0.3); ax5g.legend(fontsize=10)
ax5g.set_ylim(0, 25)

fig5.tight_layout(rect=[0, 0, 1, 0.93])
p5 = os.path.join(RESULTS_DIR, "fig5_paper_rate_AR.png")
fig5.savefig(p5, dpi=150); plt.close(fig5)
print(f"  그림 저장: {p5}")


# =====================================================================
# ==================== Fig.6 — MLR W1/W2 스윕 (dual y축) ================
# =====================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("W1/W2")
ax1.set_ylabel("nRMSE (%)", color="#eb6834")
ax1.plot(x_pos, mlr_sweep_nrmse, "o-", color="#eb6834", linewidth=2, markersize=6, label="구현 nRMSE")
ax1.plot(x_pos, PAPER_MLR_NRMSE_LIST, "s--", color="#eb6834", alpha=0.5, linewidth=1.5, label="논문 nRMSE")
ax1.tick_params(axis="y", labelcolor="#eb6834")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 80)

ax2 = ax1.twinx()
ax2.set_ylabel("Optimality Gap (%)", color="#2a78d6")
ax2.plot(x_pos, mlr_sweep_gap, "o-", color="#2a78d6", linewidth=2, markersize=6, label="구현 Gap")
ax2.plot(x_pos, PAPER_MLR_GAP_LIST, "s--", color="#2a78d6", alpha=0.5, linewidth=1.5, label="논문 Gap")
ax2.tick_params(axis="y", labelcolor="#2a78d6")
ax2.set_ylim(0, 25)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(W_LABELS, rotation=45, ha="right", fontsize=9)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
fig.suptitle("Fig.6 — MLR, W1/W2 스윕 (penalty=50%)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
p6 = os.path.join(RESULTS_DIR, "fig6_paper_W1W2_MLR.png")
fig.savefig(p6, dpi=150); plt.close(fig)
print(f"  그림 저장: {p6}")


# =====================================================================
# ==================== Fig.8 — MLR penalty rate 스윕 (2분할) =============
# =====================================================================
fig8, (ax8n, ax8g) = plt.subplots(1, 2, figsize=(13, 5))
fig8.suptitle("Fig.8 — MLR, penalty rate 스윕 (W1=W2=1)", fontsize=13, fontweight="bold")

ax8n.plot(x_rate, mlr_rate_n, "o-", color="#eb6834", linewidth=2, markersize=5, label="nRMSE")
ax8n.set_xticks(x_rate); ax8n.set_xticklabels(lbl_rate, rotation=45)
ax8n.set_xlabel("벌금비용률"); ax8n.set_ylabel("nRMSE (%)")
ax8n.set_title("nRMSE"); ax8n.grid(True, alpha=0.3); ax8n.legend(fontsize=10)
ax8n.set_ylim(0, 80)

ax8g.plot(x_rate, mlr_rate_g, "o-", color="#2a78d6", linewidth=2, markersize=5, label="Gap")
if hl is not None:
    ax8g.axvline(hl, color="#898781", linestyle=":", linewidth=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")
ax8g.set_xticks(x_rate); ax8g.set_xticklabels(lbl_rate, rotation=45)
ax8g.set_xlabel("벌금비용률"); ax8g.set_ylabel("Optimality Gap (%)")
ax8g.set_title("Optimality Gap"); ax8g.grid(True, alpha=0.3); ax8g.legend(fontsize=10)
ax8g.set_ylim(0, 25)

fig8.tight_layout(rect=[0, 0, 1, 0.93])
p8 = os.path.join(RESULTS_DIR, "fig8_paper_rate_MLR.png")
fig8.savefig(p8, dpi=150); plt.close(fig8)
print(f"  그림 저장: {p8}")


# =====================================================================
# ==================== CSV 저장 ========================================
# =====================================================================
# AR W1/W2 스윕 CSV
ar_csv = os.path.join(RESULTS_DIR, "fig3_paper_AR.csv")
with open(ar_csv, "w") as f:
    f.write("Label,W1,W2,Code_nRMSE,Paper_nRMSE,Delta_nRMSE,Code_Gap,Paper_Gap,Delta_Gap\n")
    for i in range(N_SWEEP):
        dn = ar_sweep_nrmse[i] - PAPER_AR_NRMSE_LIST[i]
        dg = ar_sweep_gap[i] - PAPER_AR_GAP_LIST[i]
        f.write(f"{W_LABELS[i]},{W1_LIST[i]},{W2_LIST[i]},{ar_sweep_nrmse[i]:.2f},{PAPER_AR_NRMSE_LIST[i]:.2f},{dn:.2f},{ar_sweep_gap[i]:.2f},{PAPER_AR_GAP_LIST[i]:.2f},{dg:.2f}\n")
print(f"\n  CSV 저장: {ar_csv}")

# MLR W1/W2 스윕 CSV
mlr_csv = os.path.join(RESULTS_DIR, "fig6_paper_MLR.csv")
with open(mlr_csv, "w") as f:
    f.write("Label,W1,W2,Code_nRMSE,Paper_nRMSE,Delta_nRMSE,Code_Gap,Paper_Gap,Delta_Gap\n")
    for i in range(N_SWEEP):
        dn = mlr_sweep_nrmse[i] - PAPER_MLR_NRMSE_LIST[i]
        dg = mlr_sweep_gap[i] - PAPER_MLR_GAP_LIST[i]
        f.write(f"{W_LABELS[i]},{W1_LIST[i]},{W2_LIST[i]},{mlr_sweep_nrmse[i]:.2f},{PAPER_MLR_NRMSE_LIST[i]:.2f},{dn:.2f},{mlr_sweep_gap[i]:.2f},{PAPER_MLR_GAP_LIST[i]:.2f},{dg:.2f}\n")
print(f"  CSV 저장: {mlr_csv}")

# AR rate 스윕 CSV
ar_rate_csv = os.path.join(RESULTS_DIR, "fig5_paper_AR.csv")
with open(ar_rate_csv, "w") as f:
    f.write("Rate,Code_nRMSE,Code_Gap\n")
    for i in range(len(RATE_LIST)):
        f.write(f"{RATE_LIST[i]},{ar_rate_n[i]:.2f},{ar_rate_g[i]:.2f}\n")
print(f"  CSV 저장: {ar_rate_csv}")

# MLR rate 스윕 CSV
mlr_rate_csv = os.path.join(RESULTS_DIR, "fig8_paper_MLR.csv")
with open(mlr_rate_csv, "w") as f:
    f.write("Rate,Code_nRMSE,Code_Gap\n")
    for i in range(len(RATE_LIST)):
        f.write(f"{RATE_LIST[i]},{mlr_rate_n[i]:.2f},{mlr_rate_g[i]:.2f}\n")
print(f"  CSV 저장: {mlr_rate_csv}")

print("\n" + "=" * 100)
print("  완료")
print("=" * 100)
print(f"  그림: {p3}, {p5}, {p6}, {p8}")
print(f"  CSV : {ar_csv}, {mlr_csv}, {ar_rate_csv}, {mlr_rate_csv}")