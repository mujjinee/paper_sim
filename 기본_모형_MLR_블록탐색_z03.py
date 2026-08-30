# -*- coding: utf-8 -*-

# =====================================================================
# 기본_모형_MLR_블록탐색.py
#
# 목적: 기본_모형_AR_블록탐색_4개월.py 와 짝을 이루는 MLR 버전. 논문
#       Figure 1이 "4개월" MISO 가격 데이터를 쓴다고 밝힌 것에 착안해,
#       merged_for_simulation_z03.csv 전체 구간을 120일 블록(학습 90일+
#       테스트 30일)으로 30일씩 겹치게 슬라이딩하면서 "기본 모형 MLR"
#       (pooled bounded LAD, 논문 Eq.6/8)의 nRMSE가 구간마다 어떻게
#       달라지는지, 그리고 논문(21.76%)에 가장 가까운 블록이 어디인지 찾는다.
#
#       MLR 은 AR과 달리 "직전 하루" 같은 이력(history)이 필요 없다 -
#       그냥 그 블록의 학습 구간 행(날짜x시간대)을 그대로 회귀에 쓰면 된다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone1)
# 낮 시간대: Sydney 현지시간 09:00~20:00 (local_hour 9~20)
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
from scipy import sparse                     # 희소행렬(linprog 제약식용) 라이브러리
from scipy.optimize import linprog           # 선형계획법(LP) 솔버 - LAD 회귀를 푸는 데 씀

BASE_DIR = os.getcwd()                                            # 현재 작업 디렉터리를 기준 경로로 사용
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")  # 읽어올 병합 데이터 파일 경로

LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_DAYS = 300                 # 학습 일수
TEST_DAYS = 100                  # 테스트 일수
WINDOW_DAYS = TRAIN_DAYS + TEST_DAYS   # 한 블록의 길이 = 400일
STEP_DAYS = 30                   # 다음 블록으로 넘어갈 때 며칠씩 옮길지 (120일보다 작아서 블록끼리 겹침)

PAPER_NRMSE = 21.76                # 논문 Table 4, 기본 모형 MLR 의 nRMSE(%) - 비교용


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)                       # csv 파일 전체를 한 번에 읽어옴
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])   # local_date 열을 날짜 타입으로 변환

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()              # 낮 시간대 행만 골라서 새 표로 복사
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START   # 0~11 로 다시 번호 매김


# =====================================================================
# 2. 완전한(12시간 다 있는) 날짜만 골라 정렬된 날짜 목록을 만듦
#    (실제 회귀 입력은 행 단위로 쓰므로, 여기서는 "날짜 목록"만 확보)
# =====================================================================
hour_counts = daylight_table.groupby("local_date")["hour_idx"].nunique()   # 날짜별로 시간대가 몇 개 있는지 셈
complete_dates = sorted(hour_counts[hour_counts == 12].index)                # 정확히 12개 있는 날짜만 남겨 정렬

daylight_table = daylight_table[daylight_table["local_date"].isin(complete_dates)]  # 완전한 날짜 행만 남김
daylight_table = daylight_table.sort_values(["local_date", "hour_idx"])              # 날짜, 시간대 순서로 정렬

n_total_days = len(complete_dates)                          # 전체 사용 가능한 날짜 수
print(f"전체 사용 가능 날짜: {n_total_days}일 ({complete_dates[0]} ~ {complete_dates[-1]})")


# =====================================================================
# 3. 120일 블록을 30일씩 겹치게 슬라이딩하면서, 블록마다 기본 모형 MLR 을
#    학습(pooled bounded LAD) + 예측 + 채점
# =====================================================================

n_blocks = (n_total_days - WINDOW_DAYS) // STEP_DAYS + 1     # 만들 수 있는 120일 블록 개수
print(f"400일 블록 개수(30일씩 슬라이딩): {n_blocks}\n")

block_results = []                                             # 블록마다 (테스트시작, 테스트끝, nRMSE) 를 담을 리스트

for block_number in range(n_blocks):                            # 블록 0번째부터 마지막 블록까지 순서대로

    window_start = block_number * STEP_DAYS                       # 이 블록이 시작하는 날짜 인덱스
    window_end = window_start + WINDOW_DAYS                        # 이 블록이 끝나는 날짜 인덱스 (미포함)

    window_dates = complete_dates[window_start:window_end]          # 이 블록에 해당하는 날짜들
    train_dates = set(window_dates[:TRAIN_DAYS])                      # 앞쪽 90일 = 학습용 날짜 집합
    test_dates = set(window_dates[TRAIN_DAYS:])                       # 뒤쪽 30일 = 테스트용 날짜 집합

    train_start_date = window_dates[0]                                # 학습 시작일
    test_start_date = window_dates[TRAIN_DAYS]                        # 테스트 시작일
    test_end_date = window_dates[-1]                                  # 테스트 마지막일

    train_rows = daylight_table[daylight_table["local_date"].isin(train_dates)]   # 이 블록의 학습 행
    test_rows = daylight_table[daylight_table["local_date"].isin(test_dates)].sort_values(["local_date", "hour_idx"])  # 테스트 행

    # ---- 3-1. 회귀 입력행렬(X) 만들기 - 절편 + dSSRD + dTSR + Hour ----
    n_train_obs = len(train_rows)                                # 학습 관측치 개수 (90*12=1080 이어야 정상)
    X_train = np.column_stack([
        np.ones(n_train_obs),                                       # 절편
        train_rows["dssrd"].to_numpy(),                              # dSSRD
        train_rows["dtsr"].to_numpy(),                                # dTSR
        train_rows["hour_idx"].to_numpy(),                             # Hour(0~11)
    ])
    y_train = train_rows["solar_power"].to_numpy()                  # 학습 정답값

    n_test_obs = len(test_rows)                                   # 테스트 관측치 개수 (30*12=360 이어야 정상)
    X_test = np.column_stack([
        np.ones(n_test_obs), test_rows["dssrd"].to_numpy(),
        test_rows["dtsr"].to_numpy(), test_rows["hour_idx"].to_numpy(),
    ])
    y_test = test_rows["solar_power"].to_numpy()                    # 테스트 정답값

    # ---- 3-2. 일반 LAD 회귀 하나를 풀어서 계수 4개를 구함 (제약 없이 절대오차만 최소화) ----
    n_features = 4                                                 # 절편, dSSRD, dTSR, Hour
    X_sparse = sparse.csr_matrix(X_train)                            # 입력행렬을 희소행렬로 변환
    identity_matrix = sparse.eye(n_train_obs, format="csr")            # 단위행렬 (절대값 처리용)

    constraint_block_1 = sparse.hstack([X_sparse, -identity_matrix])   #  X@beta - u <= y
    constraint_block_2 = sparse.hstack([-X_sparse, -identity_matrix])  # -X@beta - u <= -y

    all_constraints = sparse.vstack([
        constraint_block_1, constraint_block_2
    ], format="csr")

    constraint_limits = np.concatenate([
        y_train, -y_train,
    ])

    objective_coefficients = np.concatenate([
        np.zeros(n_features), np.ones(n_train_obs) / n_train_obs,
    ])

    variable_bounds = [(None, None)] * n_features + [(0.0, None)] * n_train_obs

    lp_result = linprog(
        objective_coefficients, A_ub=all_constraints, b_ub=constraint_limits,
        bounds=variable_bounds, method="highs",
    )
    mlr_coefficients = lp_result.x[:n_features]                    # 결과에서 beta(4개)만 뽑아 저장

    # ---- 3-3. 테스트 예측 + nRMSE 계산 ----
    test_forecast = np.zeros(n_test_obs)                            # 예측 결과를 담을 빈 배열
    for i in range(n_test_obs):                                        # 테스트 행을 하나씩 순서대로
        raw_prediction = np.dot(mlr_coefficients, X_test[i])              # 계수와 입력을 곱해서 더함
        clipped_prediction = min(max(raw_prediction, 0.0), 1.0)             # 예측값을 0~1 범위로 잘라냄
        test_forecast[i] = clipped_prediction                                # 예측 결과 저장

    sum_of_squared_error = 0.0                                       # 제곱오차 누적 변수
    for i in range(n_test_obs):                                        # 테스트 표본을 하나씩 순서대로
        error_i = y_test[i] - test_forecast[i]                            # 오차(실제-예측)
        sum_of_squared_error = sum_of_squared_error + error_i * error_i     # 제곱오차 누적

    mean_squared_error = sum_of_squared_error / n_test_obs             # 제곱오차 평균
    rmse_value = mean_squared_error ** 0.5                               # RMSE

    sum_of_actual = 0.0                                               # 실제값 합계 누적 변수
    for i in range(n_test_obs):                                          # 테스트 표본을 하나씩 순서대로
        sum_of_actual = sum_of_actual + y_test[i]                           # 실제값 누적

    average_actual = sum_of_actual / n_test_obs                        # 실제 발전량 평균
    nrmse_percent = 100.0 * rmse_value / average_actual                  # nRMSE(%)

    print(f"블록{block_number + 1:>2} 학습:{train_start_date}~ "
          f"테스트:{test_start_date}~{test_end_date}  nRMSE={nrmse_percent:.2f}%")

    block_results.append((block_number + 1, test_start_date, test_end_date, nrmse_percent))


# =====================================================================
# 4. 결과 요약: 전체 블록 표 + 최고/최저/논문과 가장 근사한 블록
# =====================================================================

print()
print("=== 기본 모형 MLR (bounded LAD) - 전체 구간 블록별 nRMSE ===")
print(f"{'블록':<6}{'테스트 시작':<14}{'테스트 끝':<14}{'nRMSE':>10}{'논문 대비':>12}")
for block_number, test_start_date, test_end_date, nrmse_percent in block_results:
    diff = nrmse_percent - PAPER_NRMSE
    print(f"{block_number:<6}{str(test_start_date):<14}{str(test_end_date):<14}{nrmse_percent:>9.2f}%{diff:>+11.2f}%p")

best_block = min(block_results, key=lambda row: row[3])                # nRMSE 값 자체가 가장 작은 블록
worst_block = max(block_results, key=lambda row: row[3])               # nRMSE 값 자체가 가장 큰 블록
closest_block = min(block_results, key=lambda row: abs(row[3] - PAPER_NRMSE))   # 논문과 가장 근사한 블록

print()
print(f"최고(최소) 블록: 블록{best_block[0]} (테스트 {best_block[1]}~{best_block[2]}) nRMSE={best_block[3]:.2f}%")
print(f"최저(최대) 블록: 블록{worst_block[0]} (테스트 {worst_block[1]}~{worst_block[2]}) nRMSE={worst_block[3]:.2f}%")
print(f"논문과 가장 근사한 블록: 블록{closest_block[0]} (테스트 {closest_block[1]}~{closest_block[2]}) "
      f"nRMSE={closest_block[3]:.2f}%  (|차이|={abs(closest_block[3]-PAPER_NRMSE):.2f}%p)")
print(f"논문 기본 모형 MLR: {PAPER_NRMSE:.2f}%")
