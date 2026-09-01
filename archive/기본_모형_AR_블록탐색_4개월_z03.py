# -*- coding: utf-8 -*-

# =====================================================================
# 기본_모형_AR_블록탐색_4개월.py
#
# 목적: 논문 5.1절에는 "훈련·테스트 300일/100일"이라는 서술과 "3개월(훈련)
#       +1개월(테스트)"이라는 서술이 함께 있다(약 13개월 vs 약 4개월로
#       3배 넘게 차이). 논문 Figure 1이 "MISO 가격 데이터 4개월"을 쓴다고
#       명시하는데, 이게 Table 3/4 실험과 같은 가격 데이터 소스일 가능성이
#       높아서, "3개월/1개월(=4개월)" 쪽이 논문의 실제 실험 규모였을
#       가능성을 검토해본다.
#
#       그래서 이번엔 400일(300+100) 대신 **120일(학습 90일 + 테스트
#       30일)** 블록으로, merged_for_simulation_z03.csv 전체 구간을
#       30일씩 겹치게 슬라이딩하며 "기본 모형 AR"(bounded LAD, 논문 Eq.3)
#       nRMSE가 어떻게 나오는지 확인한다. 나머지(Sydney 현지시간, 직전
#       하루 lag 설계, bounded LAD)는 기본_모형_AR_블록탐색.py(400일 버전)
#       와 완전히 동일하다.
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

HOURS_PER_DAY = 12               # 하루 낮 시간대 개수 (Sydney 현지시간 9시~20시)
LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_DAYS = 90                  # 학습 일수 (3개월 근사)
TEST_DAYS = 30                   # 테스트 일수 (1개월 근사)
WINDOW_DAYS = TRAIN_DAYS + TEST_DAYS   # 한 블록의 길이 = 120일 (=논문 Fig.1의 "4개월")
STEP_DAYS = 30                   # 다음 블록으로 넘어갈 때 며칠씩 옮길지 (120일보다 작아서 블록끼리 겹침)

PAPER_NRMSE = 34.76               # 논문 Table 3, 기본 모형 AR 의 nRMSE(%) - 비교용


# =====================================================================
# 1. 데이터 읽기 + Sydney 현지시간 낮 시간대만 남기기
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)                       # csv 파일 전체를 한 번에 읽어옴
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])   # local_date 열을 날짜 타입으로 변환

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
# ↑ local_hour 가 9시 이상, 21시 미만(=9~20시)인 행만 True 인 판단 열을 만듦

daylight_table = raw_table[is_daylight].copy()              # 낮 시간대 행만 골라서 새 표로 복사
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START
# ↑ local_hour(9~20) 를 0~11 로 다시 번호 매김 (hour_idx)


# =====================================================================
# 2. 완전한(12시간 다 있는) 날짜만 골라 (날짜 x 12시간) 발전량 배열로 바꿈
# =====================================================================
hour_counts = daylight_table.groupby("local_date")["hour_idx"].nunique()   # 날짜별로 시간대가 몇 개 있는지 셈
complete_dates = sorted(hour_counts[hour_counts == HOURS_PER_DAY].index)     # 정확히 12개 있는 날짜만 남겨 정렬

daylight_table = daylight_table[daylight_table["local_date"].isin(complete_dates)]  # 완전한 날짜 행만 남김
daylight_table = daylight_table.sort_values(["local_date", "hour_idx"])              # 날짜, 시간대 순서로 정렬

n_total_days = len(complete_dates)                          # 전체 사용 가능한 날짜 수
print(f"전체 사용 가능 날짜: {n_total_days}일 ({complete_dates[0]} ~ {complete_dates[-1]})")

full_solar = np.zeros((n_total_days, HOURS_PER_DAY))          # 전체 기간 발전량을 담을 빈 배열 (날짜수, 12)
row_counter = 0                                                  # daylight_table 을 순서대로 셀 카운터
for _, one_row in daylight_table.iterrows():                      # 정렬된 표를 한 줄씩 순서대로 확인
    day_position = row_counter // HOURS_PER_DAY                     # 이 행이 몇 번째 날짜인지
    hour_position = row_counter % HOURS_PER_DAY                     # 이 행이 몇 번째 시간대인지 (0~11)
    full_solar[day_position, hour_position] = one_row["solar_power"]  # 해당 칸에 발전량 값을 채워 넣음
    row_counter = row_counter + 1                                    # 카운터를 하나 증가시킴


# =====================================================================
# 3. 120일 블록을 30일씩 겹치게 슬라이딩하면서, 블록마다 기본 모형 AR 을
#    학습(bounded LAD, 논문 Eq.3 "직전 하루" 입력) + 예측 + 채점
# =====================================================================

n_blocks = (n_total_days - WINDOW_DAYS) // STEP_DAYS + 1     # 만들 수 있는 120일 블록 개수
print(f"120일 블록 개수(30일씩 슬라이딩): {n_blocks}\n")

block_results = []                                             # 블록마다 (테스트시작, 테스트끝, nRMSE) 를 담을 리스트

for block_number in range(n_blocks):                            # 블록 0번째부터 마지막 블록까지 순서대로

    window_start = block_number * STEP_DAYS                       # 이 블록이 시작하는 날짜 인덱스
    window_end = window_start + WINDOW_DAYS                        # 이 블록이 끝나는 날짜 인덱스 (미포함)

    window_dates = complete_dates[window_start:window_end]          # 이 블록에 해당하는 날짜들
    window_solar = full_solar[window_start:window_end]               # 이 블록에 해당하는 발전량 표 (400, 12)

    train_solar = window_solar[:TRAIN_DAYS]                          # 앞쪽 90일 = 학습용
    test_solar = window_solar[TRAIN_DAYS:]                            # 뒤쪽 30일 = 테스트용

    train_start_date = window_dates[0]                                # 학습 시작일
    test_start_date = window_dates[TRAIN_DAYS]                        # 테스트 시작일
    test_end_date = window_dates[-1]                                  # 테스트 마지막일

    # ---- 3-1. "직전 하루" 이력(history) 확보 ----
    if window_start > 0:                                              # 블록 시작 전에 날짜가 있으면(첫 블록이 아니면)
        history_day = full_solar[window_start - 1]                      # 블록 시작 바로 전날을 이력으로 씀
        ar_target_solar = train_solar                                    # 학습 목표값은 90일 전부 사용
    else:                                                                # 첫 블록(window_start=0)이면
        history_day = train_solar[0]                                     # 학습 첫날 자체를 이력으로 대신 씀
        ar_target_solar = train_solar[1:]                                # (그만큼 학습 목표는 299일만 사용)

    n_ar_rows = ar_target_solar.shape[0]                                # 이번 블록의 AR 학습 표본 개수 (299 또는 300)

    # ---- 3-2. AR 입력행렬(X) 만들기: 절편 + "직전 하루" 12시간(역순) ----
    history_and_train = np.vstack([history_day.reshape(1, -1), ar_target_solar])   # 이력 하루 + 학습일들을 이어붙임
    ar_intercept_column = np.ones((n_ar_rows, 1))                        # 절편용 1로만 채운 열
    ar_lag_features = np.zeros((n_ar_rows, HOURS_PER_DAY))                 # "직전 하루" 12시간 값을 담을 빈 배열
    for day_index in range(n_ar_rows):                                      # 학습 표본 0번째부터 마지막까지
        previous_day_values = history_and_train[day_index]                    # 이 학습일의 "바로 전날" 12시간 값
        ar_lag_features[day_index] = previous_day_values[::-1]                 # 시간을 거꾸로 뒤집어서 저장
    ar_design_matrix = np.hstack([ar_intercept_column, ar_lag_features])   # 절편 열 + lag 12개 열

    n_features = ar_design_matrix.shape[1]                              # 입력 변수 개수 (절편 포함 13개)
    coefficients_by_hour = np.zeros((HOURS_PER_DAY, n_features))          # 시간대별 계수를 저장할 빈 배열

    # ---- 3-3. 시간대(0~11)마다 bounded LAD 회귀를 풀어서 계수를 구함 ----
    for hour in range(HOURS_PER_DAY):                                     # 시간대 0부터 11까지 순서대로

        y_this_hour = ar_target_solar[:, hour]                              # 이 시간대의 정답값(실제 발전량)

        X_sparse = sparse.csr_matrix(ar_design_matrix)                        # 입력행렬을 희소행렬로 변환
        identity_matrix = sparse.eye(n_ar_rows, format="csr")                   # 단위행렬 (절대값 처리용)
        zero_block = sparse.csr_matrix((n_ar_rows, n_ar_rows))                   # 영행렬 (범위제약용)

        constraint_block_1 = sparse.hstack([X_sparse, -identity_matrix])   #  X@beta - u <= y
        constraint_block_2 = sparse.hstack([-X_sparse, -identity_matrix])  # -X@beta - u <= -y
        constraint_block_3 = sparse.hstack([X_sparse, zero_block])         #  X@beta <= 1 (예측값 상한)
        constraint_block_4 = sparse.hstack([-X_sparse, zero_block])        # -X@beta <= 0 (예측값 하한)

        all_constraints = sparse.vstack([                                # 4묶음을 하나의 제약행렬로 합침
            constraint_block_1, constraint_block_2, constraint_block_3, constraint_block_4
        ], format="csr")

        constraint_limits = np.concatenate([                             # 각 제약식의 우변 값들
            y_this_hour, -y_this_hour, np.ones(n_ar_rows), np.zeros(n_ar_rows),
        ])

        objective_coefficients = np.concatenate([                        # 목적함수 계수: beta=0, u=1/n
            np.zeros(n_features), np.ones(n_ar_rows) / n_ar_rows,
        ])

        variable_bounds = [(None, None)] * n_features + [(0.0, None)] * n_ar_rows   # beta 자유, u>=0

        lp_result = linprog(                                              # 선형계획법을 실제로 풂
            objective_coefficients, A_ub=all_constraints, b_ub=constraint_limits,
            bounds=variable_bounds, method="highs",
        )

        coefficients_by_hour[hour] = lp_result.x[:n_features]             # beta(13개)만 뽑아 이 시간대 계수로 저장

    # ---- 3-4. 테스트 30일을 하루씩 순서대로 예측 (rolling one-day-ahead) ----
    test_forecast = np.zeros((TEST_DAYS, HOURS_PER_DAY))                   # 예측 결과를 담을 빈 배열
    previous_day_actual = train_solar[-1]                                    # 테스트 첫날 예측에 쓸 "직전 하루"

    for day_index in range(TEST_DAYS):                                       # 테스트 0번째 날부터 99번째 날까지
        feature_vector = np.concatenate([[1.0], previous_day_actual[::-1]])    # [절편 1] + [직전 하루 역순]
        for hour in range(HOURS_PER_DAY):                                       # 이 날의 시간대 0~11을 하나씩 예측
            raw_prediction = np.dot(coefficients_by_hour[hour], feature_vector)    # 계수와 입력의 내적
            clipped_prediction = min(max(raw_prediction, 0.0), 1.0)                 # 0~1 범위로 잘라냄
            test_forecast[day_index, hour] = clipped_prediction                       # 예측 결과 저장
        previous_day_actual = test_solar[day_index]                              # "직전 하루"를 오늘의 실제값으로 갱신

    # ---- 3-5. nRMSE 계산 ----
    actual_flat = test_solar.flatten()                                     # 실제값을 1200개짜리 한 줄로 펼침
    predicted_flat = test_forecast.flatten()                                 # 예측값도 1200개짜리 한 줄로 펼침

    sum_of_squared_error = 0.0                                              # 제곱오차 누적 변수
    for i in range(len(actual_flat)):                                         # 1200개를 하나씩 순서대로
        error_i = actual_flat[i] - predicted_flat[i]                            # 오차(실제-예측)
        sum_of_squared_error = sum_of_squared_error + error_i * error_i           # 제곱오차 누적

    mean_squared_error = sum_of_squared_error / len(actual_flat)              # 제곱오차 평균
    rmse_value = mean_squared_error ** 0.5                                      # RMSE

    sum_of_actual = 0.0                                                       # 실제값 합계 누적 변수
    for i in range(len(actual_flat)):                                           # 1200개를 하나씩 순서대로
        sum_of_actual = sum_of_actual + actual_flat[i]                            # 실제값 누적

    average_actual = sum_of_actual / len(actual_flat)                          # 실제 발전량 평균
    nrmse_percent = 100.0 * rmse_value / average_actual                          # nRMSE(%)

    print(f"블록{block_number + 1:>2} 학습:{train_start_date.date()}~ "
          f"테스트:{test_start_date.date()}~{test_end_date.date()}  nRMSE={nrmse_percent:.2f}%")

    block_results.append((block_number + 1, test_start_date.date(), test_end_date.date(), nrmse_percent))


# =====================================================================
# 4. 결과 요약: 전체 블록 표 + 최고/최저 블록
# =====================================================================

print()
print("=== 기본 모형 AR (bounded LAD) - 전체 구간 블록별 nRMSE ===")
print(f"{'블록':<6}{'테스트 시작':<14}{'테스트 끝':<14}{'nRMSE':>10}{'논문 대비':>12}")
for block_number, test_start_date, test_end_date, nrmse_percent in block_results:
    diff = nrmse_percent - PAPER_NRMSE
    print(f"{block_number:<6}{str(test_start_date):<14}{str(test_end_date):<14}{nrmse_percent:>9.2f}%{diff:>+11.2f}%p")

best_block = min(block_results, key=lambda row: row[3])                # nRMSE 가 가장 작은(=값 자체가 가장 좋은) 블록
worst_block = max(block_results, key=lambda row: row[3])               # nRMSE 가 가장 큰(=가장 나쁜) 블록
closest_block = min(block_results, key=lambda row: abs(row[3] - PAPER_NRMSE))   # |nRMSE-논문| 이 가장 작은(=가장 근사한) 블록

print()
print(f"최고(최소) 블록: 블록{best_block[0]} (테스트 {best_block[1]}~{best_block[2]}) nRMSE={best_block[3]:.2f}%")
print(f"최저(최대) 블록: 블록{worst_block[0]} (테스트 {worst_block[1]}~{worst_block[2]}) nRMSE={worst_block[3]:.2f}%")
print(f"논문과 가장 근사한 블록: 블록{closest_block[0]} (테스트 {closest_block[1]}~{closest_block[2]}) "
      f"nRMSE={closest_block[3]:.2f}%  (|차이|={abs(closest_block[3]-PAPER_NRMSE):.2f}%p)")
print(f"논문 기본 모형 AR: {PAPER_NRMSE:.2f}%")
