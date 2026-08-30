# -*- coding: utf-8 -*-

# =====================================================================
# 기본_모형_MLR_z02_독자stepwise.py
#
# 목적: 논문이 보고한 최종 변수(SSRD, TSR, Hour) 를 그대로 재사용하는 대신,
#       논문 369행이 설명한 절차("먼저 모든 변수에 MLR을 적용해 p-값을 확인한
#       뒤, 가장 큰 p-값을 가진 변수를 제거하는 과정을 모든 p-값이 0.05
#       미만이 될 때까지 반복")를 **우리 Zone2 블록9 데이터로 직접 재현**해서,
#       우리 데이터 기준으로는 어떤 변수 조합이 최적인지, 그리고 그걸로
#       nRMSE가 얼마나 나오는지 확인한다.
#
# 후보 14개 변수 (논문과 동일한 구성): 기상 12개(dssrd/dtsr는 차분 적용된
# 버전 사용) + Month + Hour.
#
# 절차:
#   1) 후진 단계적 선택(backward stepwise)은 논문 설명 그대로 "p-값" 기준으로
#      하므로, 이 단계만 **일반 최소제곱(OLS)**으로 수행한다 (p-값은 OLS
#      계수의 t-검정에서 나오는 표준적인 통계량이라, LAD보다 OLS가 이 절차에
#      맞음 - 실제로 이런 "p-값/LogWorth 기준 후진제거"는 JMP 같은 통계
#      소프트웨어의 표준 stepwise 기능과 동일한 방식이다).
#   2) 최종적으로 남은 변수로 실제 예측 모델을 만들 때는, 지금까지와 동일하게
#      "bounded LAD"(학습 중 예측값 [0,1] 강제, 논문 Eq.8 손실함수)로 다시
#      학습해서 테스트 nRMSE만 확인한다 (gap은 안 봄).
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z02.csv (Zone2), 블록9 구간
# =====================================================================

import os                                    # 파일 경로를 다루는 표준 라이브러리
import numpy as np                           # 숫자 배열(행렬) 계산 라이브러리
import pandas as pd                          # 표(csv) 데이터를 다루는 라이브러리
import statsmodels.api as sm                 # OLS p-값 계산용 (후진 단계적 선택 전용)
from scipy import sparse                     # 희소행렬(linprog 제약식용) 라이브러리
from scipy.optimize import linprog           # 선형계획법(LP) 솔버 - 최종 LAD 회귀를 푸는 데 씀

BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # 이 파이썬 파일이 있는 폴더 경로
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z02.csv")  # 읽어올 병합 데이터 파일 경로 (Zone2)

LOCAL_HOUR_START = 9             # 낮 시간대 시작 시(local_hour 기준)
LOCAL_HOUR_END = 21              # 낮 시간대 끝(이 값 미만까지, 즉 9~20시)

TRAIN_START = pd.Timestamp("2012-11-28")   # 학습 시작일
TRAIN_END = pd.Timestamp("2013-09-23")     # 학습 마지막일 (300일째)
TEST_START = pd.Timestamp("2013-09-24")    # 테스트 시작일
TEST_END = pd.Timestamp("2014-01-01")      # 테스트 마지막일 (100일째)

P_VALUE_CUTOFF = 0.05             # 논문과 동일한 후진 제거 기준 (p >= 0.05 면 제거)

PAPER_NRMSE = 21.76                # 논문 Table 4, 기본 모형 MLR(3변수)의 nRMSE(%) - 비교용
BASELINE_NRMSE_Z02 = 32.32          # 논문이 보고한 3변수(SSRD,TSR,Hour) 그대로 썼을 때 z02 결과 - 비교용


# =====================================================================
# 1. 데이터 읽기 + 낮 시간대만 남기고, 후보 14개 변수 준비
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)                       # csv 파일 전체를 한 번에 읽어옴
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])   # local_date 열을 날짜 타입으로 변환

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()              # 낮 시간대 행만 골라서 새 표로 복사
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START   # 0~11 로 다시 번호 매김
daylight_table["month"] = daylight_table["local_date"].dt.month                 # 월(1~12) 열 추가 (논문의 Month 후보)

is_train_date = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train_date].copy()                    # 학습 구간 행만 골라냄

is_test_date = (daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test_date].copy()                      # 테스트 구간 행만 골라냄

print("학습 구간:", TRAIN_START.date(), "~", TRAIN_END.date(), "행 수:", len(train_rows))  # 3600 이어야 정상
print("테스트 구간:", TEST_START.date(), "~", TEST_END.date(), "행 수:", len(test_rows))    # 1200 이어야 정상

# 논문 Table 2 기상변수 12개(익명명은 이미 실제 명칭으로 바뀐 상태) + Month + Hour = 14개 후보
CANDIDATE_VARS = ["tclw", "tciw", "sp", "r", "tcc", "u10", "v10", "t2m",
                   "dssrd", "strd", "dtsr", "tp", "month", "hour_idx"]
# ↑ ssrd/tsr 대신, 앞서 확인한 대로 차분된 dssrd/dtsr 사용 (원시 누적값이 아님)

print("\n후보 변수(14개):", CANDIDATE_VARS)


# =====================================================================
# 2. 후진 단계적 선택(backward stepwise) - OLS p-값 기준, 논문 절차 그대로
# =====================================================================

remaining_vars = list(CANDIDATE_VARS)          # 아직 안 지워진 변수 목록 (처음엔 14개 전부)

print("\n=== 후진 단계적 선택 시작 (OLS, p >= 0.05 면 제거) ===")

while True:                                      # p-값이 전부 0.05 미만이 될 때까지 반복
    X_train_step = train_rows[remaining_vars].to_numpy()       # 현재 남은 변수들로 입력행렬 구성
    X_train_step = sm.add_constant(X_train_step)                # 절편(상수항) 열 추가
    y_train_step = train_rows["solar_power"].to_numpy()          # 정답값(실제 발전량)

    ols_model = sm.OLS(y_train_step, X_train_step)                # OLS 모델 정의
    ols_result = ols_model.fit()                                    # 학습 실행

    p_values = np.asarray(ols_result.pvalues)[1:]                  # 절편 제외, 각 변수의 p-값만 뽑음(numpy 배열로)
    worst_index = np.argmax(p_values)                                # p-값이 가장 큰(가장 안 유의한) 변수의 위치
    worst_p_value = p_values[worst_index]                             # 그 변수의 p-값
    worst_var_name = remaining_vars[worst_index]                       # 그 변수의 이름

    if worst_p_value < P_VALUE_CUTOFF:                              # 가장 나쁜 변수조차 0.05 미만이면
        print(f"모든 변수 p<0.05 -> 선택 종료. 남은 변수: {remaining_vars}")
        break                                                          # 반복 종료

    print(f"  제거: {worst_var_name} (p={worst_p_value:.4f})  -> 남은 변수 수: {len(remaining_vars) - 1}")
    remaining_vars.remove(worst_var_name)                            # 가장 안 유의한 변수를 목록에서 제거

    if len(remaining_vars) == 1:                                    # 변수가 1개까지 줄었으면 더 못 줄임
        print(f"변수 1개까지 줄어듦 -> 선택 종료. 남은 변수: {remaining_vars}")
        break

selected_vars = remaining_vars                    # 최종적으로 남은(선택된) 변수들
print(f"\n=== 우리 데이터 기준 최종 선택 변수: {selected_vars} ===")
print(f"(참고: 논문이 보고한 최종 변수는 ['ssrd', 'tsr', 'hour'] 였음)")


# =====================================================================
# 3. 선택된 변수로 최종 예측 모델 학습 - bounded LAD (이전과 동일 방식)
# =====================================================================

n_features = len(selected_vars) + 1                          # 절편 포함 변수 개수

X_train = train_rows[selected_vars].to_numpy()                  # 학습용 입력행렬 (절편 제외)
X_train = np.hstack([np.ones((len(X_train), 1)), X_train])        # 절편 열을 앞에 붙임
y_train = train_rows["solar_power"].to_numpy()                     # 학습용 정답값

X_test = test_rows[selected_vars].to_numpy()                     # 테스트용 입력행렬 (절편 제외)
X_test = np.hstack([np.ones((len(X_test), 1)), X_test])            # 절편 열을 앞에 붙임
y_test = test_rows.sort_values(["local_date", "hour_idx"])["solar_power"].to_numpy()  # 테스트 정답값

n_train_obs = X_train.shape[0]                                # 학습 표본 개수 (3600)

X_sparse = sparse.csr_matrix(X_train)                            # 입력행렬을 희소행렬로 변환
identity_matrix = sparse.eye(n_train_obs, format="csr")             # 단위행렬 (절대값 처리용)
zero_block = sparse.csr_matrix((n_train_obs, n_train_obs))           # 영행렬 (범위제약용)

constraint_block_1 = sparse.hstack([X_sparse, -identity_matrix])   #  X@beta - u <= y
constraint_block_2 = sparse.hstack([-X_sparse, -identity_matrix])  # -X@beta - u <= -y
constraint_block_3 = sparse.hstack([X_sparse, zero_block])         #  X@beta <= 1 (예측값 상한)
constraint_block_4 = sparse.hstack([-X_sparse, zero_block])        # -X@beta <= 0 (예측값 하한)

all_constraints = sparse.vstack([
    constraint_block_1, constraint_block_2, constraint_block_3, constraint_block_4
], format="csr")

constraint_limits = np.concatenate([
    y_train, -y_train, np.ones(n_train_obs), np.zeros(n_train_obs),
])

objective_coefficients = np.concatenate([
    np.zeros(n_features), np.ones(n_train_obs) / n_train_obs,
])

variable_bounds = [(None, None)] * n_features + [(0.0, None)] * n_train_obs

lp_result = linprog(
    objective_coefficients, A_ub=all_constraints, b_ub=constraint_limits,
    bounds=variable_bounds, method="highs",
)

mlr_coefficients = lp_result.x[:n_features]                      # 결과에서 beta(절편+선택변수 개수)만 뽑음
print(f"\n최종 bounded LAD 회귀 완료 (성공 여부: {lp_result.success})")
print(f"변수: ['intercept'] + {selected_vars}")
print(f"계수: {mlr_coefficients}")


# =====================================================================
# 4. 테스트 예측 + nRMSE 계산 (gap은 안 봄)
# =====================================================================

test_forecast = np.zeros(len(y_test))                            # 예측 결과를 담을 빈 배열
for i in range(len(y_test)):                                        # 테스트 행을 하나씩 순서대로
    raw_prediction = np.dot(mlr_coefficients, X_test[i])              # 계수와 입력의 내적
    clipped_prediction = min(max(raw_prediction, 0.0), 1.0)             # 0~1 범위로 잘라냄
    test_forecast[i] = clipped_prediction                                # 예측 결과 저장

sum_of_squared_error = 0.0                                         # 제곱오차 누적 변수
for i in range(len(y_test)):                                          # 테스트 표본을 하나씩 순서대로
    error_i = y_test[i] - test_forecast[i]                              # 오차(실제-예측)
    sum_of_squared_error = sum_of_squared_error + error_i * error_i       # 제곱오차 누적

mean_squared_error = sum_of_squared_error / len(y_test)             # 제곱오차 평균
rmse_value = mean_squared_error ** 0.5                                # RMSE

sum_of_actual = 0.0                                                 # 실제값 합계 누적 변수
for i in range(len(y_test)):                                          # 테스트 표본을 하나씩 순서대로
    sum_of_actual = sum_of_actual + y_test[i]                           # 실제값 누적

average_actual = sum_of_actual / len(y_test)                        # 실제 발전량 평균
nrmse_percent = 100.0 * rmse_value / average_actual                   # nRMSE(%)


# =====================================================================
# 5. 결과 출력
# =====================================================================

print()
print("=== 독자적 stepwise MLR (Zone2, 블록9) 결과 ===")
print(f"선택된 변수: {selected_vars}")
print(f"nRMSE = {nrmse_percent:.4f} %")
print()
print(f"{'구성':<45}{'nRMSE':>10}")
print(f"{'논문 (참고, 논문 자체 데이터)':<45}{PAPER_NRMSE:>9.2f}%")
print(f"{'논문 변수(ssrd,tsr,hour) 그대로, z02':<45}{BASELINE_NRMSE_Z02:>9.2f}%")
print(f"{'독자 stepwise 변수 ' + str(selected_vars):<45}{nrmse_percent:>9.2f}%")
