# -*- coding: utf-8 -*-
"""
AR(자기회귀) 모델만 떼어내서 성능(nRMSE)을 확인하는 스크립트. (쉬운 버전)

이 스크립트가 하는 일을 순서대로 말로 풀면:
  1. data/predictors15.csv 라는 원본 파일에서 태양광 발전량만 꺼낸다.
  2. 발전량 데이터를 "하루 12시간짜리 표"로 정리한다.
  3. 이 표를 앞쪽 300일(학습용)과 뒤쪽 100일(테스트용)로 자른다.
  4. 학습용 데이터로 "AR 모델"(=과거 발전량만 보고 미래를 예측하는 아주 단순한 모델)을 만든다.
  5. 테스트용 100일에 대해 하루씩 순서대로 예측해보고, 얼마나 틀렸는지(nRMSE)를 계산한다.
  6. 전체 데이터가 기니까, 300+100=400일짜리 구간을 여러 개 잘라서 각각 반복한다.

용어 설명:
  - zone : GEFCom2014 데이터에는 태양광 발전소가 3곳(zone 1,2,3) 있다.
  - nRMSE : 예측이 실제와 얼마나 다른지를 %로 나타낸 값. 작을수록 예측이 잘 맞은 것.
  - LAD(절대오차합) : 이 논문이 AR 모델을 학습할 때 쓴 방식. (제곱오차 대신 절댓값 오차를 최소화)

실행 방법:
    python ar_only_full_range.py
"""

import os                                            # 파일/폴더 경로를 다루기 위한 표준 라이브러리
import numpy as np                                   # 숫자 배열(행렬) 계산을 위한 라이브러리
import pandas as pd                                  # CSV 같은 표 형태 데이터를 다루기 위한 라이브러리
from sklearn.linear_model import QuantileRegressor   # "절대오차합"을 최소화하는 회귀모델 (논문이 쓴 방식)


# ============================================================
# 0. 설정값 모음 (여기 숫자만 바꾸면 동작이 바뀜)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 이 파이썬 파일이 들어있는 폴더 경로
PREDICTORS_FILE = os.path.join(BASE_DIR, "data", "predictors15.csv")   # 읽어올 원본 태양광 데이터 파일 경로

ZONE_ID_LIST = [1, 2, 3]     # 태양광 발전소 3곳의 zone 번호. 이 리스트에 있는 zone들을 전부 처리함
HOURS_PER_DAY = 12           # 하루 중 태양광이 나오는 시간대 개수 (UTC 0시~11시, 총 12개)
LAG_DAYS = 12                # AR 모델이 예측할 때 참고하는 "과거 며칠"의 개수 (논문: 12일)
TRAIN_DAYS = 300             # 학습(모델을 만드는 데 쓰는) 기간의 날짜 수 (논문과 동일)
TEST_DAYS = 100              # 테스트(성능을 확인하는 데 쓰는) 기간의 날짜 수 (논문과 동일)
WINDOW_DAYS = TRAIN_DAYS + TEST_DAYS   # 학습+테스트를 합친 한 구간의 길이 = 400일

PAPER_NRMSE = 34.76          # 논문 Table 3에 실려있는 AR 모델의 nRMSE(%). 우리 결과와 비교할 기준값


# ============================================================
# 1. 데이터 준비: predictors15.csv -> "날짜 x 시간대" 표
# ============================================================
def load_one_zone(zone_id):
    """
    predictors15.csv에서 하나의 zone(zone_id)에 해당하는 태양광 발전량만 꺼내서,
    "날짜별로 12시간치 발전량이 나란히 있는 표"로 정리해서 돌려준다.

    반환값:
        complete_dates : 사용 가능한 날짜들의 리스트 (오래된 날짜 -> 최신 날짜 순서)
        power_table    : (날짜 수, 12) 크기의 표.
                         power_table[i][h] = complete_dates[i]번째 날의 h번째 시간대 발전량
    """
    raw = pd.read_csv(PREDICTORS_FILE)                       # csv 파일 전체를 한 번에 읽어옴
    raw["TIMESTAMP"] = pd.to_datetime(                       # TIMESTAMP 열(문자열)을 "날짜+시간" 타입으로 변환
        raw["TIMESTAMP"], format="%Y%m%d %H:%M"              # 원본 형식이 "20120401 01:00" 같은 모양이라 형식을 알려줌
    )

    zone_rows = raw[raw["ZONEID"] == zone_id]                 # 원하는 zone_id에 해당하는 행만 남김
    zone_rows = zone_rows[zone_rows["TIMESTAMP"].dt.hour < HOURS_PER_DAY]   # 시(hour)가 0~11인 행만 남김 (낮 시간대)
    zone_rows = zone_rows.dropna(subset=["POWER"])            # 발전량(POWER)이 비어있는(NaN) 행은 버림

    # 아래 딕셔너리에 "날짜 -> {시간대: 발전량}" 형태로 하나씩 채워 넣을 것임
    power_by_date = {}                                        # 예: {2012-04-02: {0: 0.1, 1: 0.2, ...}, ...}

    for _, row in zone_rows.iterrows():                       # 살아남은 행들을 한 줄씩 순서대로 확인
        the_date = row["TIMESTAMP"].date()                    # 이 행의 "날짜" 부분만 뽑음 (시간은 버림)
        the_hour = row["TIMESTAMP"].hour                      # 이 행의 "시(hour)" 부분만 뽑음 (0~11 중 하나)
        the_power = row["POWER"]                               # 이 행의 발전량 값

        if the_date not in power_by_date:                     # 이 날짜를 처음 만났다면
            power_by_date[the_date] = {}                       # 그 날짜용으로 빈 딕셔너리를 새로 만들어줌
        power_by_date[the_date][the_hour] = the_power          # 그 날짜, 그 시간대에 발전량 값을 저장

    complete_dates = []                                        # 12시간이 전부 채워진 "완전한 날짜"만 담을 리스트
    for the_date, hour_to_power in power_by_date.items():      # 지금까지 모은 날짜들을 하나씩 확인
        if len(hour_to_power) == HOURS_PER_DAY:                # 그 날짜에 시간대가 정확히 12개 다 있으면
            complete_dates.append(the_date)                    # "완전한 날짜" 리스트에 추가

    complete_dates.sort()                                      # 날짜를 오래된 것 -> 최신 것 순서로 정렬

    day_count = len(complete_dates)                            # 사용할 수 있는 날짜의 총 개수
    power_table = np.zeros((day_count, HOURS_PER_DAY))         # 0으로 채워진 빈 표를 미리 만들어둠 (날짜 수 x 12)

    for day_index in range(day_count):                         # 0번째 날짜부터 마지막 날짜까지 순서대로
        the_date = complete_dates[day_index]                   # 이번에 채울 날짜
        for hour in range(HOURS_PER_DAY):                      # 시간대 0부터 11까지 순서대로
            power_table[day_index, hour] = power_by_date[the_date][hour]  # 표의 (날짜, 시간대) 칸을 채움

    return complete_dates, power_table                         # 날짜 리스트와 표를 함께 돌려줌


# ============================================================
# 2. AR 모델: "과거 12일의 같은 시간대"만 보고 다음날을 예측
# ============================================================
class ARModel:
    """
    하루 12개 시간대(0시~11시) 각각에 대해 "따로따로" 회귀 모델을 만든다.
    즉 이 클래스 하나가 실제로는 12개의 서로 다른 모델을 갖게 된다.

    h번째 시간대 모델의 논리:
        "과거 12일 동안 h시에 발전량이 얼마였는지" -> "내일 h시 발전량 예측"
    """

    def __init__(self):
        self.hour_models = {}      # {시간대 번호: 그 시간대 전용 모델} 을 저장해둘 딕셔너리 (처음엔 비어있음)

    def train(self, train_power_table):
        """
        train_power_table: (학습 날짜 수, 12) 크기의 표.
        이 표를 갖고 시간대별로 12개의 모델을 각각 학습시킨다.
        """
        total_train_days = train_power_table.shape[0]          # 학습 표에 들어있는 날짜 수

        for hour in range(HOURS_PER_DAY):                      # 시간대 0부터 11까지 하나씩 처리
            feature_list = []                                  # 이 시간대 모델의 "입력값"들을 모을 빈 리스트
            answer_list = []                                    # 이 시간대 모델의 "정답값"들을 모을 빈 리스트

            for day in range(LAG_DAYS, total_train_days):        # 과거 12일이 있어야 하니, 13번째 날부터 시작
                past_12_days = train_power_table[day - LAG_DAYS: day, hour]  # 이 날 기준 직전 12일치, 이 시간대 값들
                today_value = train_power_table[day, hour]        # 이 날, 이 시간대의 실제 발전량 (모델이 맞혀야 할 정답)

                feature_list.append(past_12_days)                # 입력값 목록에 이번 것 추가
                answer_list.append(today_value)                   # 정답값 목록에 이번 것 추가

            # QuantileRegressor(quantile=0.5)는 "절대오차합(LAD)"을 최소화하는 회귀모델이다 (논문이 쓴 손실함수)
            model_for_this_hour = QuantileRegressor(              # 이 시간대 전용 모델을 새로 만듦
                quantile=0.5,                                      # 0.5 = 중앙값 회귀 = 절대오차합 최소화와 동일한 효과
                alpha=0.0,                                         # 0 = 계수 크기에 대한 규제(패널티)를 걸지 않음
                solver="highs",                                    # 계산에 쓸 최적화 알고리즘 이름
            )
            model_for_this_hour.fit(                               # 실제로 학습을 실행 (입력값 -> 정답값 관계를 찾음)
                np.array(feature_list),                             # 입력값 리스트를 숫자 배열로 변환해서 전달
                np.array(answer_list),                               # 정답값 리스트도 숫자 배열로 변환해서 전달
            )

            self.hour_models[hour] = model_for_this_hour           # 학습이 끝난 모델을 딕셔너리에 저장해둠

    def predict_one_day(self, last_12_days_table):
        """
        last_12_days_table: (12, 12) 크기 표. "가장 최근 12일"의 발전량.
        이걸 이용해서 "다음 하루" 12개 시간대의 발전량을 예측해서 돌려준다.
        """
        tomorrow_forecast = np.zeros(HOURS_PER_DAY)                 # 예측 결과를 담을 빈 배열 (길이 12)

        for hour in range(HOURS_PER_DAY):                          # 시간대 0부터 11까지 하나씩 처리
            past_values = last_12_days_table[:, hour]                # 최근 12일 중, 이 시간대에 해당하는 값들만 뽑음
            past_values = past_values.reshape(1, -1)                  # 모델에 넣을 수 있는 모양(1행 x 12열)으로 바꿈

            model_for_this_hour = self.hour_models[hour]              # 이 시간대를 담당하는 모델을 꺼내옴
            raw_prediction = model_for_this_hour.predict(past_values)[0]  # 모델로 예측값 하나를 계산

            tomorrow_forecast[hour] = np.clip(raw_prediction, 0, 1)   # 발전량은 0~1 사이여야 하므로, 범위를 벗어나면 잘라냄

        return tomorrow_forecast                                     # 12개 시간대의 예측값을 돌려줌


# ============================================================
# 3. 평가지표: nRMSE 계산 (값이 작을수록 예측이 정확함)
# ============================================================
def calculate_nrmse(actual_values, predicted_values):
    """
    actual_values, predicted_values: 같은 길이의 1차원 배열 (실제값, 예측값)
    반환값: nRMSE (%), 작을수록 좋은 예측
    """
    error = actual_values - predicted_values         # 실제값과 예측값의 차이 (칸마다 오차)
    squared_error = error ** 2                        # 오차를 제곱함 (음수/양수 상관없이 크기만 보기 위해)
    mean_squared_error = np.mean(squared_error)        # 제곱한 오차들의 평균값
    rmse = np.sqrt(mean_squared_error)                 # 평균제곱오차에 제곱근을 씌움 = RMSE

    average_actual = np.mean(actual_values)              # 실제 발전량들의 평균값
    nrmse_percent = 100 * rmse / average_actual            # RMSE를 평균 발전량으로 나누고 100을 곱해 % 단위로 만듦

    return nrmse_percent                                # 최종 nRMSE(%) 값을 돌려줌


# ============================================================
# 4. 한 구간(400일)에 대해 학습 + 테스트 + 채점을 실행
# ============================================================
def run_ar_on_one_window(zone_label, day_list, power_table, window_start_index):
    """
    전체 날짜 중 window_start_index번째 날짜부터 400일(학습 300 + 테스트 100)을 잘라내서,
    그 구간으로 AR 모델을 학습시키고 테스트해서 nRMSE를 계산해준다.
    """
    window_end_index = window_start_index + WINDOW_DAYS              # 이 구간이 끝나는 위치 (시작 + 400)

    window_days = day_list[window_start_index:window_end_index]      # 이 구간에 해당하는 날짜들만 잘라냄
    window_power = power_table[window_start_index:window_end_index]  # 이 구간에 해당하는 발전량 표만 잘라냄

    train_power = window_power[:TRAIN_DAYS]                # 이 구간의 앞쪽 300일 = 학습용 발전량 표
    test_power = window_power[TRAIN_DAYS:]                  # 이 구간의 뒤쪽 100일 = 테스트용 발전량 표

    train_start_date = window_days[0]                       # 학습 구간의 첫째 날
    train_end_date = window_days[TRAIN_DAYS - 1]             # 학습 구간의 마지막 날
    test_start_date = window_days[TRAIN_DAYS]                # 테스트 구간의 첫째 날
    test_end_date = window_days[-1]                          # 테스트 구간의 마지막 날 (리스트의 맨 끝)

    print(f"[{zone_label}] 학습기간: {train_start_date} ~ {train_end_date} ({TRAIN_DAYS}일)")   # 학습 기간 출력
    print(f"[{zone_label}] 테스트기간: {test_start_date} ~ {test_end_date} ({TEST_DAYS}일)")     # 테스트 기간 출력

    ar_model = ARModel()               # AR 모델 객체를 새로 하나 만듦 (아직 학습 안 된 빈 상태)
    ar_model.train(train_power)         # 학습용 데이터로 12개 시간대 모델을 각각 학습시킴

    # ---- 테스트 100일을, 실제로 서비스하듯 "하루씩 순서대로" 예측한다 ----
    known_history = train_power.copy()          # "지금까지 알고 있는 발전량 기록". 처음엔 학습 데이터 전체가 기록임
    predicted_power = np.zeros_like(test_power)   # 예측 결과를 담을 빈 표. test_power와 똑같은 크기로 0으로 채움

    for test_day_index in range(TEST_DAYS):                    # 테스트 100일을 0번째부터 99번째까지 순서대로
        most_recent_12_days = known_history[-LAG_DAYS:]         # 지금까지 기록 중 "가장 최근 12일"만 꺼냄
        one_day_forecast = ar_model.predict_one_day(most_recent_12_days)  # 그 12일을 갖고 다음날 하루를 예측

        predicted_power[test_day_index] = one_day_forecast       # 예측 결과를 결과 표의 해당 날짜 위치에 저장

        actual_today = test_power[test_day_index]                 # 오늘(test_day_index번째 테스트 날)의 실제 발전량
        known_history = np.vstack([known_history, actual_today])   # 실제값을 기록에 추가함 (다음날 예측 때 쓰려고)

    # ---- 예측이 얼마나 정확했는지 채점 ----
    actual_flat = test_power.flatten()                # (100, 12) 표를 1200개짜리 한 줄로 펼침
    predicted_flat = predicted_power.flatten()          # 예측값 표도 똑같이 한 줄로 펼침
    nrmse_score = calculate_nrmse(actual_flat, predicted_flat)  # 두 값을 비교해서 nRMSE(%) 계산

    print(f"[{zone_label}] nRMSE = {nrmse_score:.2f}%")   # 이 구간의 채점 결과 출력

    return nrmse_score, test_start_date, test_end_date     # 채점 결과와 테스트 시작/끝 날짜를 함께 돌려줌


# ============================================================
# 5. 전체 실행: zone마다, 400일 구간마다 반복
# ============================================================
def main():
    all_results = []       # 모든 zone, 모든 구간의 결과를 하나씩 담아둘 빈 리스트

    for zone_id in ZONE_ID_LIST:                    # zone 1, 2, 3을 순서대로 하나씩 처리
        zone_label = f"z{zone_id:02d}"               # 화면에 보여줄 이름 (예: z01, z02, z03)

        day_list, power_table = load_one_zone(zone_id)    # 이 zone의 날짜 목록과 발전량 표를 읽어옴
        total_days = len(day_list)                          # 이 zone에서 쓸 수 있는 날짜의 총 개수

        print(f"\n[{zone_label}] 전체 사용 가능 날짜: {total_days}일 ({day_list[0]} ~ {day_list[-1]})")  # 요약 출력

        number_of_blocks = total_days // WINDOW_DAYS         # 400일짜리 구간을 몇 개나 만들 수 있는지 계산 (나머지 버림)

        for block_number in range(number_of_blocks):          # 0번째 구간부터 마지막 구간까지 순서대로
            window_start_index = block_number * WINDOW_DAYS    # 이번 구간이 시작하는 위치(인덱스) 계산

            nrmse_score, test_start, test_end = run_ar_on_one_window(   # 이 구간으로 학습+테스트+채점 실행
                zone_label, day_list, power_table, window_start_index
            )

            all_results.append({                    # 이번 구간의 결과를 딕셔너리로 만들어 리스트에 추가
                "zone": zone_label,                   # 어느 zone인지
                "block": block_number + 1,            # 몇 번째 구간인지 (사람이 보기 편하게 1부터 시작)
                "test_start": test_start,              # 테스트 시작 날짜
                "test_end": test_end,                  # 테스트 마지막 날짜
                "nrmse": nrmse_score,                  # 이 구간의 nRMSE(%) 결과
            })

        leftover_days = total_days - number_of_blocks * WINDOW_DAYS   # 400일 구간으로 못 나눈 나머지 날짜 수
        if leftover_days > 0:                                          # 나머지가 있다면
            print(f"[{zone_label}] 마지막 {leftover_days}일은 400일이 안 돼서 사용하지 않음")  # 안내 메시지 출력

    # ---- 모든 결과를 표로 정리해서 마지막에 한 번 더 보여줌 ----
    print("\n=== 전체 결과 요약 ===")                         # 요약 제목 출력
    for result in all_results:                                 # 저장해둔 결과를 하나씩 꺼내서
        gap_from_paper = result["nrmse"] - PAPER_NRMSE          # 논문 값(34.76%)과의 차이를 계산
        print(
            f"{result['zone']} 블록{result['block']} "                       # zone 이름과 몇 번째 구간인지
            f"(테스트 {result['test_start']}~{result['test_end']}): "         # 테스트 기간
            f"nRMSE={result['nrmse']:.2f}%  (논문 대비 {gap_from_paper:+.2f}%p)"  # nRMSE와 논문과의 차이
        )

    print(f"\n논문 AR nRMSE = {PAPER_NRMSE:.2f}%  (비교 기준값)")   # 마지막으로 논문 기준값을 한 번 더 보여줌


if __name__ == "__main__":     # 이 파일을 "직접" 실행했을 때만 아래 줄이 실행됨 (다른 파일이 import할 땐 실행 안 됨)
    main()                      # 실제 실행은 main() 함수 안에서 전부 이루어짐
