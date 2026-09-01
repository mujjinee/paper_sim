# -*- coding: utf-8 -*-
"""
AR(자기회귀) 모델을 400일 블록(학습 300일 + 테스트 100일, 논문과 동일한 크기)으로
"겹치게(overlap)" 슬라이딩하면서 성능을 확인하는 스크립트.

ar_only_full_range.py 는 400일 블록을 "겹치지 않게" 2개만 만들었다
(블록1: 1~400일째, 블록2: 401~800일째).
이 스크립트는 STEP_DAYS(예: 30일)씩만 옮겨가며 400일 블록을 계속 잘라내므로,
블록끼리 날짜 구간이 서로 겹친다. 그 대신 전체 820일 구간을 훨씬 촘촘하게 훑어볼 수 있다.

실행 방법:
    python ar_only_400day_sliding.py
"""

import os                                            # 파일/폴더 경로를 다루기 위한 표준 라이브러리
import numpy as np                                   # 숫자 배열(행렬) 계산을 위한 라이브러리
import pandas as pd                                  # CSV 같은 표 형태 데이터를 다루기 위한 라이브러리
from sklearn.linear_model import QuantileRegressor   # "절대오차합"을 최소화하는 회귀모델 (논문이 쓴 방식)


# ============================================================
# 0. 설정값 모음
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 이 파이썬 파일이 들어있는 폴더 경로
PREDICTORS_FILE = os.path.join(BASE_DIR, "data", "predictors15.csv")   # 읽어올 원본 태양광 데이터 파일 경로

ZONE_ID_LIST = [1, 2, 3]     # 태양광 발전소 3곳의 zone 번호
HOURS_PER_DAY = 12           # 하루 중 태양광이 나오는 시간대 개수 (UTC 0시~11시, 총 12개)
LAG_DAYS = 12                # AR 모델이 예측할 때 참고하는 "과거 며칠"의 개수 (논문: 12일)
TRAIN_DAYS = 300             # 학습 기간의 날짜 수 (논문과 동일)
TEST_DAYS = 100              # 테스트 기간의 날짜 수 (논문과 동일)
WINDOW_DAYS = TRAIN_DAYS + TEST_DAYS   # 학습+테스트를 합친 한 구간의 길이 = 400일
STEP_DAYS = 30                # 다음 블록으로 넘어갈 때 며칠씩 옮길지 -> 400일보다 작으므로 블록끼리 겹치게 됨

PAPER_NRMSE = 34.76          # 논문 Table 3에 실려있는 AR 모델의 nRMSE(%). 비교 기준값


# ============================================================
# 1. 데이터 준비: predictors15.csv -> "날짜 x 시간대" 표
# ============================================================
def load_one_zone(zone_id):
    """
    predictors15.csv에서 하나의 zone(zone_id)에 해당하는 태양광 발전량만 꺼내서,
    "날짜별로 12시간치 발전량이 나란히 있는 표"로 정리해서 돌려준다.
    """
    raw = pd.read_csv(PREDICTORS_FILE)                       # csv 파일 전체를 한 번에 읽어옴
    raw["TIMESTAMP"] = pd.to_datetime(                       # TIMESTAMP 열(문자열)을 "날짜+시간" 타입으로 변환
        raw["TIMESTAMP"], format="%Y%m%d %H:%M"              # 원본 형식이 "20120401 01:00" 같은 모양이라 형식을 알려줌
    )

    zone_rows = raw[raw["ZONEID"] == zone_id]                 # 원하는 zone_id에 해당하는 행만 남김
    zone_rows = zone_rows[zone_rows["TIMESTAMP"].dt.hour < HOURS_PER_DAY]   # 시(hour)가 0~11인 행만 남김 (낮 시간대)
    zone_rows = zone_rows.dropna(subset=["POWER"])            # 발전량(POWER)이 비어있는(NaN) 행은 버림

    power_by_date = {}                                        # "날짜 -> {시간대: 발전량}" 을 채워 넣을 딕셔너리

    for _, row in zone_rows.iterrows():                       # 살아남은 행들을 한 줄씩 순서대로 확인
        the_date = row["TIMESTAMP"].date()                    # 이 행의 "날짜" 부분만 뽑음
        the_hour = row["TIMESTAMP"].hour                      # 이 행의 "시(hour)" 부분만 뽑음 (0~11 중 하나)
        the_power = row["POWER"]                               # 이 행의 발전량 값

        if the_date not in power_by_date:                     # 이 날짜를 처음 만났다면
            power_by_date[the_date] = {}                       # 빈 딕셔너리를 새로 만들어줌
        power_by_date[the_date][the_hour] = the_power          # 그 날짜, 그 시간대에 발전량 저장

    complete_dates = []                                        # 12시간이 전부 채워진 "완전한 날짜"만 담을 리스트
    for the_date, hour_to_power in power_by_date.items():      # 지금까지 모은 날짜들을 하나씩 확인
        if len(hour_to_power) == HOURS_PER_DAY:                # 그 날짜에 시간대가 정확히 12개 다 있으면
            complete_dates.append(the_date)                    # "완전한 날짜" 리스트에 추가

    complete_dates.sort()                                      # 날짜를 오래된 것 -> 최신 것 순서로 정렬

    day_count = len(complete_dates)                            # 사용할 수 있는 날짜의 총 개수
    power_table = np.zeros((day_count, HOURS_PER_DAY))         # 0으로 채워진 빈 표를 미리 만들어둠

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
    """

    def __init__(self):
        self.hour_models = {}      # {시간대 번호: 그 시간대 전용 모델}

    def train(self, train_power_table):
        total_train_days = train_power_table.shape[0]          # 학습 표에 들어있는 날짜 수

        for hour in range(HOURS_PER_DAY):                      # 시간대 0부터 11까지 하나씩 처리
            feature_list = []                                  # 입력값(과거 12일치)을 모을 빈 리스트
            answer_list = []                                    # 정답값(그날 발전량)을 모을 빈 리스트

            for day in range(LAG_DAYS, total_train_days):        # 과거 12일이 있어야 하니, 13번째 날부터 시작
                past_12_days = train_power_table[day - LAG_DAYS: day, hour]  # 직전 12일치, 이 시간대 값들
                today_value = train_power_table[day, hour]        # 이 날, 이 시간대의 실제 발전량

                feature_list.append(past_12_days)                # 입력값 목록에 추가
                answer_list.append(today_value)                   # 정답값 목록에 추가

            model_for_this_hour = QuantileRegressor(              # 절대오차합(LAD)을 최소화하는 회귀모델 생성
                quantile=0.5, alpha=0.0, solver="highs",
            )
            model_for_this_hour.fit(                               # 학습 실행
                np.array(feature_list), np.array(answer_list),
            )

            self.hour_models[hour] = model_for_this_hour           # 학습된 모델 저장

    def predict_one_day(self, last_12_days_table):
        tomorrow_forecast = np.zeros(HOURS_PER_DAY)                 # 예측 결과를 담을 빈 배열

        for hour in range(HOURS_PER_DAY):                          # 시간대 0부터 11까지 하나씩 처리
            past_values = last_12_days_table[:, hour].reshape(1, -1)  # 이 시간대의 최근 12일 값 (모델 입력 모양으로)
            model_for_this_hour = self.hour_models[hour]              # 이 시간대 담당 모델
            raw_prediction = model_for_this_hour.predict(past_values)[0]  # 예측 실행
            tomorrow_forecast[hour] = np.clip(raw_prediction, 0, 1)   # 0~1 범위로 잘라냄

        return tomorrow_forecast                                     # 12개 시간대 예측값 반환


# ============================================================
# 3. 평가지표: nRMSE
# ============================================================
def calculate_nrmse(actual_values, predicted_values):
    error = actual_values - predicted_values          # 실제값과 예측값의 차이
    rmse = np.sqrt(np.mean(error ** 2))                 # 오차 제곱의 평균 -> 제곱근 = RMSE
    average_actual = np.mean(actual_values)              # 실제 발전량들의 평균값
    return 100 * rmse / average_actual                   # RMSE를 평균으로 나눠 %로 표현


# ============================================================
# 4. 한 구간(400일)에 대해 학습 + 테스트 + 채점을 실행
# ============================================================
def run_ar_on_one_window(zone_label, day_list, power_table, window_start_index):
    window_end_index = window_start_index + WINDOW_DAYS              # 이 구간이 끝나는 위치

    window_days = day_list[window_start_index:window_end_index]      # 이 구간의 날짜들
    window_power = power_table[window_start_index:window_end_index]  # 이 구간의 발전량 표

    train_power = window_power[:TRAIN_DAYS]                # 앞쪽 300일 = 학습용
    test_power = window_power[TRAIN_DAYS:]                  # 뒤쪽 100일 = 테스트용

    test_start_date = window_days[TRAIN_DAYS]                # 테스트 시작 날짜
    test_end_date = window_days[-1]                          # 테스트 마지막 날짜

    ar_model = ARModel()               # AR 모델 생성
    ar_model.train(train_power)         # 학습 데이터로 12개 시간대 모델 학습

    known_history = train_power.copy()          # "지금까지 알고 있는" 발전량 기록 (처음엔 학습 데이터 전체)
    predicted_power = np.zeros_like(test_power)   # 예측 결과를 담을 빈 표

    for test_day_index in range(TEST_DAYS):                    # 테스트 100일을 하루씩 순서대로
        most_recent_12_days = known_history[-LAG_DAYS:]         # 최근 12일 꺼냄
        predicted_power[test_day_index] = ar_model.predict_one_day(most_recent_12_days)  # 다음날 예측

        actual_today = test_power[test_day_index]                 # 오늘의 실제 발전량
        known_history = np.vstack([known_history, actual_today])   # 기록에 실제값 추가

    nrmse_score = calculate_nrmse(test_power.flatten(), predicted_power.flatten())  # nRMSE 계산

    print(f"[{zone_label}] 테스트 {test_start_date}~{test_end_date}: nRMSE = {nrmse_score:.2f}%")

    return nrmse_score, test_start_date, test_end_date


# ============================================================
# 5. 전체 실행: zone마다, STEP_DAYS씩 옮겨가며 400일 블록을 반복 (겹침 허용)
# ============================================================
def main():
    all_results = []       # 모든 zone, 모든 블록의 결과를 담을 빈 리스트

    for zone_id in ZONE_ID_LIST:                    # zone 1, 2, 3을 순서대로 처리
        zone_label = f"z{zone_id:02d}"               # 화면에 보여줄 이름

        day_list, power_table = load_one_zone(zone_id)    # 이 zone의 날짜 목록과 발전량 표
        total_days = len(day_list)                          # 사용 가능한 총 날짜 수

        print(f"\n[{zone_label}] 전체 사용 가능 날짜: {total_days}일 ({day_list[0]} ~ {day_list[-1]})")

        # window_start_index를 0, STEP_DAYS, 2*STEP_DAYS, ... 로 옮겨가며 400일씩 잘라냄
        # (WINDOW_DAYS보다 STEP_DAYS가 작으므로 블록끼리 날짜가 겹친다)
        window_start_index = 0                                # 첫 블록은 0번째 날짜부터 시작
        block_number = 0                                       # 블록 번호 (0부터 시작, 출력할 땐 +1)
        while window_start_index + WINDOW_DAYS <= total_days:   # 400일을 다 채울 수 있는 동안 계속 반복
            block_number += 1                                   # 블록 번호 증가

            nrmse_score, test_start, test_end = run_ar_on_one_window(
                zone_label, day_list, power_table, window_start_index
            )

            all_results.append({                    # 이번 블록의 결과를 저장
                "zone": zone_label,
                "block": block_number,
                "test_start": test_start,
                "test_end": test_end,
                "nrmse": nrmse_score,
            })

            window_start_index += STEP_DAYS                     # 다음 블록은 STEP_DAYS만큼만 뒤로 이동 (겹침 발생)

    # ---- 결과 요약 출력 ----
    print(f"\n=== 전체 결과 요약 (400일 블록 = 300일 학습 + 100일 테스트, {STEP_DAYS}일씩 겹치게 이동) ===")
    for result in all_results:                                 # 결과를 하나씩 꺼내서
        gap_from_paper = result["nrmse"] - PAPER_NRMSE          # 논문 값과의 차이
        print(
            f"{result['zone']} 블록{result['block']:>2} "
            f"(테스트 {result['test_start']}~{result['test_end']}): "
            f"nRMSE={result['nrmse']:.2f}%  (논문 대비 {gap_from_paper:+.2f}%p)"
        )

    # ---- zone별 최고/최저 블록 출력 ----
    print("\n=== zone별 최고/최저 블록 ===")
    for zone_id in ZONE_ID_LIST:
        zone_label = f"z{zone_id:02d}"
        zone_results = [r for r in all_results if r["zone"] == zone_label]   # 이 zone의 결과만 골라냄
        best = min(zone_results, key=lambda r: r["nrmse"])                    # nRMSE가 가장 작은(=가장 좋은) 블록
        worst = max(zone_results, key=lambda r: r["nrmse"])                   # nRMSE가 가장 큰(=가장 나쁜) 블록
        print(f"{zone_label} 최고: 블록{best['block']} (테스트 {best['test_start']}~{best['test_end']}) "
              f"nRMSE={best['nrmse']:.2f}%")
        print(f"{zone_label} 최저: 블록{worst['block']} (테스트 {worst['test_start']}~{worst['test_end']}) "
              f"nRMSE={worst['nrmse']:.2f}%")

    print(f"\n논문 AR nRMSE = {PAPER_NRMSE:.2f}%  (비교 기준값)")


if __name__ == "__main__":     # 이 파일을 직접 실행했을 때만 아래 줄 실행
    main()
