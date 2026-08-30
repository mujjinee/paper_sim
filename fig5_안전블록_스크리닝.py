# -*- coding: utf-8 -*-

# =====================================================================
# fig5_안전블록_스크리닝.py
#
# 목적: "Fig.5(W1=W2=1 고정, 벌금율 스윕)가 논문처럼 완만한 곡선을
#       그리려면 어떤 블록을 써야 하나?"에 답하기 위해, 앞서 도출한
#       발산 조건식을 MILP 없이 산술만으로 전 블록에 스크리닝한다.
#
#       발산 조건(부족 구간에서 목적함수가 x=1까지 계속 줄어드는 조건):
#           W1/W2 > 2·denom_h / (n·scale·DA_i)
#       W1=W2=1 이므로, "안전 여유(margin_h)"를
#           margin_h = 2·denom_h / (n·scale·mean(DA_h)) - 1
#       로 정의한다. margin_h > 0 이면 그 시간대는 W1=W2=1에서 안전
#       (발산 없음), margin_h 가 클수록 여유가 크다. 12개 시간대 중
#       가장 나쁜(최소) margin 을 그 블록의 "안전도" 점수로 삼는다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
# =====================================================================

import os
import numpy as np
import pandas as pd

BASE_DIR = os.getcwd()

HOURS_PER_DAY = 12
LOCAL_HOUR_START, LOCAL_HOUR_END = 9, 21
TRAIN_DAYS, TEST_DAYS = 90, 30
WINDOW_DAYS = TRAIN_DAYS + TEST_DAYS
STEP_DAYS = 30
CAPACITY_MW, DURATION_HOURS, PENALTY_RATE = 30.0, 1.0, 0.5
W1, W2 = 1.0, 1.0   # Fig.5 의 고정점

zone_files = ["merged_for_simulation_z01.csv", "merged_for_simulation_z02.csv", "merged_for_simulation_z03.csv"]
zone_names = ["z01", "z02", "z03"]

all_block_results = []   # (zone, block_no, train_start, train_end, test_start, test_end, min_margin, worst_hour)

for zi in range(3):
    raw_table = pd.read_csv(os.path.join(BASE_DIR, zone_files[zi]))
    raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])
    is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
    daylight_table = raw_table[is_daylight].copy()
    daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START

    hour_counts = daylight_table.groupby("local_date")["hour_idx"].nunique()
    complete_dates = sorted(hour_counts[hour_counts == 12].index)
    daylight_table = daylight_table[daylight_table["local_date"].isin(complete_dates)]
    daylight_table = daylight_table.sort_values(["local_date", "hour_idx"])

    n_total_days = len(complete_dates)
    n_blocks = (n_total_days - WINDOW_DAYS) // STEP_DAYS + 1

    for block_number in range(n_blocks):
        window_start = block_number * STEP_DAYS
        window_end = window_start + WINDOW_DAYS
        window_dates = complete_dates[window_start:window_end]
        train_dates = set(window_dates[:TRAIN_DAYS])
        test_dates_list = window_dates[TRAIN_DAYS:]

        train_rows = daylight_table[daylight_table["local_date"].isin(train_dates)]
        n_train_days = len(train_dates)

        min_margin_this_block = None   # 이 블록의 12시간 중 가장 나쁜(최소) 안전여유
        worst_hour_this_block = None

        for hour in range(HOURS_PER_DAY):
            hour_rows = train_rows[train_rows["hour_idx"] == hour].sort_values("local_date")
            y = hour_rows["solar_power"].to_numpy()
            da = hour_rows["da_price"].to_numpy()
            rt = hour_rows["rt_price"].to_numpy()
            n = len(y)
            if n == 0:
                continue

            scale = CAPACITY_MW * DURATION_HOURS
            oracle = np.zeros(n)
            for i in range(n):
                a, d_, r_ = y[i], da[i], rt[i]
                pc = PENALTY_RATE * d_
                p0 = scale * (r_ * a)
                pS = scale * (d_ * a)
                surp1 = max(a - 1.0, 0.0); short1 = max(1.0 - a, 0.0)
                p1 = scale * (d_ * 1.0 + r_ * surp1 - pc * short1)
                oracle[i] = max(p0, pS, p1)
            denom_h = oracle.sum()

            mean_da = da.mean()
            # 안전 여유: margin_h = 2*denom_h/(n*scale*mean_da) - 1
            margin_h = (2.0 * denom_h) / (n * scale * mean_da) - 1.0

            if min_margin_this_block is None or margin_h < min_margin_this_block:
                min_margin_this_block = margin_h
                worst_hour_this_block = hour

        all_block_results.append({
            "zone": zone_names[zi], "block": block_number + 1,
            "train_start": str(window_dates[0].date()), "train_end": str(window_dates[TRAIN_DAYS-1].date()),
            "test_start": str(test_dates_list[0].date()), "test_end": str(test_dates_list[-1].date()),
            "min_margin": min_margin_this_block, "worst_hour": worst_hour_this_block,
        })
        print(f"[{zone_names[zi]}] 블록{block_number+1:>2}  테스트 {str(test_dates_list[0].date())}~{str(test_dates_list[-1].date())}  "
              f"최소안전여유={min_margin_this_block:+.3f} (최악시간대={worst_hour_this_block})")

result_table = pd.DataFrame(all_block_results)
result_table = result_table.sort_values("min_margin", ascending=False)
result_table.to_csv(os.path.join(BASE_DIR, "results", "fig5_안전블록_스크리닝.csv"), index=False)

print()
print("=== 안전 여유(min_margin) 상위 10개 블록 (클수록 W1=W2=1에서 안전) ===")
print(result_table.head(10).to_string(index=False))
print()
print("=== 안전 여유 하위 5개 블록 (가장 발산하기 쉬움, z03/블록18 위치 확인용) ===")
print(result_table.tail(5).to_string(index=False))
