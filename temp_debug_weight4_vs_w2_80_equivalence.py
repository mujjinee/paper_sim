# -*- coding: utf-8 -*-
# =====================================================================
# temp_debug_weight4_vs_w2_80_equivalence.py
#
# 검증: 방법2 Fine Tune의 (W1=1, W2=20, W2_BALANCE_WEIGHT=4)가
# (W1=1, W2=80, W2_BALANCE_WEIGHT=1)과 정말 100% 동일한 MILP인지.
# obj_x/sc/yc 계수와 bin_list(이진변수 판정)까지 전부 직접 비교한다.
# =====================================================================
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")
LOCAL_HOUR_START, LOCAL_HOUR_END, HOURS_PER_DAY = 9, 21, 12
TRAIN_START = pd.Timestamp("2013-08-25"); TRAIN_END = pd.Timestamp("2013-11-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")
KPI_RATE = 0.5

raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])
is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START
is_train = (daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])
train_dates = sorted(train_rows["local_date"].unique()); n_train_days = len(train_dates)
train_da = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt = np.zeros((n_train_days, HOURS_PER_DAY))
for _, row in train_rows.iterrows():
    d = (row["local_date"] - TRAIN_START).days; h = row["hour_idx"]
    train_da[d, h] = row["da_price"]; train_rt[d, h] = row["rt_price"]

hour = 0
da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
pc_h = rt_h + KPI_RATE * da_h  # PC = RT + rate*DA (방법2 Fine Tune)

def coeffs(W1, W2, weight):
    W2_eff = W2 * weight
    sc = -W1 * rt_h + W2_eff
    yc = W1 * pc_h + W2_eff
    obj_x = -W1 * da_h
    bin_list = [i for i in range(len(sc)) if sc[i] + yc[i] < 0]
    return obj_x, sc, yc, bin_list

obj_x_A, sc_A, yc_A, bin_A = coeffs(W1=1, W2=20, weight=4)   # 방법2 Fine Tune 그대로
obj_x_B, sc_B, yc_B, bin_B = coeffs(W1=1, W2=80, weight=1)   # W2=80 직접, weight 없음

print("=" * 70)
print("(W1=1, W2=20, Weight=4)  vs  (W1=1, W2=80, Weight=1) 계수 비교")
print("=" * 70)
print(f"W2_eff 값:  A(20*4)={20*4}   B(80*1)={80*1}   {'동일' if 20*4==80*1 else '다름'}")
print(f"max|obj_x_A - obj_x_B| = {np.max(np.abs(obj_x_A-obj_x_B)):.2e}")
print(f"max|sc_A - sc_B|       = {np.max(np.abs(sc_A-sc_B)):.2e}")
print(f"max|yc_A - yc_B|       = {np.max(np.abs(yc_A-yc_B)):.2e}")
print(f"bin_list 완전 동일?     = {bin_A == bin_B}  (개수: A={len(bin_A)}, B={len(bin_B)})")
print()
if (np.max(np.abs(obj_x_A-obj_x_B))==0 and np.max(np.abs(sc_A-sc_B))==0
        and np.max(np.abs(yc_A-yc_B))==0 and bin_A==bin_B):
    print(">>> 결론: 목적함수 계수·이진변수 판정 전부 비트 단위로 완전히 동일하다.")
    print("    즉 (W1=1,W2=20,Weight=4)와 (W1=1,W2=80,Weight=1)은 '비슷한 결과를 주는")
    print("    별개의 문제'가 아니라, 처음부터 같은 MILP다 — 풀 필요도 없이 계수 자체가 같다.")
else:
    print(">>> 결론: 차이가 있다 — 추가 조사 필요.")
