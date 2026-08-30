# -*- coding: utf-8 -*-

# =====================================================================
# z03_블록18_Fig1_3_4_5_비교분석.py
#
# 목적: 논문(Karimi & Kwon, 2022) Fig.1 / Fig.3 / Fig.4 / Fig.5 각각이
#       보여주는 "특성"을, 우리가 새로 찾은 z03/블록18(4개월, 학습
#       2013-08-25~2013-11-22 / 테스트 2013-11-23~2013-12-22) 데이터로
#       똑같이 재현해서 비교할 수 있는 원자료(CSV)를 만든다.
#
#       - Fig.1 재현: 4개월 구간 전체(24시간)의 DA/RT 가격 박스플롯 통계
#       - Fig.3 재현: 벌금비용률 50%일 때, W1/W2 비율별 AR vs 제안모형
#                     nRMSE·optimality gap
#       - Fig.4 재현: W1=W2=1, 벌금비용률 50%일 때, 테스트 구간 중
#                     7일 샘플의 실제/AR예측/제안모형예측 발전량
#       - Fig.5 재현: W1=W2=1일 때, 벌금비용률 0~100%별 AR vs 제안모형
#                     nRMSE·optimality gap
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일, 반복은 for
#             문으로만). 거의 모든 줄에 그 줄이 뭘 하는지 주석을 단다.
#
# 데이터: merged_for_simulation_z03.csv (Zone3)
# 구간: 학습 2013-08-25~2013-11-22(90일), 테스트 2013-11-23~2013-12-22(30일)
# =====================================================================

import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, Bounds, LinearConstraint, milp

BASE_DIR = os.getcwd()
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")

HOURS_PER_DAY = 12
LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21

TRAIN_START = pd.Timestamp("2013-08-25")
TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23")
TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0

OUT_DIR = os.path.join(BASE_DIR, "results", "fig_comparison_z03_block18")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# 1. 데이터 읽기 (전체 24시간 - Fig.1 재현용)
# =====================================================================
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

# ---- Fig.1 재현: 4개월 전체(학습+테스트) 구간, 24시간 전부 사용 ----
is_block_window = (raw_table["local_date"] >= TRAIN_START) & (raw_table["local_date"] <= TEST_END)
block_all_hours = raw_table[is_block_window].copy()   # 24시간 전부 포함한 4개월치 원본 행

fig1_rows = []   # 시간대별 박스플롯 통계(최소/1사분위/중앙값/3사분위/최대/평균)를 담을 리스트
for h in range(24):
    da_h = block_all_hours[block_all_hours["local_hour"] == h]["da_price"].to_numpy()
    rt_h = block_all_hours[block_all_hours["local_hour"] == h]["rt_price"].to_numpy()
    fig1_rows.append({
        "hour": h,
        "da_min": np.min(da_h), "da_q1": np.percentile(da_h, 25), "da_median": np.median(da_h),
        "da_q3": np.percentile(da_h, 75), "da_max": np.max(da_h), "da_mean": np.mean(da_h),
        "rt_min": np.min(rt_h), "rt_q1": np.percentile(rt_h, 25), "rt_median": np.median(rt_h),
        "rt_q3": np.percentile(rt_h, 75), "rt_max": np.max(rt_h), "rt_mean": np.mean(rt_h),
    })
fig1_table = pd.DataFrame(fig1_rows)
fig1_table.to_csv(os.path.join(OUT_DIR, "fig1_price_boxstats.csv"), index=False)
print("[Fig.1] 저장 완료:", os.path.join(OUT_DIR, "fig1_price_boxstats.csv"))
print(f"  DA 평균(전체 24h): {block_all_hours['da_price'].mean():.2f}  RT 평균: {block_all_hours['rt_price'].mean():.2f}")


# =====================================================================
# 2. 낮 시간대(9~20시)만 남기고, 학습/테스트/이력 구간을 (일 x 12) 배열로
#    바꾸기 - 기본_모형_AR / 논문_제안_모형_AR 과 동일한 절차
# =====================================================================
is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START

history_rows = daylight_table[daylight_table["local_date"] == HISTORY_DATE].copy().sort_values("hour_idx")
train_rows = daylight_table[(daylight_table["local_date"] >= TRAIN_START) & (daylight_table["local_date"] <= TRAIN_END)].copy()
train_rows = train_rows.sort_values(["local_date", "hour_idx"])
test_rows = daylight_table[(daylight_table["local_date"] >= TEST_START) & (daylight_table["local_date"] <= TEST_END)].copy()
test_rows = test_rows.sort_values(["local_date", "hour_idx"])

history_solar = np.zeros((1, HOURS_PER_DAY))
for i, (_, r) in enumerate(history_rows.iterrows()):
    history_solar[0, i % HOURS_PER_DAY] = r["solar_power"]

train_dates_sorted = sorted(train_rows["local_date"].unique())
n_train_days = len(train_dates_sorted)
train_solar = np.zeros((n_train_days, HOURS_PER_DAY))
train_da = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt = np.zeros((n_train_days, HOURS_PER_DAY))
for i, (_, r) in enumerate(train_rows.iterrows()):
    d, h = i // HOURS_PER_DAY, i % HOURS_PER_DAY
    train_solar[d, h] = r["solar_power"]; train_da[d, h] = r["da_price"]; train_rt[d, h] = r["rt_price"]

test_dates_sorted = sorted(test_rows["local_date"].unique())
n_test_days = len(test_dates_sorted)
test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt = np.zeros((n_test_days, HOURS_PER_DAY))
for i, (_, r) in enumerate(test_rows.iterrows()):
    d, h = i // HOURS_PER_DAY, i % HOURS_PER_DAY
    test_solar[d, h] = r["solar_power"]; test_da[d, h] = r["da_price"]; test_rt[d, h] = r["rt_price"]

print(f"\n학습일수={n_train_days}, 테스트일수={n_test_days}")

# ---- AR 입력행렬(공통, W1/W2/penalty 와 무관) ----
hist_and_train = np.vstack([history_solar, train_solar])
ar_X = np.hstack([np.ones((n_train_days, 1)), np.array([hist_and_train[d][::-1] for d in range(n_train_days)])])
n_features = ar_X.shape[1]


# =====================================================================
# 3. AR(기본 모형) 학습 - bounded LAD, 벌금율/W1/W2 와 무관하므로 한 번만
#    (기본_모형_AR.py 와 완전히 동일한 절차)
# =====================================================================
ar_coef_by_hour = np.zeros((HOURS_PER_DAY, n_features))
for hour in range(HOURS_PER_DAY):
    y_h = train_solar[:, hour]
    n = n_train_days
    Xs = sparse.csr_matrix(ar_X); I = sparse.eye(n, format="csr"); Z = sparse.csr_matrix((n, n))
    A = sparse.vstack([
        sparse.hstack([Xs, -I]), sparse.hstack([-Xs, -I]),
        sparse.hstack([Xs, Z]), sparse.hstack([-Xs, Z]),
    ], format="csr")
    b = np.concatenate([y_h, -y_h, np.ones(n), np.zeros(n)])
    c = np.concatenate([np.zeros(n_features), np.ones(n) / n])
    bounds = [(None, None)] * n_features + [(0.0, None)] * n
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    ar_coef_by_hour[hour] = res.x[:n_features]

ar_test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))
prev_day = train_solar[-1]
for d in range(n_test_days):
    feat = np.concatenate([[1.0], prev_day[::-1]])
    for hour in range(HOURS_PER_DAY):
        raw = np.dot(ar_coef_by_hour[hour], feat)
        ar_test_forecast[d, hour] = min(max(raw, 0.0), 1.0)
    prev_day = test_solar[d]

ar_actual_flat = test_solar.flatten()
ar_pred_flat = ar_test_forecast.flatten()
ar_rmse = np.sqrt(np.mean((ar_actual_flat - ar_pred_flat) ** 2))
ar_nrmse = 100.0 * ar_rmse / np.mean(ar_actual_flat)
print(f"[AR 기본모형] nRMSE={ar_nrmse:.4f}% (penalty/W1/W2 와 무관, 고정)")


# =====================================================================
# 4. 제안모형 MILP 하나를 통째로 도는 "블록" - 여러 (W1,W2,penalty) 조합에
#    재사용하기 위해, 반복되는 절차를 순서대로 다시 쓴다 (def 금지 규칙 때문에
#    루프 안에서 매번 그대로 반복 작성함)
# =====================================================================

def_note = None  # (실제 함수 아님 - 위에서부터 죽 이어지는 설명용 이름일 뿐)

# ---- 실험 조합 리스트 만들기 (Fig.3 용: 벌금율=50% 고정, W1/W2 비율 스윕) ----
fig3_ratio_labels = ["1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]
fig3_w1_list = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 5.0, 10.0, 20.0, 1.0]
fig3_w2_list = [20.0, 10.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

fig3_results = []   # 결과를 담을 리스트 (AR 포함)
fig3_results.append({"config": "AR", "w1": None, "w2": None, "penalty_rate": 0.5,
                      "nrmse": ar_nrmse, "gap": None})   # AR gap 은 아래에서 벌금율 0.5로 채움

for cfg_i in range(len(fig3_ratio_labels)):
    W1 = fig3_w1_list[cfg_i]
    W2 = fig3_w2_list[cfg_i]
    PENALTY_RATE = 0.5

    coef_by_hour = np.zeros((HOURS_PER_DAY, n_features))
    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
        n = n_train_days

        oracle_train = np.zeros(n)
        for i in range(n):
            a, d, r = y_h[i], da_h[i], rt_h[i]
            pc = PENALTY_RATE * d
            p0 = CAPACITY_MW * DURATION_HOURS * (r * a)
            pS = CAPACITY_MW * DURATION_HOURS * (d * a)
            surp1 = max(a - 1.0, 0.0); short1 = max(1.0 - a, 0.0)
            p1 = CAPACITY_MW * DURATION_HOURS * (d * 1.0 + r * surp1 - pc * short1)
            oracle_train[i] = max(p0, pS, p1)
        denom = oracle_train.sum()
        scale = CAPACITY_MW * DURATION_HOURS

        surplus_cost = -W1 * scale * rt_h / denom + W2 / n
        shortage_cost = W1 * scale * (PENALTY_RATE * da_h) / denom + W2 / n
        binary_rows = np.flatnonzero(surplus_cost + shortage_cost < 0.0)
        n_binary = len(binary_rows)

        beta_s, x_s = 0, n_features
        yp_s, ym_s, z_s = n_features + n, n_features + 2 * n, n_features + 3 * n
        n_var = n_features + 3 * n + n_binary

        obj = np.zeros(n_var)
        obj[x_s:x_s + n] = -W1 * scale * da_h / denom
        obj[yp_s:yp_s + n] = surplus_cost
        obj[ym_s:ym_s + n] = shortage_cost

        Xs = sparse.csr_matrix(ar_X); I = sparse.eye(n, format="csr")
        eq_a = sparse.lil_matrix((n, n_var)); eq_a[:, beta_s:beta_s + n_features] = -Xs; eq_a[:, x_s:x_s + n] = I
        eq_b = sparse.lil_matrix((n, n_var)); eq_b[:, x_s:x_s + n] = I; eq_b[:, yp_s:yp_s + n] = I; eq_b[:, ym_s:ym_s + n] = -I
        eqs = sparse.vstack([eq_a, eq_b], format="csr")
        rhs = np.concatenate([np.zeros(n), y_h])
        cons = [LinearConstraint(eqs, rhs, rhs)]

        if n_binary:
            comp = sparse.lil_matrix((2 * n_binary, n_var))
            for k in range(n_binary):
                row = binary_rows[k]
                comp[k, yp_s + row] = 1.0; comp[k, z_s + k] = 1.0
                comp[n_binary + k, ym_s + row] = 1.0; comp[n_binary + k, z_s + k] = -1.0
            up = np.concatenate([np.ones(n_binary), np.zeros(n_binary)])
            lo = np.full(2 * n_binary, -np.inf)
            cons.append(LinearConstraint(comp.tocsr(), lo, up))

        lb = np.concatenate([np.full(n_features, -np.inf), np.zeros(3 * n + n_binary)])
        ub = np.concatenate([np.full(n_features, np.inf), np.ones(3 * n + n_binary)])
        integ = np.zeros(n_var, dtype=int)
        for k in range(n_binary):
            integ[z_s + k] = 1

        res = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=cons, options={"mip_rel_gap": 1e-9})
        coef_by_hour[hour] = res.x[beta_s:beta_s + n_features]

    forecast = np.zeros((n_test_days, HOURS_PER_DAY))
    prev = train_solar[-1]
    for d in range(n_test_days):
        feat = np.concatenate([[1.0], prev[::-1]])
        for hour in range(HOURS_PER_DAY):
            forecast[d, hour] = min(max(np.dot(coef_by_hour[hour], feat), 0.0), 1.0)
        prev = test_solar[d]

    act = test_solar.flatten(); pred = forecast.flatten()
    nrmse = 100.0 * np.sqrt(np.mean((act - pred) ** 2)) / np.mean(act)

    da_f = test_da.flatten(); rt_f = test_rt.flatten()
    realized = 0.0; oracle = 0.0
    for i in range(len(act)):
        a, x, d, r = act[i], pred[i], da_f[i], rt_f[i]
        pc = PENALTY_RATE * d
        mis = a - x; surp = max(mis, 0.0); short = max(-mis, 0.0)
        realized += CAPACITY_MW * DURATION_HOURS * (d * x + r * surp - pc * short)
        oracle += CAPACITY_MW * DURATION_HOURS * max(r * a, d * a)
    gap = 100.0 * (oracle - realized) / oracle

    fig3_results.append({"config": fig3_ratio_labels[cfg_i], "w1": W1, "w2": W2,
                          "penalty_rate": PENALTY_RATE, "nrmse": nrmse, "gap": gap})
    print(f"[Fig.3] W1/W2={fig3_ratio_labels[cfg_i]:>5}  nRMSE={nrmse:.3f}%  gap={gap:.3f}%")

    if fig3_ratio_labels[cfg_i] == "1/1":
        fig4_coef_by_hour = coef_by_hour.copy()      # Fig.4 는 W1=W2=1 이므로 여기서 계수 재활용
        fig4_proposed_forecast = forecast.copy()

# ---- AR 의 gap 은 벌금율 50%로 별도 계산해서 채워 넣기 ----
da_f = test_da.flatten(); rt_f = test_rt.flatten(); act = ar_actual_flat; pred = ar_pred_flat
realized = 0.0; oracle = 0.0
for i in range(len(act)):
    a, x, d, r = act[i], pred[i], da_f[i], rt_f[i]
    pc = 0.5 * d
    mis = a - x; surp = max(mis, 0.0); short = max(-mis, 0.0)
    realized += CAPACITY_MW * DURATION_HOURS * (d * x + r * surp - pc * short)
    oracle += CAPACITY_MW * DURATION_HOURS * max(r * a, d * a)
ar_gap_50 = 100.0 * (oracle - realized) / oracle
fig3_results[0]["gap"] = ar_gap_50
print(f"[Fig.3] AR (참고선)                    nRMSE={ar_nrmse:.3f}%  gap={ar_gap_50:.3f}%")

fig3_table = pd.DataFrame(fig3_results)
fig3_table.to_csv(os.path.join(OUT_DIR, "fig3_w1w2_sweep.csv"), index=False)
print("[Fig.3] 저장 완료:", os.path.join(OUT_DIR, "fig3_w1w2_sweep.csv"))


# =====================================================================
# 5. Fig.4 재현: W1=W2=1, 벌금율 50% 제안모형(위에서 이미 학습됨) vs AR vs
#    실제, 테스트 30일 중 7일 샘플 뽑기 (3일 간격으로 7일 = 21일 범위,
#    논문처럼 태양광 피크가 뚜렷이 보이는 간격으로 선택)
# =====================================================================
sample_day_indices = [0, 4, 8, 12, 16, 20, 24]   # 30일 중 4일 간격으로 7일 선택
fig4_rows = []
for si, day_idx in enumerate(sample_day_indices):
    for hour in range(HOURS_PER_DAY):
        fig4_rows.append({
            "sample_day": si, "actual_test_day_index": day_idx,
            "test_date": str(test_dates_sorted[day_idx].date()),
            "hour_idx": hour,
            "actual": test_solar[day_idx, hour],
            "ar_forecast": ar_test_forecast[day_idx, hour],
            "proposed_forecast": fig4_proposed_forecast[day_idx, hour],
        })
fig4_table = pd.DataFrame(fig4_rows)
fig4_table.to_csv(os.path.join(OUT_DIR, "fig4_sample_days.csv"), index=False)
print("[Fig.4] 저장 완료:", os.path.join(OUT_DIR, "fig4_sample_days.csv"))


# =====================================================================
# 6. Fig.5 재현: W1=W2=1 고정, 벌금율 0~100%(11단계) 스윕
#    - AR: 계수 자체는 벌금율과 무관 (nRMSE 고정), gap 만 벌금율별로 재계산
#    - 제안모형: 벌금율이 학습 목적함수 자체에 들어가므로 매번 재학습
# =====================================================================
penalty_rate_list = [round(0.1 * k, 1) for k in range(11)]   # 0.0, 0.1, ..., 1.0
fig5_results = []

# ---- AR: gap 만 벌금율별로 재계산 ----
for pr in penalty_rate_list:
    da_f = test_da.flatten(); rt_f = test_rt.flatten(); act = ar_actual_flat; pred = ar_pred_flat
    realized = 0.0; oracle = 0.0
    for i in range(len(act)):
        a, x, d, r = act[i], pred[i], da_f[i], rt_f[i]
        pc = pr * d
        mis = a - x; surp = max(mis, 0.0); short = max(-mis, 0.0)
        realized += CAPACITY_MW * DURATION_HOURS * (d * x + r * surp - pc * short)
        oracle += CAPACITY_MW * DURATION_HOURS * max(r * a, d * a)
    gap = 100.0 * (oracle - realized) / oracle
    fig5_results.append({"model": "AR", "penalty_rate": pr, "nrmse": ar_nrmse, "gap": gap})
    print(f"[Fig.5-AR] penalty={pr:.0%}  nRMSE={ar_nrmse:.3f}%  gap={gap:.3f}%")

# ---- 제안모형(W1=W2=1): 벌금율별로 재학습 ----
W1, W2 = 1.0, 1.0
for pr in penalty_rate_list:
    coef_by_hour = np.zeros((HOURS_PER_DAY, n_features))
    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]
        n = n_train_days

        oracle_train = np.zeros(n)
        for i in range(n):
            a, d, r = y_h[i], da_h[i], rt_h[i]
            pc = pr * d
            p0 = CAPACITY_MW * DURATION_HOURS * (r * a)
            pS = CAPACITY_MW * DURATION_HOURS * (d * a)
            surp1 = max(a - 1.0, 0.0); short1 = max(1.0 - a, 0.0)
            p1 = CAPACITY_MW * DURATION_HOURS * (d * 1.0 + r * surp1 - pc * short1)
            oracle_train[i] = max(p0, pS, p1)
        denom = oracle_train.sum()
        scale = CAPACITY_MW * DURATION_HOURS

        surplus_cost = -W1 * scale * rt_h / denom + W2 / n
        shortage_cost = W1 * scale * (pr * da_h) / denom + W2 / n
        binary_rows = np.flatnonzero(surplus_cost + shortage_cost < 0.0)
        n_binary = len(binary_rows)

        beta_s, x_s = 0, n_features
        yp_s, ym_s, z_s = n_features + n, n_features + 2 * n, n_features + 3 * n
        n_var = n_features + 3 * n + n_binary

        obj = np.zeros(n_var)
        obj[x_s:x_s + n] = -W1 * scale * da_h / denom
        obj[yp_s:yp_s + n] = surplus_cost
        obj[ym_s:ym_s + n] = shortage_cost

        Xs = sparse.csr_matrix(ar_X); I = sparse.eye(n, format="csr")
        eq_a = sparse.lil_matrix((n, n_var)); eq_a[:, beta_s:beta_s + n_features] = -Xs; eq_a[:, x_s:x_s + n] = I
        eq_b = sparse.lil_matrix((n, n_var)); eq_b[:, x_s:x_s + n] = I; eq_b[:, yp_s:yp_s + n] = I; eq_b[:, ym_s:ym_s + n] = -I
        eqs = sparse.vstack([eq_a, eq_b], format="csr")
        rhs = np.concatenate([np.zeros(n), y_h])
        cons = [LinearConstraint(eqs, rhs, rhs)]

        if n_binary:
            comp = sparse.lil_matrix((2 * n_binary, n_var))
            for k in range(n_binary):
                row = binary_rows[k]
                comp[k, yp_s + row] = 1.0; comp[k, z_s + k] = 1.0
                comp[n_binary + k, ym_s + row] = 1.0; comp[n_binary + k, z_s + k] = -1.0
            up = np.concatenate([np.ones(n_binary), np.zeros(n_binary)])
            lo = np.full(2 * n_binary, -np.inf)
            cons.append(LinearConstraint(comp.tocsr(), lo, up))

        lb = np.concatenate([np.full(n_features, -np.inf), np.zeros(3 * n + n_binary)])
        ub = np.concatenate([np.full(n_features, np.inf), np.ones(3 * n + n_binary)])
        integ = np.zeros(n_var, dtype=int)
        for k in range(n_binary):
            integ[z_s + k] = 1

        res = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=cons, options={"mip_rel_gap": 1e-9})
        coef_by_hour[hour] = res.x[beta_s:beta_s + n_features]

    forecast = np.zeros((n_test_days, HOURS_PER_DAY))
    prev = train_solar[-1]
    for d in range(n_test_days):
        feat = np.concatenate([[1.0], prev[::-1]])
        for hour in range(HOURS_PER_DAY):
            forecast[d, hour] = min(max(np.dot(coef_by_hour[hour], feat), 0.0), 1.0)
        prev = test_solar[d]

    act = test_solar.flatten(); pred = forecast.flatten()
    nrmse = 100.0 * np.sqrt(np.mean((act - pred) ** 2)) / np.mean(act)

    da_f = test_da.flatten(); rt_f = test_rt.flatten()
    realized = 0.0; oracle = 0.0
    for i in range(len(act)):
        a, x, d, r = act[i], pred[i], da_f[i], rt_f[i]
        pc = pr * d
        mis = a - x; surp = max(mis, 0.0); short = max(-mis, 0.0)
        realized += CAPACITY_MW * DURATION_HOURS * (d * x + r * surp - pc * short)
        oracle += CAPACITY_MW * DURATION_HOURS * max(r * a, d * a)
    gap = 100.0 * (oracle - realized) / oracle

    fig5_results.append({"model": "Proposed", "penalty_rate": pr, "nrmse": nrmse, "gap": gap})
    print(f"[Fig.5-Proposed] penalty={pr:.0%}  nRMSE={nrmse:.3f}%  gap={gap:.3f}%")

fig5_table = pd.DataFrame(fig5_results)
fig5_table.to_csv(os.path.join(OUT_DIR, "fig5_penalty_sweep.csv"), index=False)
print("[Fig.5] 저장 완료:", os.path.join(OUT_DIR, "fig5_penalty_sweep.csv"))

print("\n=== 전체 완료 ===")
