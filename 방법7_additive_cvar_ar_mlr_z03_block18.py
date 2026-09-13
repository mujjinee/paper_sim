# -*- coding: utf-8 -*-
# =====================================================================
# 방법7_additive_cvar_ar_mlr_z03_block18.py
#
# 시도방법 7 — Additive CVaR 확장 (시도방법7.md 참고)
#
# 방법2(PC = RT + rate×DA, `방법2_4term_ar_mlr_z03_block18.py`)의 목적함수에
# additive CVaR 항을 그대로 더한 것이다. 원본 아이디어는
# `D:\03_JiWon\APEN_new\APEN_new\추가방법론_5_(4)_수정된_CVaR_시간별지표\
# proposed_ar_4term_additive_cvar_z03_block18.py`(Gurobi 버전)에 있고,
# 그 목적함수·오라클·CVaR 선형화 구조를 이 프로젝트의 scipy(HiGHS) MILP
# 방식(방법2)에 그대로 옮겨 통합했다.
#
#   기존(방법2) 목적함수:
#     min  W2·Σ|S_t - x_t|  +  W1·Σ regret_t
#   방법7(additive CVaR) 목적함수:
#     min  W2·Σ|S_t - x_t|  +  W1·Σ regret_t
#          + W1·λ·[ N·ζ + (1/(1-α))·Σ u_t ]   (u_t ≥ regret_t - ζ, u_t ≥ 0)
#
#   regret_t = oracle_t(raw, 3후보 {0,S_t,1}) - profit_t(x_t,y+_t,y-_t)
#   PC_t = RT_t + rate×DA_t (방법2와 동일)
#
# 기존 total regret 항(W1·Σregret)의 계수는 그대로 두고 CVaR 항을 "추가"만
# 하는 것이 시도방법7.md §13의 핵심 — λ가 커져도 total regret 항의 비중이
# 줄어들지 않는다(초기 가중평균 방식의 문제를 해결한 최종안).
#
# 평가 지표:
#   - nRMSE, Optimality Gap : 방법2와 동일한 compute_gap_4term() 그대로 재사용
#     (오라클 2후보 {commit=0, commit=actual} — 방법2와 직접 비교 가능하게 유지)
#   - Total regret / CVaR90 / Max regret / Mean regret : 시도방법7.md 정의대로
#     테스트셋에서 3후보 오라클({0,S_t,1})로 별도 계산 (compute_regret_diagnostics)
#
# 출력:
#   - λ=0(=방법2와 수학적으로 동일) vs λ=0.05(제안) 비교: Fig.3/5/6/8 (방법7_fig#_***)
#   - KPI 지점(W1=1,W2=20,rate=50%)에서 λ 스윕(0, 0.025, 0.05, 0.075, 0.10) 표
#     (시도방법7.md §14와 동일 실험) — nRMSE/Gap/Total regret/CVaR90/Max regret
#   - λ=0 등가성 검증(§16): CVaR 항을 붙였을 때 λ=0이면 방법2와 예측값이
#     거의 같아야 한다 — 최대 예측값 차이를 출력해 확인
# =====================================================================

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import csv

# =====================================================================
# 0. 한글 폰트 + matplotlib 스타일
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

C_GRID = "#DDDDDD"

# =====================================================================
# 1. 설정 — z03/블록18 (방법2와 완전히 동일)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "simulation_output")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21
HOURS_PER_DAY = 12

TRAIN_START = pd.Timestamp("2013-08-25")
TRAIN_END = pd.Timestamp("2013-11-22")
TEST_START = pd.Timestamp("2013-11-23")
TEST_END = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS

KPI_W1 = 1
KPI_W2 = 20
KPI_RATE = 0.5

FIG_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1),
                (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG_LABELS = ["AR/MLR", "1/20", "1/10", "1/5", "1/2",
              "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]
FIG_RATES = [round(0.1 * i, 1) for i in range(11)]
FIG58_W1 = 1
FIG58_W2 = 1

# 방법2와 동일한 raw+weight 보정 (오늘_GapRate비교.md §10~11, 방법2 §5.2.2 참고)
W2_BALANCE_WEIGHT = 4

# ── CVaR 설정 (시도방법7.md §9, §13, §14, §23~24) ──
CVAR_ALPHA = 0.90                 # CVaR 신뢰수준
# [재보정] λ=0.05는 원본(추가방법론_5_(4), W2_BALANCE_WEIGHT 보정 없음)의 최적값이다.
# 방법2의 W2_eff=W2*4 보정을 얹으면 CVaR 항의 상대적 영향력이 희석돼 λ=0.05는 거의
# 효과가 없다(시도방법7.md §23, temp_debug_cvar_weight_dilution.py로 확인). 가중치
# 보정이 있는 상태(이 스크립트)에서 λ를 0.05~1.5로 다시 스윕한 결과
# (temp_debug_cvar_lambda_resweep.py, results/simulation_output/temp_cvar_lambda_resweep_weighted.csv),
# λ=0.2가 "OG·Total regret·nRMSE는 거의 그대로 두고 Maximum regret만 크게 줄인다"는
# CVaR의 원래 취지(§5.6)에 가장 잘 맞았다 — AR 기준 ΔnRMSE=+0.22%p, ΔGap=-0.14%p,
# ΔTotalRegret=-0.94%(거의 유지) 이면서 ΔMaxRegret=-11.27%(꼬리 최대값만 대폭 감소).
# 더 큰 CVaR90 감소(원본 λ=0.05의 -10%대에 가까움)를 원하면 λ=0.4~0.5도 후보지만,
# 그 경우 ΔnRMSE가 +2~2.6%p로 커진다(시도방법7.md §24 표 참고) — 트레이드오프 성격이
# 다르므로 기본값은 "부작용 최소" 쪽인 0.2로 둔다.
CVAR_LAMBDA_MAIN = 0.2
LAMBDA_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # 재보정 스윕(시도방법7.md §24)

# [Fig.5/8 전용 재보정] λ=0.2를 그대로 Fig.5/8(W1=W2=1 고정, rate만 스윕)에 쓰면
# W2_eff=W2*4=4로 KPI 지점(W2_eff=80)보다 20배 작아서 CVaR가 훨씬 강하게 작동해
# nRMSE가 크게 무너진다(방법7 재실행 결과 확인 — 예: rate=0에서 nRMSE +36%p 폭등,
# 시도방법7.md §25). W1=W2=1, rate=0.5 기준점에서 λ를 0~0.1로 다시 스윕한 결과
# (temp_debug_cvar_lambda_resweep_fig58.py, .../temp_cvar_lambda_resweep_fig58.csv),
# 이 영역은 KPI 지점보다 CVaR 트레이드오프가 훨씬 불리했다 — ΔnRMSE를 KPI와 같은
# 기준(~1%p 이내)으로 제한하면 tail(MaxRegret) 개선이 사실상 0에 가깝다(λ=0.0075:
# ΔnRMSE+0.37%p·ΔMaxRegret 0%). 절충안으로 λ=0.01(ΔnRMSE+1.20%p·ΔMaxRegret -0.21%,
# 그나마 tail이 아주 조금이라도 움직이는 가장 작은 값)을 Fig.5/8 전용 값으로 쓴다 —
# 그래도 KPI 지점만큼 "부작용 없이 꼬리만 자르는" 효과는 없다는 걸 그대로 보여준다.
CVAR_LAMBDA_FIG58 = 0.01

# ── 논문 참고값 (방법2와 동일 — Table 3/4 + Fig.5/8 판독값) ──
PAPER_AR_NRMSE = 34.76
PAPER_AR_GAP = 15.04
PAPER_MLR_NRMSE = 21.76
PAPER_MLR_GAP = 12.59
PAPER_PROP_NRMSE = [34.89, 35.14, 36.28, 41.09, 44.95,
                    46.11, 48.27, 49.21, 49.61, 50.07]
PAPER_PROP_GAP = [13.91, 13.42, 12.71, 11.88, 11.44,
                  11.38, 11.38, 11.36, 11.36, 11.36]
PAPER_PROP_MLR_NRMSE = [21.92, 21.84, 21.75, 21.66, 22.01,
                        23.32, 27.75, 30.62, 35.76, 37.67]
PAPER_PROP_MLR_GAP = [11.91, 11.68, 11.15, 10.65, 10.28,
                      9.91, 9.51, 9.32, 9.27, 9.28]
FIG5_PAPER_AR_NRMSE = [34.76] * 11
FIG5_PAPER_AR_GAP = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]
FIG5_PAPER_PROP_NRMSE = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
FIG5_PAPER_PROP_GAP = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]
FIG8_PAPER_MLR_NRMSE = [21.76] * 11
FIG8_PAPER_PROP_NRMSE = [23.92, 22.56, 21.76, 21.41, 21.67, 22.01, 22.45, 23.14, 23.76, 24.98, 26.10]
FIG8_PAPER_MLR_GAP = [8.18, 9.05, 9.93, 10.83, 11.71, 12.60, 13.48, 14.37, 15.25, 16.13, 16.98]
FIG8_PAPER_PROP_GAP = [7.59, 8.53, 9.17, 9.56, 9.94, 10.28, 10.52, 10.68, 10.85, 10.96, 10.93]


# =====================================================================
# 2. 데이터 로딩 (방법2와 완전히 동일)
# =====================================================================
print("=" * 70)
print("  1. 데이터 로딩")
print("=" * 70)
raw_table = pd.read_csv(MERGED_FILE)
raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])

is_daylight = (raw_table["local_hour"] >= LOCAL_HOUR_START) & \
              (raw_table["local_hour"] < LOCAL_HOUR_END)
daylight_table = raw_table[is_daylight].copy()
daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START

is_history = daylight_table["local_date"] == HISTORY_DATE
history_rows = daylight_table[is_history].copy().sort_values("hour_idx")
history_solar = np.zeros((1, HOURS_PER_DAY))
for _, row in history_rows.iterrows():
    history_solar[0, row["hour_idx"]] = row["solar_power"]

is_train = (daylight_table["local_date"] >= TRAIN_START) & \
           (daylight_table["local_date"] <= TRAIN_END)
train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])

is_test = (daylight_table["local_date"] >= TEST_START) & \
          (daylight_table["local_date"] <= TEST_END)
test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])

train_dates = sorted(train_rows["local_date"].unique())
n_train_days = len(train_dates)
test_dates = sorted(test_rows["local_date"].unique())
n_test_days = len(test_dates)

train_solar = np.zeros((n_train_days, HOURS_PER_DAY))
train_da_price = np.zeros((n_train_days, HOURS_PER_DAY))
train_rt_price = np.zeros((n_train_days, HOURS_PER_DAY))
for _, row in train_rows.iterrows():
    d = (row["local_date"] - TRAIN_START).days
    h = row["hour_idx"]
    train_solar[d, h] = row["solar_power"]
    train_da_price[d, h] = row["da_price"]
    train_rt_price[d, h] = row["rt_price"]

test_solar = np.zeros((n_test_days, HOURS_PER_DAY))
test_da_price = np.zeros((n_test_days, HOURS_PER_DAY))
test_rt_price = np.zeros((n_test_days, HOURS_PER_DAY))
for _, row in test_rows.iterrows():
    d = (row["local_date"] - TEST_START).days
    h = row["hour_idx"]
    test_solar[d, h] = row["solar_power"]
    test_da_price[d, h] = row["da_price"]
    test_rt_price[d, h] = row["rt_price"]

n_train_obs = len(train_rows)
mlr_train_solar = train_rows["solar_power"].to_numpy()
mlr_train_dssrd = train_rows["dssrd"].to_numpy()
mlr_train_dtsr = train_rows["dtsr"].to_numpy()
mlr_train_hour = train_rows["hour_idx"].to_numpy(dtype=float)
mlr_train_da = train_rows["da_price"].to_numpy()
mlr_train_rt = train_rows["rt_price"].to_numpy()

n_test_obs = len(test_rows)
actual_flat = test_rows["solar_power"].to_numpy()
da_flat = test_rows["da_price"].to_numpy()
rt_flat = test_rows["rt_price"].to_numpy()

n_features_mlr = 4
X_mlr_train = np.column_stack([np.ones(n_train_obs),
                                mlr_train_dssrd, mlr_train_dtsr, mlr_train_hour])
X_mlr_test = np.column_stack([np.ones(n_test_obs),
                               test_rows["dssrd"].to_numpy(),
                               test_rows["dtsr"].to_numpy(),
                               test_rows["hour_idx"].to_numpy(dtype=float)])

n_features_ar = 13
history_and_train = np.vstack([history_solar, train_solar])
n_ar_rows = n_train_days
ar_intercept = np.ones((n_ar_rows, 1))
ar_lag = np.zeros((n_ar_rows, HOURS_PER_DAY))
for d in range(n_ar_rows):
    ar_lag[d] = history_and_train[d][::-1]
ar_design = np.hstack([ar_intercept, ar_lag])

actual_flat_all = test_solar.flatten()
da_flat_all = test_da_price.flatten()
rt_flat_all = test_rt_price.flatten()
n_test_obs_all = len(actual_flat_all)

print(f"  train: {n_train_days}일({n_train_obs}행), test: {n_test_days}일({n_test_obs}행)")


# =====================================================================
# 3. 평가 함수 — nRMSE / Gap(방법2와 동일, 오라클 2후보) / CVaR 진단지표
# =====================================================================
def compute_gap_4term(pred_flat, penalty_rate, actual, da, rt):
    """방법2와 완전히 동일한 Gap 정의 (오라클 2후보 {commit=0, commit=actual}) —
    방법7과 방법2를 직접 비교하려면 이 지표 정의가 같아야 한다."""
    sum_realized = 0.0
    sum_oracle = 0.0
    for i in range(len(pred_flat)):
        a = actual[i]; x = pred_flat[i]
        dp = da[i]; rp = rt[i]
        pc = rp + penalty_rate * dp
        mismatch = a - x
        surplus = max(mismatch, 0)
        shortage = max(-mismatch, 0)
        realized = scale * (dp * x + rp * surplus - pc * shortage)
        sum_realized += realized
        p0 = scale * rp * a
        pa = scale * dp * a
        oracle = max(p0, pa)
        sum_oracle += oracle
    return 100.0 * (sum_oracle - sum_realized) / sum_oracle if sum_oracle > 1e-10 else 0.0


def nrmse(pred_flat, actual):
    rmse = np.sqrt(np.mean((actual - pred_flat) ** 2))
    return 100.0 * rmse / np.mean(actual)


def auto_ylim(*series, pad_frac=0.12, round_to=5, floor_at_zero=True):
    vals = []
    for s in series:
        vals.extend([v for v in s if v is not None])
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = span * pad_frac if span > 0 else max(hi * 0.1, round_to)
    lo -= pad; hi += pad
    lo = round_to * np.floor(lo / round_to)
    hi = round_to * np.ceil(hi / round_to)
    if floor_at_zero:
        lo = max(lo, 0)
    return float(lo), float(hi)


def oracle_3cand_raw(y, da, rt, pc):
    """학습용 raw(unscaled) 3후보 오라클 {commit=0, commit=S, commit=1} —
    시도방법7.md §5.1 / 추가방법론_5_(4) 참고코드 §5-1~5-3과 동일 정의.
    CVaR 항의 regret_t = oracle_t - profit_t 계산에 쓰는 상수다."""
    p0 = rt * y
    pa = da * y
    surplus_full = np.maximum(y - 1.0, 0.0)
    shortage_full = np.maximum(1.0 - y, 0.0)
    p1 = da * 1.0 + rt * surplus_full - pc * shortage_full
    return np.maximum.reduce([p0, pa, p1])


def compute_regret_diagnostics(pred_flat, penalty_rate, actual, da, rt):
    """테스트셋 economic regret 기반 진단지표 (시도방법7.md §5.2~5.4, §7.4).
    scale(=30)을 곱한 화폐 단위 — 3후보 오라클 {0,S,1}을 쓴다(Gap과는 다른 오라클)."""
    pc = rt + penalty_rate * da
    mismatch = actual - pred_flat
    surplus = np.maximum(mismatch, 0.0)
    shortage = np.maximum(-mismatch, 0.0)
    realized = scale * (da * pred_flat + rt * surplus - pc * shortage)
    oracle = scale * oracle_3cand_raw(actual, da, rt, pc)
    regret = oracle - realized
    regret = np.where(np.abs(regret) < 1e-6, 0.0, regret)
    if np.min(regret) < -1e-3:
        print(f"  [경고] 음수 regret 발견 (최소={np.min(regret):.6f}) — 3후보 오라클을 확인하세요")
    n = len(regret)
    tail_count = max(1, int(np.ceil((1.0 - CVAR_ALPHA) * n)))
    sorted_r = np.sort(regret)
    cvar90 = float(np.mean(sorted_r[-tail_count:]))
    return {
        "total_regret": float(np.sum(regret)),
        "mean_regret": float(np.mean(regret)),
        "cvar90_regret": cvar90,
        "max_regret": float(np.max(regret)),
        "gap_from_regret": 100.0 * float(np.sum(regret)) / float(np.sum(oracle)),
    }


# =====================================================================
# 4. AR baseline (방법2와 완전히 동일 — CVaR과 무관, bounded LAD)
# =====================================================================
print("\n" + "=" * 70)
print("  2. AR baseline (bounded LAD, 시간대별)")
print("=" * 70)
ar_coefficients = np.zeros((HOURS_PER_DAY, n_features_ar))
for h in range(HOURS_PER_DAY):
    y_h = train_solar[:, h]
    X_sp = sparse.csr_matrix(ar_design)
    I_n = sparse.eye(n_ar_rows, format="csr")
    A = sparse.vstack([sparse.hstack([X_sp, -I_n]),
                       sparse.hstack([-X_sp, -I_n])], format="csr")
    b = np.concatenate([y_h, -y_h])
    c = np.concatenate([np.zeros(n_features_ar), np.ones(n_ar_rows) / n_ar_rows])
    vb = [(None, None)] * n_features_ar + [(0.0, None)] * n_ar_rows
    res = linprog(c, A_ub=A, b_ub=b, bounds=vb, method="highs")
    ar_coefficients[h] = res.x[:n_features_ar]

ar_test_forecast = np.zeros((n_test_days, HOURS_PER_DAY))
prev_day = train_solar[-1]
for d in range(n_test_days):
    feat = np.concatenate([[1.0], prev_day[::-1]])
    for h in range(HOURS_PER_DAY):
        ar_test_forecast[d, h] = np.clip(np.dot(ar_coefficients[h], feat), 0, 1)
    prev_day = test_solar[d]

ar_pred_flat = ar_test_forecast.flatten()
ar_nrmse = nrmse(ar_pred_flat, actual_flat_all)
print(f"  AR baseline nRMSE = {ar_nrmse:.2f}%")


# =====================================================================
# 5. MLR baseline (방법2와 완전히 동일)
# =====================================================================
print("\n" + "=" * 70)
print("  3. MLR baseline (bounded LAD, pooled)")
print("=" * 70)
X_sp_mlr = sparse.csr_matrix(X_mlr_train)
I_mlr = sparse.eye(n_train_obs, format="csr")
A_mlr = sparse.vstack([sparse.hstack([X_sp_mlr, -I_mlr]),
                        sparse.hstack([-X_sp_mlr, -I_mlr])], format="csr")
b_mlr = np.concatenate([mlr_train_solar, -mlr_train_solar])
c_mlr = np.concatenate([np.zeros(n_features_mlr), np.ones(n_train_obs) / n_train_obs])
vb_mlr = [(None, None)] * n_features_mlr + [(0.0, None)] * n_train_obs
res_mlr = linprog(c_mlr, A_ub=A_mlr, b_ub=b_mlr, bounds=vb_mlr, method="highs")
mlr_coefficients = res_mlr.x[:n_features_mlr]

mlr_pred_flat = np.clip(X_mlr_test @ mlr_coefficients, 0, 1)
mlr_nrmse = nrmse(mlr_pred_flat, actual_flat)
print(f"  MLR baseline nRMSE = {mlr_nrmse:.2f}%")


# =====================================================================
# 6. 제안모형 MILP — AR (시간대별) — 방법2 + additive CVaR
# =====================================================================
# 변수 순서: [beta(13), x(n_obs), y+(n_obs), y-(n_obs), z_bin(n_bin), zeta(1), s(n_obs)]
# CVaR 제약: s_i + zeta + DA_i*x_i + RT_i*y+_i - PC_i*y-_i >= oracle_i
#   (<=> s_i >= regret_i - zeta,  regret_i = oracle_i - profit_i)
def solve_ar_proposed_cvar(penalty_rate, W1, W2, lam):
    """AR 제안모형 + additive CVaR: 12시간대별 MILP (raw 계수 + W2 balance weight + CVaR)"""
    W2_eff = W2 * W2_BALANCE_WEIGHT
    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))
    for hour in range(HOURS_PER_DAY):
        y_h = train_solar[:, hour]
        da_h = train_da_price[:, hour]
        rt_h = train_rt_price[:, hour]
        n_obs = n_ar_rows

        pc_h = rt_h + penalty_rate * da_h  # PC = RT + rate*DA (방법2와 동일)

        sc = -W1 * rt_h + W2_eff       # surplus cost
        yc = W1 * pc_h + W2_eff        # shortage cost
        oracle_h = oracle_3cand_raw(y_h, da_h, rt_h, pc_h)  # CVaR용 raw 3후보 오라클

        bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
        bin_arr = np.array(bin_list, dtype=int)
        n_bin = len(bin_arr)

        b_s = 0; x_s = n_features_ar; yp_s = n_features_ar + n_obs
        ym_s = n_features_ar + 2 * n_obs; z_s = n_features_ar + 3 * n_obs
        zeta_s = z_s + n_bin; s_s = zeta_s + 1
        n_var = s_s + n_obs

        obj = np.zeros(n_var)
        obj[x_s:x_s + n_obs] = -W1 * da_h
        obj[yp_s:yp_s + n_obs] = sc
        obj[ym_s:ym_s + n_obs] = yc
        obj[zeta_s] = W1 * lam * n_obs
        obj[s_s:s_s + n_obs] = W1 * lam / (1.0 - CVAR_ALPHA)

        X_sp = sparse.csr_matrix(ar_design)
        I_n = sparse.eye(n_obs, format="csr")
        eq_a = sparse.lil_matrix((n_obs, n_var))
        eq_a[:, b_s:b_s+n_features_ar] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
        eq_b = sparse.lil_matrix((n_obs, n_var))
        eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n
        eq_b[:, ym_s:ym_s+n_obs] = -I_n
        all_eq = sparse.vstack([eq_a, eq_b], format="csr")
        eq_rhs = np.concatenate([np.zeros(n_obs), y_h])
        all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

        if n_bin > 0:
            comp = sparse.lil_matrix((2 * n_bin, n_var))
            for k in range(n_bin):
                r = bin_arr[k]
                comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
                comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
            all_con.append(LinearConstraint(comp.tocsr(),
                          np.full(2*n_bin, -np.inf),
                          np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

        # CVaR 선형화 제약: s_i + zeta + DA_i*x_i + RT_i*y+_i - PC_i*y-_i >= oracle_i
        if lam > 0.0:
            cvar_a = sparse.lil_matrix((n_obs, n_var))
            for i in range(n_obs):
                cvar_a[i, s_s + i] = 1.0
                cvar_a[i, zeta_s] = 1.0
                cvar_a[i, x_s + i] = da_h[i]
                cvar_a[i, yp_s + i] = rt_h[i]
                cvar_a[i, ym_s + i] = -pc_h[i]
            all_con.append(LinearConstraint(cvar_a.tocsr(), oracle_h, np.full(n_obs, np.inf)))
        # lam=0이면 zeta/s의 목적함수 계수가 0이라 제약을 걸지 않아도 beta에 영향 없음
        # (등가성은 §16 검증 루틴에서 별도로 확인)

        lb = np.concatenate([np.full(n_features_ar, -np.inf), np.zeros(3*n_obs+n_bin), [0.0], np.zeros(n_obs)])
        ub = np.concatenate([np.full(n_features_ar, np.inf), np.ones(3*n_obs+n_bin), [np.inf], np.full(n_obs, np.inf)])
        integ = np.zeros(n_var, dtype=int)
        integ[z_s:z_s+n_bin] = 1

        r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),
                 constraints=all_con, options={"mip_rel_gap": 1e-9})
        if not r.success:
            raise RuntimeError(f"AR CVaR MILP failed h={hour} lam={lam}: {r.message}")
        coeffs[hour] = r.x[b_s:b_s+n_features_ar]

    fc = np.zeros((n_test_days, HOURS_PER_DAY))
    prev = train_solar[-1]
    for d in range(n_test_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY):
            fc[d, h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = test_solar[d]
    return fc.flatten()


# =====================================================================
# 7. 제안모형 MILP — MLR (pooled) — 방법2 + additive CVaR
# =====================================================================
def solve_mlr_proposed_cvar(penalty_rate, W1, W2, lam):
    """MLR 제안모형 + additive CVaR: pooled MILP 하나 (raw 계수 + W2 balance weight + CVaR)"""
    W2_eff = W2 * W2_BALANCE_WEIGHT
    n_obs = n_train_obs

    pc_all = mlr_train_rt + penalty_rate * mlr_train_da
    sc = -W1 * mlr_train_rt + W2_eff
    yc = W1 * pc_all + W2_eff
    oracle_all = oracle_3cand_raw(mlr_train_solar, mlr_train_da, mlr_train_rt, pc_all)

    bin_list = [i for i in range(n_obs) if sc[i] + yc[i] < 0]
    bin_arr = np.array(bin_list, dtype=int)
    n_bin = len(bin_arr)

    b_s = 0; x_s = n_features_mlr; yp_s = n_features_mlr + n_obs
    ym_s = n_features_mlr + 2 * n_obs; z_s = n_features_mlr + 3 * n_obs
    zeta_s = z_s + n_bin; s_s = zeta_s + 1
    n_var = s_s + n_obs

    obj = np.zeros(n_var)
    obj[x_s:x_s + n_obs] = -W1 * mlr_train_da
    obj[yp_s:yp_s + n_obs] = sc
    obj[ym_s:ym_s + n_obs] = yc
    obj[zeta_s] = W1 * lam * n_obs
    obj[s_s:s_s + n_obs] = W1 * lam / (1.0 - CVAR_ALPHA)

    X_sp = sparse.csr_matrix(X_mlr_train)
    I_n = sparse.eye(n_obs, format="csr")
    eq_a = sparse.lil_matrix((n_obs, n_var))
    eq_a[:, b_s:b_s+n_features_mlr] = -X_sp; eq_a[:, x_s:x_s+n_obs] = I_n
    eq_b = sparse.lil_matrix((n_obs, n_var))
    eq_b[:, x_s:x_s+n_obs] = I_n; eq_b[:, yp_s:yp_s+n_obs] = I_n
    eq_b[:, ym_s:ym_s+n_obs] = -I_n
    all_eq = sparse.vstack([eq_a, eq_b], format="csr")
    eq_rhs = np.concatenate([np.zeros(n_obs), mlr_train_solar])
    all_con = [LinearConstraint(all_eq, eq_rhs, eq_rhs)]

    if n_bin > 0:
        comp = sparse.lil_matrix((2 * n_bin, n_var))
        for k in range(n_bin):
            r = bin_arr[k]
            comp[k, yp_s+r] = 1.0; comp[k, z_s+k] = 1.0
            comp[n_bin+k, ym_s+r] = 1.0; comp[n_bin+k, z_s+k] = -1.0
        all_con.append(LinearConstraint(comp.tocsr(),
                      np.full(2*n_bin, -np.inf),
                      np.concatenate([np.ones(n_bin), np.zeros(n_bin)])))

    if lam > 0.0:
        cvar_a = sparse.lil_matrix((n_obs, n_var))
        for i in range(n_obs):
            cvar_a[i, s_s + i] = 1.0
            cvar_a[i, zeta_s] = 1.0
            cvar_a[i, x_s + i] = mlr_train_da[i]
            cvar_a[i, yp_s + i] = mlr_train_rt[i]
            cvar_a[i, ym_s + i] = -pc_all[i]
        all_con.append(LinearConstraint(cvar_a.tocsr(), oracle_all, np.full(n_obs, np.inf)))

    lb = np.concatenate([np.full(n_features_mlr, -np.inf), np.zeros(3*n_obs+n_bin), [0.0], np.zeros(n_obs)])
    ub = np.concatenate([np.full(n_features_mlr, np.inf), np.ones(3*n_obs+n_bin), [np.inf], np.full(n_obs, np.inf)])
    integ = np.zeros(n_var, dtype=int)
    integ[z_s:z_s+n_bin] = 1

    r = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub),
             constraints=all_con, options={"mip_rel_gap": 1e-9})
    if not r.success:
        raise RuntimeError(f"MLR CVaR MILP failed lam={lam}: {r.message}")
    coeffs = r.x[b_s:b_s+n_features_mlr]
    return np.clip(X_mlr_test @ coeffs, 0, 1)


# =====================================================================
# 8. 캐시 — (rate, W1, W2, lam) 키
# =====================================================================
cache_ar = {}
cache_mlr = {}

def get_ar(pr, w1, w2, lam):
    k = (pr, w1, w2, lam)
    if k not in cache_ar:
        pred = solve_ar_proposed_cvar(pr, w1, w2, lam)
        gap = compute_gap_4term(pred, pr, actual_flat_all, da_flat_all, rt_flat_all)
        diag = compute_regret_diagnostics(pred, pr, actual_flat_all, da_flat_all, rt_flat_all)
        cache_ar[k] = (nrmse(pred, actual_flat_all), gap, diag, pred)
    return cache_ar[k]

def get_mlr(pr, w1, w2, lam):
    k = (pr, w1, w2, lam)
    if k not in cache_mlr:
        pred = solve_mlr_proposed_cvar(pr, w1, w2, lam)
        gap = compute_gap_4term(pred, pr, actual_flat, da_flat, rt_flat)
        diag = compute_regret_diagnostics(pred, pr, actual_flat, da_flat, rt_flat)
        cache_mlr[k] = (nrmse(pred, actual_flat), gap, diag, pred)
    return cache_mlr[k]


# =====================================================================
# 9. λ=0 등가성 검증 (시도방법7.md §16) — 방법2와 예측값이 거의 같아야 함
# =====================================================================
print("\n" + "=" * 70)
print("  4. λ=0 등가성 검증 (CVaR 항을 붙였을 때 λ=0이면 방법2와 같은가?)")
print("=" * 70)
ar_n0, ar_g0, ar_diag0, ar_pred0 = get_ar(KPI_RATE, KPI_W1, KPI_W2, 0.0)
print(f"  λ=0, W1=1,W2=20,rate=50%: AR nRMSE={ar_n0:.4f}%, Gap={ar_g0:.4f}%")
print("  ↑ 이 값을 방법2_4term_ar_mlr_z03_block18.py의 같은 KPI 지점 출력과 대조할 것 —")
print("    시도방법7.md §16 기준 예측값 최대차이 ~1e-4 수준(=사실상 동일)이어야 정상이다.")
print("    (zeta/s 변수는 λ=0이어도 그대로 남아있지만 목적함수 계수가 0이라 beta에는 영향이 없다)")


# =====================================================================
# 10. KPI 지점 λ 스윕 (시도방법7.md §14와 동일 실험)
# =====================================================================
print("\n" + "=" * 70)
print(f"  5. λ 스윕 — W1={KPI_W1}, W2={KPI_W2}, rate={KPI_RATE} (시도방법7.md §14)")
print("=" * 70)

lambda_rows = []
for lam in LAMBDA_SWEEP:
    a_n, a_g, a_diag, _ = get_ar(KPI_RATE, KPI_W1, KPI_W2, lam)
    m_n, m_g, m_diag, _ = get_mlr(KPI_RATE, KPI_W1, KPI_W2, lam)
    lambda_rows.append((lam, a_n, a_g, a_diag, m_n, m_g, m_diag))
    print(f"  λ={lam:.3f}  AR(nRMSE={a_n:.2f}%, Gap={a_g:.2f}%, "
          f"TotalRegret={a_diag['total_regret']:.1f}, CVaR90={a_diag['cvar90_regret']:.1f}, "
          f"MaxRegret={a_diag['max_regret']:.1f})")
    print(f"           MLR(nRMSE={m_n:.2f}%, Gap={m_g:.2f}%, "
          f"TotalRegret={m_diag['total_regret']:.1f}, CVaR90={m_diag['cvar90_regret']:.1f}, "
          f"MaxRegret={m_diag['max_regret']:.1f})")

lambda_csv = os.path.join(OUT_DIR, "방법7_lambda_sweep_ar_mlr_z03_block18.csv")
with open(lambda_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["lambda", "AR_nRMSE", "AR_Gap", "AR_TotalRegret", "AR_MeanRegret",
                "AR_CVaR90", "AR_MaxRegret",
                "MLR_nRMSE", "MLR_Gap", "MLR_TotalRegret", "MLR_MeanRegret",
                "MLR_CVaR90", "MLR_MaxRegret"])
    for lam, a_n, a_g, a_diag, m_n, m_g, m_diag in lambda_rows:
        w.writerow([lam, round(a_n, 4), round(a_g, 4),
                    round(a_diag["total_regret"], 2), round(a_diag["mean_regret"], 4),
                    round(a_diag["cvar90_regret"], 2), round(a_diag["max_regret"], 2),
                    round(m_n, 4), round(m_g, 4),
                    round(m_diag["total_regret"], 2), round(m_diag["mean_regret"], 4),
                    round(m_diag["cvar90_regret"], 2), round(m_diag["max_regret"], 2)])
print(f"\n  saved: {lambda_csv}")

# λ=0 대비 변화율(%) — 시도방법7.md §14 표와 같은 형식
lam0 = lambda_rows[0]
print("\n  λ=0 대비 변화 (AR):")
for lam, a_n, a_g, a_diag, *_ in lambda_rows:
    if lam == 0.0:
        continue
    d_n = a_n - lam0[1]; d_g = a_g - lam0[2]
    d_tr = 100.0 * (a_diag["total_regret"] - lam0[3]["total_regret"]) / abs(lam0[3]["total_regret"])
    d_cvar = 100.0 * (a_diag["cvar90_regret"] - lam0[3]["cvar90_regret"]) / abs(lam0[3]["cvar90_regret"])
    print(f"    λ={lam:.3f}: ΔnRMSE={d_n:+.2f}%p  ΔGap={d_g:+.2f}%p  "
          f"ΔTotalRegret={d_tr:+.2f}%  ΔCVaR90={d_cvar:+.2f}%")


# =====================================================================
# 11. Fig.3/6 (W1/W2 스윕, rate=50% 고정) — λ=0 vs λ=0.05 vs 논문
# =====================================================================
print("\n" + "=" * 70)
print(f"  6. Fig.3/6 데이터 — W1/W2 10개점 스윕 (rate={KPI_RATE}, λ=0 vs λ={CVAR_LAMBDA_MAIN})")
print("=" * 70)

fig3_ar_n0 = [ar_nrmse]; fig3_ar_g0 = [ar_g0]
fig3_ar_n5 = [ar_nrmse]; fig3_ar_g5 = [None]
fig6_mlr_n0 = [mlr_nrmse]; fig6_mlr_g0 = None
fig6_mlr_n5 = [mlr_nrmse]; fig6_mlr_g5 = [None]

mlr_n0_kpi, mlr_g0_kpi, _, _ = get_mlr(KPI_RATE, KPI_W1, KPI_W2, 0.0)
fig6_mlr_g0 = [mlr_g0_kpi]
_, ar_g5_base, _, _ = get_ar(KPI_RATE, KPI_W1, KPI_W2, CVAR_LAMBDA_MAIN)
fig3_ar_g5[0] = ar_g5_base
_, mlr_g5_base, _, _ = get_mlr(KPI_RATE, KPI_W1, KPI_W2, CVAR_LAMBDA_MAIN)
fig6_mlr_g5[0] = mlr_g5_base

for W1, W2 in FIG_W_RATIOS:
    a_n0, a_g0, _, _ = get_ar(KPI_RATE, W1, W2, 0.0)
    a_n5, a_g5, _, _ = get_ar(KPI_RATE, W1, W2, CVAR_LAMBDA_MAIN)
    m_n0, m_g0, _, _ = get_mlr(KPI_RATE, W1, W2, 0.0)
    m_n5, m_g5, _, _ = get_mlr(KPI_RATE, W1, W2, CVAR_LAMBDA_MAIN)
    fig3_ar_n0.append(a_n0); fig3_ar_g0.append(a_g0)
    fig3_ar_n5.append(a_n5); fig3_ar_g5.append(a_g5)
    fig6_mlr_n0.append(m_n0); fig6_mlr_g0.append(m_g0)
    fig6_mlr_n5.append(m_n5); fig6_mlr_g5.append(m_g5)
    print(f"  {W1}/{W2}: AR λ0(nRMSE={a_n0:.2f}%,Gap={a_g0:.2f}%) "
          f"λ{CVAR_LAMBDA_MAIN}(nRMSE={a_n5:.2f}%,Gap={a_g5:.2f}%)  "
          f"MLR λ0(nRMSE={m_n0:.2f}%,Gap={m_g0:.2f}%) λ{CVAR_LAMBDA_MAIN}(nRMSE={m_n5:.2f}%,Gap={m_g5:.2f}%)")


# =====================================================================
# 12. Fig.5/8 (rate 0~100% 스윕, W1=W2=1) — λ=0 vs λ=0.05 vs 논문
# =====================================================================
print("\n" + "=" * 70)
print(f"  7. Fig.5/8 데이터 — rate 0~100% 스윕 (W1={FIG58_W1}, W2={FIG58_W2}, λ=0 vs λ={CVAR_LAMBDA_FIG58})")
print("  ※ Fig.3/6과 λ값이 다르다 — W1=W2=1이라 W2_eff가 KPI 지점의 1/20이라 λ도 별도 재보정했다(시도방법7.md §25)")
print("=" * 70)

fig5_ar_base_n = []; fig5_ar_base_g = []
fig5_ar_l0_n = []; fig5_ar_l0_g = []
fig5_ar_l5_n = []; fig5_ar_l5_g = []
fig8_mlr_base_n = []; fig8_mlr_base_g = []
fig8_mlr_l0_n = []; fig8_mlr_l0_g = []
fig8_mlr_l5_n = []; fig8_mlr_l5_g = []

for rate in FIG_RATES:
    ar_base_g_r = compute_gap_4term(ar_pred_flat, rate, actual_flat_all, da_flat_all, rt_flat_all)
    mlr_base_g_r = compute_gap_4term(mlr_pred_flat, rate, actual_flat, da_flat, rt_flat)
    fig5_ar_base_n.append(ar_nrmse); fig5_ar_base_g.append(ar_base_g_r)
    fig8_mlr_base_n.append(mlr_nrmse); fig8_mlr_base_g.append(mlr_base_g_r)

    a_n0, a_g0, _, _ = get_ar(rate, FIG58_W1, FIG58_W2, 0.0)
    a_n5, a_g5, _, _ = get_ar(rate, FIG58_W1, FIG58_W2, CVAR_LAMBDA_FIG58)
    m_n0, m_g0, _, _ = get_mlr(rate, FIG58_W1, FIG58_W2, 0.0)
    m_n5, m_g5, _, _ = get_mlr(rate, FIG58_W1, FIG58_W2, CVAR_LAMBDA_FIG58)
    fig5_ar_l0_n.append(a_n0); fig5_ar_l0_g.append(a_g0)
    fig5_ar_l5_n.append(a_n5); fig5_ar_l5_g.append(a_g5)
    fig8_mlr_l0_n.append(m_n0); fig8_mlr_l0_g.append(m_g0)
    fig8_mlr_l5_n.append(m_n5); fig8_mlr_l5_g.append(m_g5)

    print(f"  rate={rate:.1f}: AR λ0(nRMSE={a_n0:.2f}%,Gap={a_g0:.2f}%) "
          f"λ{CVAR_LAMBDA_FIG58}(nRMSE={a_n5:.2f}%,Gap={a_g5:.2f}%)  "
          f"MLR λ0(nRMSE={m_n0:.2f}%,Gap={m_g0:.2f}%) λ{CVAR_LAMBDA_FIG58}(nRMSE={m_n5:.2f}%,Gap={m_g5:.2f}%)")


# =====================================================================
# 13. CSV 저장
# =====================================================================
def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)
        for row in rows:
            csv.writer(f).writerow(row)
    print(f"  saved: {path}")

save_csv(os.path.join(OUT_DIR, "방법7_fig3_ar_z03_block18.csv"),
         ["Label", "nRMSE_lam0", "Gap_lam0", f"nRMSE_lam{CVAR_LAMBDA_MAIN}", f"Gap_lam{CVAR_LAMBDA_MAIN}"],
         [[FIG_LABELS[i], round(fig3_ar_n0[i], 2), round(fig3_ar_g0[i], 2),
           round(fig3_ar_n5[i], 2), round(fig3_ar_g5[i], 2)]
          for i in range(len(FIG_LABELS))])

save_csv(os.path.join(OUT_DIR, "방법7_fig6_mlr_z03_block18.csv"),
         ["Label", "nRMSE_lam0", "Gap_lam0", f"nRMSE_lam{CVAR_LAMBDA_MAIN}", f"Gap_lam{CVAR_LAMBDA_MAIN}"],
         [[FIG_LABELS[i], round(fig6_mlr_n0[i], 2), round(fig6_mlr_g0[i], 2),
           round(fig6_mlr_n5[i], 2), round(fig6_mlr_g5[i], 2)]
          for i in range(len(FIG_LABELS))])

save_csv(os.path.join(OUT_DIR, "방법7_fig5_ar_z03_block18.csv"),
         ["Rate", "AR_base_nRMSE", "AR_base_Gap", "AR_lam0_nRMSE", "AR_lam0_Gap",
          f"AR_lam{CVAR_LAMBDA_FIG58}_nRMSE", f"AR_lam{CVAR_LAMBDA_FIG58}_Gap"],
         [[FIG_RATES[i], round(fig5_ar_base_n[i], 2), round(fig5_ar_base_g[i], 2),
           round(fig5_ar_l0_n[i], 2), round(fig5_ar_l0_g[i], 2),
           round(fig5_ar_l5_n[i], 2), round(fig5_ar_l5_g[i], 2)]
          for i in range(len(FIG_RATES))])

save_csv(os.path.join(OUT_DIR, "방법7_fig8_mlr_z03_block18.csv"),
         ["Rate", "MLR_base_nRMSE", "MLR_base_Gap", "MLR_lam0_nRMSE", "MLR_lam0_Gap",
          f"MLR_lam{CVAR_LAMBDA_FIG58}_nRMSE", f"MLR_lam{CVAR_LAMBDA_FIG58}_Gap"],
         [[FIG_RATES[i], round(fig8_mlr_base_n[i], 2), round(fig8_mlr_base_g[i], 2),
           round(fig8_mlr_l0_n[i], 2), round(fig8_mlr_l0_g[i], 2),
           round(fig8_mlr_l5_n[i], 2), round(fig8_mlr_l5_g[i], 2)]
          for i in range(len(FIG_RATES))])


# =====================================================================
# 14. Fig.3 — AR, dual y축 (nRMSE 좌, Gap 우) — 논문 / λ=0 / λ=0.05
# =====================================================================
print("\n" + "=" * 70)
print("  8. Fig.3 plotting (AR, W1/W2 스윕, dual y축)")
print("=" * 70)

x = list(range(len(FIG_LABELS)))
fig, ax_l = plt.subplots(figsize=(10, 6))
fig.suptitle(f"방법7_fig3 — AR PC=RT+rate×DA + Additive CVaR(α={CVAR_ALPHA}), "
             f"rate={KPI_RATE}, W1/W2 스윕", fontsize=12)
ax_r = ax_l.twinx()

paper_nrmse_all = [PAPER_AR_NRMSE] + PAPER_PROP_NRMSE
paper_gap_all = [PAPER_AR_GAP] + PAPER_PROP_GAP
ln1 = ax_l.plot(x, paper_nrmse_all, marker="o", color="#4C72B0", linewidth=1.5,
                linestyle="--", label="논문 nRMSE")
ln2 = ax_l.plot(x, fig3_ar_n0, marker="o", color="#1a3a6b", linewidth=2,
                linestyle="-", label="재현 nRMSE (λ=0, =방법2)")
ln2b = ax_l.plot(x, fig3_ar_n5, marker="^", color="#2ca02c", linewidth=2,
                 linestyle="-", label=f"재현 nRMSE (λ={CVAR_LAMBDA_MAIN})")
ln3 = ax_r.plot(x, paper_gap_all, marker="s", color="#eb9834", linewidth=1.5,
                linestyle="--", label="논문 Gap")
ln4 = ax_r.plot(x, fig3_ar_g0, marker="s", color="#b35900", linewidth=2,
                linestyle="-", label="재현 Gap (λ=0, =방법2)")
ln4b = ax_r.plot(x, fig3_ar_g5, marker="D", color="#8b0000", linewidth=2,
                 linestyle="-", label=f"재현 Gap (λ={CVAR_LAMBDA_MAIN})")

ax_l.set_xticks(x); ax_l.set_xticklabels(FIG_LABELS)
ax_l.set_xlabel("W1/W2"); ax_l.set_ylabel("nRMSE (%)", color="#2a5599")
ax_r.set_ylabel("Optimality Gap (%)", color="#b35900")
ax_l.tick_params(axis="y", labelcolor="#2a5599")
ax_r.tick_params(axis="y", labelcolor="#b35900")
ax_l.grid(True, alpha=0.3)
ax_l.set_ylim(*auto_ylim(paper_nrmse_all, fig3_ar_n0, fig3_ar_n5))
ax_r.set_ylim(*auto_ylim(paper_gap_all, fig3_ar_g0, fig3_ar_g5))
all_ln = ln1+ln2+ln2b+ln3+ln4+ln4b
ax_l.legend(all_ln, [l.get_label() for l in all_ln], loc="upper left", fontsize=8)
fig.tight_layout()
p3 = os.path.join(RESULTS_DIR, "방법7_fig3_ar_z03_block18.png")
fig.savefig(p3, dpi=150); plt.close(fig)
print(f"  saved: {p3}")


# =====================================================================
# 15. Fig.5 — AR, rate 스윕 (nRMSE / Gap 2분할) — 논문 / λ=0 / λ=0.05
# =====================================================================
print("\n" + "=" * 70)
print("  9. Fig.5 plotting (AR, rate 스윕, 2분할)")
print("=" * 70)

x5 = list(range(len(FIG_RATES)))
lbl5 = [f"{int(r*100)}%" for r in FIG_RATES]
x5_paper = x5[:11]

fig5_fig, (ax5n, ax5g) = plt.subplots(1, 2, figsize=(13, 5))
fig5_fig.suptitle(f"방법7_fig5 — AR PC=RT+rate×DA + Additive CVaR, W1={FIG58_W1}/W2={FIG58_W2}, "
                  f"penalty rate 스윕 (λ=0 vs λ={CVAR_LAMBDA_FIG58}, Fig.5/8 전용 재보정값)", fontsize=12)

ax5n.plot(x5_paper, FIG5_PAPER_AR_NRMSE, linestyle="--", color="#4C72B0", alpha=0.8,
          marker="o", markersize=4, label="논문 AR")
ax5n.plot(x5_paper, FIG5_PAPER_PROP_NRMSE, linestyle="--", color="#eb9834", alpha=0.8,
          marker="s", markersize=4, label="논문 제안모형")
ax5n.plot(x5, fig5_ar_l0_n, marker="s", color="#b35900", linewidth=2, linestyle="-",
          label="재현 λ=0 (=방법2)")
ax5n.plot(x5, fig5_ar_l5_n, marker="^", color="#2ca02c", linewidth=2, linestyle="-",
          label=f"재현 λ={CVAR_LAMBDA_FIG58}")
ax5n.set_xticks(x5); ax5n.set_xticklabels(lbl5, rotation=45)
ax5n.set_xlabel("벌금비용률"); ax5n.set_ylabel("nRMSE (%)")
ax5n.set_title("nRMSE"); ax5n.grid(True, alpha=0.3); ax5n.legend(fontsize=8)
ax5n.set_ylim(*auto_ylim(FIG5_PAPER_AR_NRMSE, FIG5_PAPER_PROP_NRMSE, fig5_ar_l0_n, fig5_ar_l5_n))

ax5g.plot(x5_paper, FIG5_PAPER_AR_GAP, linestyle="--", color="#4C72B0", alpha=0.8,
          marker="o", markersize=4, label="논문 AR")
ax5g.plot(x5_paper, FIG5_PAPER_PROP_GAP, linestyle="--", color="#eb9834", alpha=0.8,
          marker="s", markersize=4, label="논문 제안모형")
ax5g.plot(x5, fig5_ar_l0_g, marker="s", color="#b35900", linewidth=2, linestyle="-",
          label="재현 λ=0 (=방법2)")
ax5g.plot(x5, fig5_ar_l5_g, marker="^", color="#2ca02c", linewidth=2, linestyle="-",
          label=f"재현 λ={CVAR_LAMBDA_FIG58}")
hl = FIG_RATES.index(KPI_RATE) if KPI_RATE in FIG_RATES else None
if hl is not None:
    ax5g.axvline(hl, color=C_GRID, linestyle=":", linewidth=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")
ax5g.set_xticks(x5); ax5g.set_xticklabels(lbl5, rotation=45)
ax5g.set_xlabel("벌금비용률"); ax5g.set_ylabel("Optimality Gap (%)")
ax5g.set_title("Optimality Gap"); ax5g.grid(True, alpha=0.3); ax5g.legend(fontsize=8)
ax5g.set_ylim(*auto_ylim(FIG5_PAPER_AR_GAP, FIG5_PAPER_PROP_GAP, fig5_ar_l0_g, fig5_ar_l5_g))

fig5_fig.tight_layout()
p5 = os.path.join(RESULTS_DIR, "방법7_fig5_ar_z03_block18.png")
fig5_fig.savefig(p5, dpi=150); plt.close(fig5_fig)
print(f"  saved: {p5}")


# =====================================================================
# 16. Fig.6 — MLR, dual y축 — 논문 / λ=0 / λ=0.05
# =====================================================================
print("\n" + "=" * 70)
print("  10. Fig.6 plotting (MLR, W1/W2 스윕, dual y축)")
print("=" * 70)

fig6_fig, ax_l = plt.subplots(figsize=(10, 6))
fig6_fig.suptitle(f"방법7_fig6 — MLR PC=RT+rate×DA + Additive CVaR(α={CVAR_ALPHA}), "
                  f"rate={KPI_RATE}, W1/W2 스윕", fontsize=12)
ax_r = ax_l.twinx()

paper_mlr_nrmse_all = [PAPER_MLR_NRMSE] + PAPER_PROP_MLR_NRMSE
paper_mlr_gap_all = [PAPER_MLR_GAP] + PAPER_PROP_MLR_GAP
ln1 = ax_l.plot(x, paper_mlr_nrmse_all, marker="o", color="#4C72B0", linewidth=1.5,
                linestyle="--", label="논문 nRMSE")
ln2 = ax_l.plot(x, fig6_mlr_n0, marker="o", color="#1a3a6b", linewidth=2,
                linestyle="-", label="재현 nRMSE (λ=0, =방법2)")
ln2b = ax_l.plot(x, fig6_mlr_n5, marker="^", color="#2ca02c", linewidth=2,
                 linestyle="-", label=f"재현 nRMSE (λ={CVAR_LAMBDA_MAIN})")
ln3 = ax_r.plot(x, paper_mlr_gap_all, marker="s", color="#eb9834", linewidth=1.5,
                linestyle="--", label="논문 Gap")
ln4 = ax_r.plot(x, fig6_mlr_g0, marker="s", color="#b35900", linewidth=2,
                linestyle="-", label="재현 Gap (λ=0, =방법2)")
ln4b = ax_r.plot(x, fig6_mlr_g5, marker="D", color="#8b0000", linewidth=2,
                 linestyle="-", label=f"재현 Gap (λ={CVAR_LAMBDA_MAIN})")

ax_l.set_xticks(x); ax_l.set_xticklabels(FIG_LABELS)
ax_l.set_xlabel("W1/W2"); ax_l.set_ylabel("nRMSE (%)", color="#2a5599")
ax_r.set_ylabel("Optimality Gap (%)", color="#b35900")
ax_l.tick_params(axis="y", labelcolor="#2a5599")
ax_r.tick_params(axis="y", labelcolor="#b35900")
ax_l.grid(True, alpha=0.3)
ax_l.set_ylim(*auto_ylim(paper_mlr_nrmse_all, fig6_mlr_n0, fig6_mlr_n5))
ax_r.set_ylim(*auto_ylim(paper_mlr_gap_all, fig6_mlr_g0, fig6_mlr_g5))
all_ln = ln1+ln2+ln2b+ln3+ln4+ln4b
ax_l.legend(all_ln, [l.get_label() for l in all_ln], loc="upper left", fontsize=8)
fig6_fig.tight_layout()
p6 = os.path.join(RESULTS_DIR, "방법7_fig6_mlr_z03_block18.png")
fig6_fig.savefig(p6, dpi=150); plt.close(fig6_fig)
print(f"  saved: {p6}")


# =====================================================================
# 17. Fig.8 — MLR, rate 스윕 (nRMSE / Gap 2분할) — 논문 / λ=0 / λ=0.05
# =====================================================================
print("\n" + "=" * 70)
print("  11. Fig.8 plotting (MLR, rate 스윕, 2분할)")
print("=" * 70)

fig8_fig, (ax8n, ax8g) = plt.subplots(1, 2, figsize=(13, 5))
fig8_fig.suptitle(f"방법7_fig8 — MLR PC=RT+rate×DA + Additive CVaR, W1={FIG58_W1}/W2={FIG58_W2}, "
                  f"penalty rate 스윕 (λ=0 vs λ={CVAR_LAMBDA_FIG58}, Fig.5/8 전용 재보정값)", fontsize=12)

ax8n.plot(x5_paper, FIG8_PAPER_MLR_NRMSE, linestyle="--", color="#CC4654", alpha=0.8,
          marker="o", markersize=4, label="논문 MLR")
ax8n.plot(x5_paper, FIG8_PAPER_PROP_NRMSE, linestyle="--", color="#8CAED6", alpha=0.8,
          marker="s", markersize=4, label="논문 제안모형")
ax8n.plot(x5, fig8_mlr_l0_n, marker="s", color="#3d6a8f", linewidth=2, linestyle="-",
          label="재현 λ=0 (=방법2)")
ax8n.plot(x5, fig8_mlr_l5_n, marker="^", color="#2ca02c", linewidth=2, linestyle="-",
          label=f"재현 λ={CVAR_LAMBDA_FIG58}")
ax8n.set_xticks(x5); ax8n.set_xticklabels(lbl5, rotation=45)
ax8n.set_xlabel("벌금비용률"); ax8n.set_ylabel("nRMSE (%)")
ax8n.set_title("nRMSE"); ax8n.grid(True, alpha=0.3); ax8n.legend(fontsize=8)
ax8n.set_ylim(*auto_ylim(FIG8_PAPER_MLR_NRMSE, FIG8_PAPER_PROP_NRMSE, fig8_mlr_l0_n, fig8_mlr_l5_n))

ax8g.plot(x5_paper, FIG8_PAPER_MLR_GAP, linestyle="--", color="#CC4654", alpha=0.8,
          marker="o", markersize=4, label="논문 MLR")
ax8g.plot(x5_paper, FIG8_PAPER_PROP_GAP, linestyle="--", color="#8CAED6", alpha=0.8,
          marker="s", markersize=4, label="논문 제안모형")
ax8g.plot(x5, fig8_mlr_l0_g, marker="s", color="#3d6a8f", linewidth=2, linestyle="-",
          label="재현 λ=0 (=방법2)")
ax8g.plot(x5, fig8_mlr_l5_g, marker="^", color="#2ca02c", linewidth=2, linestyle="-",
          label=f"재현 λ={CVAR_LAMBDA_FIG58}")
if hl is not None:
    ax8g.axvline(hl, color=C_GRID, linestyle=":", linewidth=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")
ax8g.set_xticks(x5); ax8g.set_xticklabels(lbl5, rotation=45)
ax8g.set_xlabel("벌금비용률"); ax8g.set_ylabel("Optimality Gap (%)")
ax8g.set_title("Optimality Gap"); ax8g.grid(True, alpha=0.3); ax8g.legend(fontsize=8)
ax8g.set_ylim(*auto_ylim(FIG8_PAPER_MLR_GAP, FIG8_PAPER_PROP_GAP, fig8_mlr_l0_g, fig8_mlr_l5_g))

fig8_fig.tight_layout()
p8 = os.path.join(RESULTS_DIR, "방법7_fig8_mlr_z03_block18.png")
fig8_fig.savefig(p8, dpi=150); plt.close(fig8_fig)
print(f"  saved: {p8}")


# =====================================================================
print("\n" + "=" * 70)
print("  완료 — AR+MLR PC=RT+rate×DA + Additive CVaR 통합 실험 (z03/블록18)")
print("=" * 70)
print(f"""
  생성 파일:
    Fig.3 (AR W1/W2):        {p3}
    Fig.5 (AR rate):         {p5}
    Fig.6 (MLR W1/W2):       {p6}
    Fig.8 (MLR rate):        {p8}
    λ 스윕 CSV:               {lambda_csv}

  ※ λ=0인 이 스크립트의 AR/MLR 결과는 방법2_4term_ar_mlr_z03_block18.py의
    KPI 지점 결과와 (수치 오차 수준까지) 같아야 한다 — 다르면 CVaR 제약/목적함수
    구현에 문제가 있다는 뜻이니 두 스크립트의 KPI 지점 출력을 대조해서 확인할 것.
""")
