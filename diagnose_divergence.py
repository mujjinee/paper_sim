# 읽기 전용 진단 스크립트 (z03/블록18, W1=2,W2=1 발산 원인 확인용)
# 가설: 새벽/저녁 근접 시간대(hour_idx=0 또는 11)는 실제발전량이 거의 0에
# 가까워 학습용 오라클 합(denom_h)이 매우 작고, 그 결과 DA수익 계수가
# 상대적으로 커져 W1>=W2 부근에서 그 시간대부터 먼저 x=1로 포화된다.

import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

BASE_DIR = r"D:\03_JiWon\JiWonProject"
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")

HOURS_PER_DAY = 12
LOCAL_HOUR_START, LOCAL_HOUR_END = 9, 21
TRAIN_START, TRAIN_END = pd.Timestamp("2013-08-25"), pd.Timestamp("2013-11-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")
CAPACITY_MW, DURATION_HOURS, PENALTY_RATE = 30.0, 1.0, 0.5
W1, W2 = 2.0, 1.0   # Fig.3 에서 nRMSE 116%로 발산하기 시작한 지점

raw = pd.read_csv(MERGED_FILE)
raw["local_date"] = pd.to_datetime(raw["local_date"])
day = raw[(raw["local_hour"] >= LOCAL_HOUR_START) & (raw["local_hour"] < LOCAL_HOUR_END)].copy()
day["hour_idx"] = day["local_hour"] - LOCAL_HOUR_START

hist = day[day["local_date"] == HISTORY_DATE].sort_values("hour_idx")
train = day[(day["local_date"] >= TRAIN_START) & (day["local_date"] <= TRAIN_END)].sort_values(["local_date", "hour_idx"])

hist_solar = np.zeros((1, HOURS_PER_DAY))
for i, (_, r) in enumerate(hist.iterrows()):
    hist_solar[0, i % HOURS_PER_DAY] = r["solar_power"]

dates = sorted(train["local_date"].unique())
n = len(dates)
solar = np.zeros((n, HOURS_PER_DAY)); da = np.zeros((n, HOURS_PER_DAY)); rt = np.zeros((n, HOURS_PER_DAY))
for i, (_, r) in enumerate(train.iterrows()):
    d, h = i // HOURS_PER_DAY, i % HOURS_PER_DAY
    solar[d, h] = r["solar_power"]; da[d, h] = r["da_price"]; rt[d, h] = r["rt_price"]

hist_and_train = np.vstack([hist_solar, solar])
X = np.hstack([np.ones((n, 1)), np.array([hist_and_train[d][::-1] for d in range(n)])])
p = X.shape[1]

print(f"{'hour_idx':>8} {'mean_S':>8} {'denom_h':>14} {'n_binary':>9} {'mean_x_test무관':>0}")
for hour in range(HOURS_PER_DAY):
    y = solar[:, hour]; da_h = da[:, hour]; rt_h = rt[:, hour]
    scale = CAPACITY_MW * DURATION_HOURS
    oracle = np.zeros(n)
    for i in range(n):
        a, d_, r_ = y[i], da_h[i], rt_h[i]
        pc = PENALTY_RATE * d_
        p0 = scale * (r_ * a); pS = scale * (d_ * a)
        s1 = max(a - 1.0, 0.0); sh1 = max(1.0 - a, 0.0)
        p1 = scale * (d_ * 1.0 + r_ * s1 - pc * sh1)
        oracle[i] = max(p0, pS, p1)
    denom = oracle.sum()

    surplus_cost = -W1 * scale * rt_h / denom + W2 / n
    shortage_cost = W1 * scale * (PENALTY_RATE * da_h) / denom + W2 / n
    binary_rows = np.flatnonzero(surplus_cost + shortage_cost < 0.0)
    n_binary = len(binary_rows)

    beta_s, x_s = 0, p
    yp_s, ym_s, z_s = p + n, p + 2 * n, p + 3 * n
    n_var = p + 3 * n + n_binary
    obj = np.zeros(n_var)
    obj[x_s:x_s + n] = -W1 * scale * da_h / denom
    obj[yp_s:yp_s + n] = surplus_cost
    obj[ym_s:ym_s + n] = shortage_cost

    Xs = sparse.csr_matrix(X); I = sparse.eye(n, format="csr")
    eq_a = sparse.lil_matrix((n, n_var)); eq_a[:, beta_s:beta_s + p] = -Xs; eq_a[:, x_s:x_s + n] = I
    eq_b = sparse.lil_matrix((n, n_var)); eq_b[:, x_s:x_s + n] = I; eq_b[:, yp_s:yp_s + n] = I; eq_b[:, ym_s:ym_s + n] = -I
    eqs = sparse.vstack([eq_a, eq_b], format="csr")
    rhs = np.concatenate([np.zeros(n), y])
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

    lb = np.concatenate([np.full(p, -np.inf), np.zeros(3 * n + n_binary)])
    ub = np.concatenate([np.full(p, np.inf), np.ones(3 * n + n_binary)])
    integ = np.zeros(n_var, dtype=int)
    for k in range(n_binary):
        integ[z_s + k] = 1
    res = milp(c=obj, integrality=integ, bounds=Bounds(lb, ub), constraints=cons, options={"mip_rel_gap": 1e-9})
    x_val = res.x[x_s:x_s + n]
    n_at_cap = np.sum(x_val > 0.999)

    print(f"{hour:>8} {y.mean():>8.4f} {denom:>14.2f} {n_binary:>9} {n_at_cap:>8}/{n} 학습표본 x=1 포화")
