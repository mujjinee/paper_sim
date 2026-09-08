# -*- coding: utf-8 -*-

# =====================================================================

# method2_pc_sum_ar_mlr_z03_block18.py

#

# 방법2 (method1_pc_max_ar_mlr_z03_block18.py 변형):

#   벌금률 정의를 PC = rate · (DA + RT)로 변경 (method1의 max(DA,RT) 대신 합)

#   이익함수: DA*x + RP*y+ - PC*y-  (3항)

#   oracle: 3후보 {0, actual, 1.0}, PC = rate*(DA+RT)

#

# AR(시간대별 12개 모델) + MLR(pooled 1개 모델) 통합

# 대상: z03/블록18 (2013-08-25~2013-12-22, 120일)

#

# 출력:

#   - 전체 그리드(50조합) AR+MLR

#   - 논문 KPI(W1=1, W2=20, rate=50%) 별도 print

#   - Fig.3(AR dual y), Fig.5(AR 2분할), Fig.6(MLR dual y), Fig.8(MLR 2분할)

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

# 0. 한글 폰트 + 색상 팔레트

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



C_AR_BASE  = "#4C72B0"

C_AR_PROP  = "#55A868"

C_MLR_BASE = "#CC4654"

C_MLR_PROP = "#8CAED6"

C_GRID     = "#DDDDDD"

C_PAPER    = "#000000"



# =====================================================================

# 1. 설정 — z03/블록18

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



TRAIN_START  = pd.Timestamp("2013-08-25")

TRAIN_END    = pd.Timestamp("2013-11-22")

TEST_START   = pd.Timestamp("2013-11-23")

TEST_END     = pd.Timestamp("2013-12-22")

HISTORY_DATE = pd.Timestamp("2013-08-24")



CAPACITY_MW = 30.0

DURATION_HOURS = 1.0

scale = CAPACITY_MW * DURATION_HOURS



# ── 논문 KPI 조건 ──

KPI_W1 = 1

KPI_W2 = 20

KPI_RATE = 0.5



# ── 그리드 ──

GRID_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 1), (1, 0)]

GRID_RATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]



# ── Fig3/6: W1/W2 10개점 ──

FIG_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1),

                (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]

FIG_LABELS = ["AR/MLR", "1/20", "1/10", "1/5", "1/2",

              "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]



# ── Fig5/8: rate 0~150% ──

FIG_RATES = [round(0.1 * i, 1) for i in range(16)]



# ── 논문 참고값 (z03/블록18, 논문 Table 3/4 기준값) ──

PAPER_AR_NRMSE  = 34.76

PAPER_AR_GAP    = 15.04

PAPER_MLR_NRMSE = 21.76

PAPER_MLR_GAP   = 12.59





# =====================================================================

# 2. 데이터 로딩

# =====================================================================

print("=" * 70)

print("  1. 데이터 로딩")

print("=" * 70)

raw_table = pd.read_csv(MERGED_FILE)

raw_table["local_date"] = pd.to_datetime(raw_table["local_date"])



is_daylight = ((raw_table["local_hour"] >= LOCAL_HOUR_START) &
               (raw_table["local_hour"] < LOCAL_HOUR_END))

daylight_table = raw_table[is_daylight].copy()

daylight_table["hour_idx"] = daylight_table["local_hour"] - LOCAL_HOUR_START



# ── AR: (날짜 x 12) 배열 ──

is_history = daylight_table["local_date"] == HISTORY_DATE

history_rows = daylight_table[is_history].copy().sort_values("hour_idx")

history_solar = np.zeros((1, HOURS_PER_DAY))

for _, row in history_rows.iterrows():

    history_solar[0, row["hour_idx"]] = row["solar_power"]



is_train = ((daylight_table["local_date"] >= TRAIN_START) &
            (daylight_table["local_date"] <= TRAIN_END))

train_rows = daylight_table[is_train].copy().sort_values(["local_date", "hour_idx"])



is_test = ((daylight_table["local_date"] >= TEST_START) &
           (daylight_table["local_date"] <= TEST_END))

test_rows = daylight_table[is_test].copy().sort_values(["local_date", "hour_idx"])



train_dates = sorted(train_rows["local_date"].unique())

n_train_days = len(train_dates)

test_dates = sorted(test_rows["local_date"].unique())

n_test_days = len(test_dates)



train_solar = np.zeros((n_train_days, HOURS_PER_DAY))

train_da = np.zeros((n_train_days, HOURS_PER_DAY))

train_rt = np.zeros((n_train_days, HOURS_PER_DAY))

for _, row in train_rows.iterrows():

    d = (row["local_date"] - TRAIN_START).days

    h = row["hour_idx"]

    train_solar[d, h] = row["solar_power"]

    train_da[d, h] = row["da_price"]

    train_rt[d, h] = row["rt_price"]



test_solar = np.zeros((n_test_days, HOURS_PER_DAY))

test_da = np.zeros((n_test_days, HOURS_PER_DAY))

test_rt = np.zeros((n_test_days, HOURS_PER_DAY))

for _, row in test_rows.iterrows():

    d = (row["local_date"] - TEST_START).days

    h = row["hour_idx"]

    test_solar[d, h] = row["solar_power"]

    test_da[d, h] = row["da_price"]

    test_rt[d, h] = row["rt_price"]



# ── MLR: flat 배열 ──

n_train_obs = len(train_rows)

mlr_train_solar = train_rows["solar_power"].to_numpy()

mlr_train_da = train_rows["da_price"].to_numpy()

mlr_train_rt = train_rows["rt_price"].to_numpy()



n_test_obs = len(test_rows)

mlr_actual = test_rows["solar_power"].to_numpy()

mlr_da = test_rows["da_price"].to_numpy()

mlr_rt = test_rows["rt_price"].to_numpy()



n_features_mlr = 4

X_mlr_train = np.column_stack([np.ones(n_train_obs),

                                train_rows["dssrd"].to_numpy(),

                                train_rows["dtsr"].to_numpy(),

                                train_rows["hour_idx"].to_numpy(dtype=float)])

X_mlr_test = np.column_stack([np.ones(n_test_obs),

                               test_rows["dssrd"].to_numpy(),

                               test_rows["dtsr"].to_numpy(),

                               test_rows["hour_idx"].to_numpy(dtype=float)])



# ── AR: 설계행렬 ──

n_features_ar = 13

history_and_train = np.vstack([history_solar, train_solar])

ar_intercept = np.ones((n_train_days, 1))

ar_lag = np.zeros((n_train_days, HOURS_PER_DAY))

for d in range(n_train_days):

    ar_lag[d] = history_and_train[d][::-1]

ar_design = np.hstack([ar_intercept, ar_lag])



# ── test flat ──

ar_actual = test_solar.flatten()

ar_da = test_da.flatten()

ar_rt = test_rt.flatten()



print(f"  train: {n_train_days}일({n_train_obs}행), test: {n_test_days}일({n_test_obs}행)")





# =====================================================================

# 3. Gap 계산 — 3항, PC = rate * (DA + RT), oracle 3후보

# =====================================================================

def compute_gap(pc_rate, pred, actual, da, rt):

    """이익함수: DA*x + RP*y+ - PC*y-, PC = rate*(DA+RT)"""

    s_r = 0.0; s_o = 0.0

    for i in range(len(pred)):

        a = actual[i]; x = pred[i]

        dp = da[i]; rp = rt[i]

        pc = pc_rate * (dp + rp)

        m = a - x

        yp = max(m, 0); ym = max(-m, 0)

        s_r += scale * (dp * x + rp * yp - pc * ym)

        p0 = scale * rp * a

        pa = scale * dp * a

        s1 = max(a - 1.0, 0); y1 = max(1.0 - a, 0)

        p1 = scale * (dp * 1.0 + rp * s1 - pc * y1)

        s_o += max(p0, pa, p1)

    return 100.0 * (s_o - s_r) / s_o if s_o > 1e-10 else 0.0



def nrmse_fn(pred, actual):

    return 100.0 * np.sqrt(np.mean((actual - pred)**2)) / np.mean(actual)





# =====================================================================

# 4. AR baseline — bounded LAD

# =====================================================================

print("\n" + "=" * 70)

print("  2. AR baseline (bounded LAD)")

print("=" * 70)

ar_coeff = np.zeros((HOURS_PER_DAY, n_features_ar))

for h in range(HOURS_PER_DAY):

    y_h = train_solar[:, h]

    Xs = sparse.csr_matrix(ar_design)

    In = sparse.eye(n_train_days, format="csr")

    A = sparse.vstack([sparse.hstack([Xs, -In]), sparse.hstack([-Xs, -In])], format="csr")

    b = np.concatenate([y_h, -y_h])

    c = np.concatenate([np.zeros(n_features_ar), np.ones(n_train_days)/n_train_days])

    vb = [(None,None)]*n_features_ar + [(0.0,None)]*n_train_days

    ar_coeff[h] = linprog(c, A_ub=A, b_ub=b, bounds=vb, method="highs").x[:n_features_ar]



ar_fc = np.zeros((n_test_days, HOURS_PER_DAY))

prev = train_solar[-1]

for d in range(n_test_days):

    fv = np.concatenate([[1.0], prev[::-1]])

    for h in range(HOURS_PER_DAY):

        ar_fc[d, h] = np.clip(np.dot(ar_coeff[h], fv), 0, 1)

    prev = test_solar[d]

ar_pred = ar_fc.flatten()

ar_nrmse = nrmse_fn(ar_pred, ar_actual)

print(f"  AR baseline nRMSE = {ar_nrmse:.2f}%")





# =====================================================================

# 5. MLR baseline — bounded LAD (pooled)

# =====================================================================

print("\n" + "=" * 70)

print("  3. MLR baseline (bounded LAD, pooled)")

print("=" * 70)

Xs = sparse.csr_matrix(X_mlr_train)

In = sparse.eye(n_train_obs, format="csr")

A = sparse.vstack([sparse.hstack([Xs, -In]), sparse.hstack([-Xs, -In])], format="csr")

b = np.concatenate([mlr_train_solar, -mlr_train_solar])

c = np.concatenate([np.zeros(n_features_mlr), np.ones(n_train_obs)/n_train_obs])

vb = [(None,None)]*n_features_mlr + [(0.0,None)]*n_train_obs

mlr_coeff = linprog(c, A_ub=A, b_ub=b, bounds=vb, method="highs").x[:n_features_mlr]

mlr_pred = np.clip(X_mlr_test @ mlr_coeff, 0, 1)

mlr_nrmse = nrmse_fn(mlr_pred, mlr_actual)

print(f"  MLR baseline nRMSE = {mlr_nrmse:.2f}%")





# =====================================================================

# 6. AR 제안모형 MILP — PC = rate*(DA + RT)

# =====================================================================

def solve_ar(pc_rate, W1, W2):

    coeffs = np.zeros((HOURS_PER_DAY, n_features_ar))

    for hour in range(HOURS_PER_DAY):

        y_h = train_solar[:, hour]; da_h = train_da[:, hour]; rt_h = train_rt[:, hour]

        n_obs = n_train_days



        oracle = np.zeros(n_obs)

        for i in range(n_obs):

            a = y_h[i]; dp = da_h[i]; rp = rt_h[i]

            pc = pc_rate * (dp + rp)

            p0 = scale * rp * a; pa = scale * dp * a

            s1 = max(a-1.0, 0); y1 = max(1.0-a, 0)

            p1 = scale * (dp*1.0 + rp*s1 - pc*y1)

            oracle[i] = max(p0, pa, p1)

        denom = oracle.sum()



        sc = np.zeros(n_obs); yc = np.zeros(n_obs)

        for i in range(n_obs):

            pc = pc_rate * (da_h[i] + rt_h[i])

            sc[i] = (-W1*scale*rt_h[i]/denom) + (W2/n_obs)

            yc[i] = (W1*scale*pc/denom) + (W2/n_obs)



        bl = [i for i in range(n_obs) if sc[i]+yc[i] < 0]

        ba = np.array(bl, dtype=int); nb = len(ba)

        bs=0; xs=n_features_ar; yps=n_features_ar+n_obs

        yms=n_features_ar+2*n_obs; zs=n_features_ar+3*n_obs

        nv = n_features_ar+3*n_obs+nb



        obj = np.zeros(nv)

        for i in range(n_obs):

            obj[xs+i] = -W1*scale*da_h[i]/denom

            obj[yps+i] = sc[i]; obj[yms+i] = yc[i]



        Xs = sparse.csr_matrix(ar_design)

        In = sparse.eye(n_obs, format="csr")

        ea = sparse.lil_matrix((n_obs, nv))

        ea[:, bs:bs+n_features_ar] = -Xs; ea[:, xs:xs+n_obs] = In

        eb = sparse.lil_matrix((n_obs, nv))

        eb[:, xs:xs+n_obs] = In; eb[:, yps:yps+n_obs] = In

        eb[:, yms:yms+n_obs] = -In

        ae = sparse.vstack([ea, eb], format="csr")

        rhs_ar = np.concatenate([np.zeros(n_obs), y_h])

        con = [LinearConstraint(ae, rhs_ar, rhs_ar)]



        if nb > 0:

            cm = sparse.lil_matrix((2*nb, nv))

            for k in range(nb):

                r = ba[k]

                cm[k, yps+r]=1.0; cm[k, zs+k]=1.0

                cm[nb+k, yms+r]=1.0; cm[nb+k, zs+k]=-1.0

            con.append(LinearConstraint(cm.tocsr(), np.full(2*nb,-np.inf),

                          np.concatenate([np.ones(nb), np.zeros(nb)])))



        lb = np.concatenate([np.full(n_features_ar,-np.inf), np.zeros(3*n_obs+nb)])

        ub = np.concatenate([np.full(n_features_ar,np.inf), np.ones(3*n_obs+nb)])

        ig = np.zeros(nv, dtype=int); ig[zs:zs+nb] = 1

        r = milp(c=obj, integrality=ig, bounds=Bounds(lb,ub),

                 constraints=con, options={"mip_rel_gap":1e-9})

        if not r.success:

            raise RuntimeError(f"AR MILP h={hour}: {r.message}")

        coeffs[hour] = r.x[bs:bs+n_features_ar]



    fc = np.zeros((n_test_days, HOURS_PER_DAY))

    prev = train_solar[-1]

    for d in range(n_test_days):

        fv = np.concatenate([[1.0], prev[::-1]])

        for h in range(HOURS_PER_DAY):

            fc[d,h] = np.clip(np.dot(coeffs[h], fv), 0, 1)

        prev = test_solar[d]

    return fc.flatten()





# =====================================================================

# 7. MLR 제안모형 MILP — PC = rate*(DA + RT)

# =====================================================================

def solve_mlr(pc_rate, W1, W2):

    n_obs = n_train_obs

    oracle = np.zeros(n_obs)

    for i in range(n_obs):

        a=mlr_train_solar[i]; dp=mlr_train_da[i]; rp=mlr_train_rt[i]

        pc = pc_rate * (dp + rp)

        p0=scale*rp*a; pa=scale*dp*a

        s1=max(a-1.0,0); y1=max(1.0-a,0)

        p1=scale*(dp*1.0+rp*s1-pc*y1)

        oracle[i] = max(p0, pa, p1)

    denom = oracle.sum()



    sc = np.zeros(n_obs); yc = np.zeros(n_obs)

    for i in range(n_obs):

        pc = pc_rate * (mlr_train_da[i] + mlr_train_rt[i])

        sc[i] = (-W1*scale*mlr_train_rt[i]/denom) + (W2/n_obs)

        yc[i] = (W1*scale*pc/denom) + (W2/n_obs)



    bl = [i for i in range(n_obs) if sc[i]+yc[i]<0]

    ba = np.array(bl, dtype=int); nb = len(ba)

    bs=0; xs=n_features_mlr; yps=n_features_mlr+n_obs

    yms=n_features_mlr+2*n_obs; zs=n_features_mlr+3*n_obs

    nv = n_features_mlr+3*n_obs+nb



    obj = np.zeros(nv)

    for i in range(n_obs):

        obj[xs+i] = -W1*scale*mlr_train_da[i]/denom

        obj[yps+i]=sc[i]; obj[yms+i]=yc[i]



    Xs = sparse.csr_matrix(X_mlr_train)

    In = sparse.eye(n_obs, format="csr")

    ea = sparse.lil_matrix((n_obs, nv))

    ea[:, bs:bs+n_features_mlr]=-Xs; ea[:, xs:xs+n_obs]=In

    eb = sparse.lil_matrix((n_obs, nv))

    eb[:, xs:xs+n_obs]=In; eb[:, yps:yps+n_obs]=In

    eb[:, yms:yms+n_obs]=-In

    ae = sparse.vstack([ea, eb], format="csr")

    rhs_mlr = np.concatenate([np.zeros(n_obs), mlr_train_solar])

    con = [LinearConstraint(ae, rhs_mlr, rhs_mlr)]



    if nb > 0:

        cm = sparse.lil_matrix((2*nb, nv))

        for k in range(nb):

            r=ba[k]; cm[k,yps+r]=1.0; cm[k,zs+k]=1.0

            cm[nb+k,yms+r]=1.0; cm[nb+k,zs+k]=-1.0

        con.append(LinearConstraint(cm.tocsr(), np.full(2*nb,-np.inf),

                  np.concatenate([np.ones(nb), np.zeros(nb)])))



    lb = np.concatenate([np.full(n_features_mlr,-np.inf), np.zeros(3*n_obs+nb)])

    ub = np.concatenate([np.full(n_features_mlr,np.inf), np.ones(3*n_obs+nb)])

    ig = np.zeros(nv, dtype=int); ig[zs:zs+nb]=1

    r = milp(c=obj, integrality=ig, bounds=Bounds(lb,ub),

             constraints=con, options={"mip_rel_gap":1e-9})

    if not r.success:

        raise RuntimeError(f"MLR MILP: {r.message}")

    return np.clip(X_mlr_test @ r.x[bs:bs+n_features_mlr], 0, 1)





# =====================================================================

# 8. 캐시

# =====================================================================

cache_ar = {}; cache_mlr = {}

def get_ar(pr, w1, w2):

    k=(pr,w1,w2)

    if k not in cache_ar:

        p = solve_ar(pr, w1, w2)

        cache_ar[k] = (nrmse_fn(p, ar_actual), compute_gap(pr, p, ar_actual, ar_da, ar_rt))

    return cache_ar[k]

def get_mlr(pr, w1, w2):

    k=(pr,w1,w2)

    if k not in cache_mlr:

        p = solve_mlr(pr, w1, w2)

        cache_mlr[k] = (nrmse_fn(p, mlr_actual), compute_gap(pr, p, mlr_actual, mlr_da, mlr_rt))

    return cache_mlr[k]





# =====================================================================

# 9. 그리드 스윕 (50조합, AR+MLR)

# =====================================================================

print("\n" + "=" * 70)

print("  4. 그리드 스윕 (rate 10개 x W1/W2 5개 = 50조합, AR+MLR)")

print("=" * 70)



grid = []

for pr in GRID_RATES:

    print(f"\n  ── rate = {pr} ──")

    for W1, W2 in GRID_W_RATIOS:

        an, ag = get_ar(pr, W1, W2)

        mn, mg = get_mlr(pr, W1, W2)

        grid.append((pr, W1, W2, an, ag, mn, mg))

        print(f"  {W1}/{W2}: AR(nRMSE={an:.2f}%, Gap={ag:.2f}%) "

              f"MLR(nRMSE={mn:.2f}%, Gap={mg:.2f}%)")



# 그리드 CSV

gc = os.path.join(OUT_DIR, "grid_method2_pc_sum_ar_mlr_z03_block18.csv")

with open(gc, "w", newline="") as f:

    w = csv.writer(f)

    w.writerow(["rate","W1","W2","AR_nRMSE","AR_Gap","MLR_nRMSE","MLR_Gap"])

    for r in grid:

        w.writerow([r[0],r[1],r[2],round(r[3],2),round(r[4],2),round(r[5],2),round(r[6],2)])

print(f"\n  saved: {gc}")





# =====================================================================

# 10. 논문 KPI 조건 (W1=1, W2=20, rate=50%)

# =====================================================================

print("\n" + "=" * 70)

print(f"  5. 논문 KPI 조건 — W1={KPI_W1}, W2={KPI_W2}, penalty rate={KPI_RATE} (50%)")

print("=" * 70)



ar_kn, ar_kg = get_ar(KPI_RATE, KPI_W1, KPI_W2)

mlr_kn, mlr_kg = get_mlr(KPI_RATE, KPI_W1, KPI_W2)

ar_bg = compute_gap(KPI_RATE, ar_pred, ar_actual, ar_da, ar_rt)

mlr_bg = compute_gap(KPI_RATE, mlr_pred, mlr_actual, mlr_da, mlr_rt)



print(f"""

  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐

  │     모델          │  nRMSE (%)   │  Gap (%)     │  nRMSE (%)   │  Gap (%)     │

  │                  │  (제안모형)   │  (제안모형)   │  (baseline)  │  (baseline)  │

  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤

  │  AR  (z03/블록18) │ {ar_kn:>10.2f}   │ {ar_kg:>10.2f}   │ {ar_nrmse:>10.2f}   │ {ar_bg:>10.2f}   │

  │  MLR (z03/블록18) │ {mlr_kn:>10.2f}   │ {mlr_kg:>10.2f}   │ {mlr_nrmse:>10.2f}   │ {mlr_bg:>10.2f}   │

  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘



  ※ 논문 KPI 기준값 (z03/블록18):

      AR  baseline nRMSE={PAPER_AR_NRMSE}%, Gap={PAPER_AR_GAP}%

      MLR baseline nRMSE={PAPER_MLR_NRMSE}%, Gap={PAPER_MLR_GAP}%

""")





# =====================================================================

# 11. Fig3/6: W1/W2 10개점 (rate=0.5) + Fig5/8: rate 0~150% (W1=W2=1)

# =====================================================================

FIG36_RATE = KPI_RATE  # 0.5



print("\n" + "=" * 70)

print(f"  6. Fig.3/6 데이터 — W1/W2 10개점 (rate={FIG36_RATE})")

print("=" * 70)



f3n = []; f3g = []

f6n = []; f6g = []

for W1, W2 in FIG_W_RATIOS:

    an, ag = get_ar(FIG36_RATE, W1, W2)

    mn, mg = get_mlr(FIG36_RATE, W1, W2)

    f3n.append(an); f3g.append(ag)

    f6n.append(mn); f6g.append(mg)

    print(f"  {W1}/{W2}: AR(nRMSE={an:.2f}%, Gap={ag:.2f}%) "

          f"MLR(nRMSE={mn:.2f}%, Gap={mg:.2f}%)")



print("\n" + "=" * 70)

print(f"  7. Fig.5/8 데이터 — rate 0~150% (W1=W2=1)")

print("=" * 70)



f5_bn=[]; f5_bg=[]; f5_pn=[]; f5_pg=[]

f8_bn=[]; f8_bg=[]; f8_pn=[]; f8_pg=[]

for rate in FIG_RATES:

    abg = compute_gap(rate, ar_pred, ar_actual, ar_da, ar_rt)

    mbg = compute_gap(rate, mlr_pred, mlr_actual, mlr_da, mlr_rt)

    f5_bn.append(ar_nrmse); f5_bg.append(abg)

    f8_bn.append(mlr_nrmse); f8_bg.append(mbg)

    an, ag = get_ar(rate, 1, 1)

    mn, mg = get_mlr(rate, 1, 1)

    f5_pn.append(an); f5_pg.append(ag)

    f8_pn.append(mn); f8_pg.append(mg)

    print(f"  rate={rate:.1f}: AR(base={abg:.2f}%,prop={an:.2f}%/{ag:.2f}%) "

          f"MLR(base={mbg:.2f}%,prop={mn:.2f}%/{mg:.2f}%)")





# ── CSV 저장 ──

def save_csv(path, hdr, rows):

    with open(path, "w", newline="") as f:

        csv.writer(f).writerow(hdr)

        for r in rows: csv.writer(f).writerow(r)

    print(f"  saved: {path}")



save_csv(os.path.join(OUT_DIR, "fig3_method2_z03.csv"),

         ["Label","nRMSE","Gap"],

         [[FIG_LABELS[i+1], round(f3n[i],2), round(f3g[i],2)] for i in range(10)])

save_csv(os.path.join(OUT_DIR, "fig5_method2_z03.csv"),

         ["Rate","base_nRMSE","base_Gap","prop_nRMSE","prop_Gap"],

         [[FIG_RATES[i], round(f5_bn[i],2), round(f5_bg[i],2),

           round(f5_pn[i],2), round(f5_pg[i],2)] for i in range(16)])

save_csv(os.path.join(OUT_DIR, "fig6_method2_z03.csv"),

         ["Label","nRMSE","Gap"],

         [[FIG_LABELS[i+1], round(f6n[i],2), round(f6g[i],2)] for i in range(10)])

save_csv(os.path.join(OUT_DIR, "fig8_method2_z03.csv"),

         ["Rate","base_nRMSE","base_Gap","prop_nRMSE","prop_Gap"],

         [[FIG_RATES[i], round(f8_bn[i],2), round(f8_bg[i],2),

           round(f8_pn[i],2), round(f8_pg[i],2)] for i in range(16)])





# =====================================================================

# 12. Fig.3 — AR, dual y축

# =====================================================================

print("\n" + "=" * 70)

print("  8. Fig.3 plotting (AR, W1/W2, dual y)")

print("=" * 70)



x11 = list(range(11))

fig, al = plt.subplots(figsize=(10, 6))

fig.suptitle(f"Fig.3 — 방법2(PC=rate·(DA+RT)), AR, rate={FIG36_RATE}", fontsize=13)

ar_ax = al.twinx()



l1 = al.axhline(ar_nrmse, color=C_AR_BASE, lw=2, alpha=0.5, label="AR baseline nRMSE")

l2 = al.plot(x11[1:], f3n, marker="o", color=C_AR_PROP, lw=2, label="제안모형 nRMSE")

l3 = ar_ax.axhline(ar_bg, color=C_AR_BASE, lw=2, alpha=0.5, ls="--", label="AR baseline Gap")

l4 = ar_ax.plot(x11[1:], f3g, marker="s", color=C_AR_PROP, lw=2, ls="--", label="제안모형 Gap")

l5 = al.axhline(PAPER_AR_NRMSE, color=C_PAPER, lw=1.5, alpha=0.8, ls=":", label="논문 AR nRMSE")

l6 = ar_ax.axhline(PAPER_AR_GAP, color=C_PAPER, lw=1.5, alpha=0.8, ls="-.", label="논문 AR Gap")



al.set_xticks(x11); al.set_xticklabels(FIG_LABELS)

al.set_xlabel("W1/W2"); al.set_ylabel("nRMSE (%)", color=C_AR_BASE)

ar_ax.set_ylabel("Optimality Gap (%)", color=C_AR_PROP)

al.tick_params(axis="y", labelcolor=C_AR_BASE)

ar_ax.tick_params(axis="y", labelcolor=C_AR_PROP)

al.grid(True, alpha=0.3, color=C_GRID)

al.legend([l1]+l2+[l3,l4]+[l5,l6],

          ["AR baseline nRMSE","제안모형 nRMSE","AR baseline Gap","제안모형 Gap",

           "논문 AR nRMSE","논문 AR Gap"],

          loc="upper left", fontsize=9)

fig.tight_layout()

p3 = os.path.join(RESULTS_DIR, "fig3_method2_pc_sum_ar_z03.png")

fig.savefig(p3, dpi=150); plt.close(fig)

print(f"  saved: {p3}")





# =====================================================================

# 13. Fig.5 — AR, rate sweep (2분할)

# =====================================================================

print("\n" + "=" * 70)

print("  9. Fig.5 plotting (AR, rate, 2분할)")

print("=" * 70)



x16 = list(range(16))

lbl = [f"{int(r*100)}%" for r in FIG_RATES]

hl = FIG_RATES.index(KPI_RATE)



f5, (a5n, a5g) = plt.subplots(1, 2, figsize=(13, 5))

f5.suptitle(f"Fig.5 — 방법2, AR, W1=W2=1, rate 스윕", fontsize=13)



a5n.plot(x16, f5_bn, marker="o", color=C_AR_BASE, lw=2, label="AR baseline nRMSE")

a5n.plot(x16, f5_pn, marker="s", color=C_AR_PROP, lw=2, label="제안모형 nRMSE")

a5n.axhline(PAPER_AR_NRMSE, color=C_PAPER, lw=1.5, alpha=0.8, ls=":", label="논문 AR nRMSE")

a5n.set_xticks(x16); a5n.set_xticklabels(lbl, rotation=45)

a5n.set_xlabel("벌금비용률"); a5n.set_ylabel("nRMSE (%)")

a5n.set_title("nRMSE"); a5n.grid(True, alpha=0.3, color=C_GRID); a5n.legend(fontsize=9)



a5g.plot(x16, f5_bg, marker="o", color=C_AR_BASE, lw=2, label="AR baseline Gap")

a5g.plot(x16, f5_pg, marker="s", color=C_AR_PROP, lw=2, label="제안모형 Gap")

a5g.axhline(PAPER_AR_GAP, color=C_PAPER, lw=1.5, alpha=0.8, ls=":", label="논문 AR Gap")

a5g.axvline(hl, color=C_GRID, ls=":", lw=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")

a5g.set_xticks(x16); a5g.set_xticklabels(lbl, rotation=45)

a5g.set_xlabel("벌금비용률"); a5g.set_ylabel("Optimality Gap (%)")

a5g.set_title("Optimality Gap"); a5g.grid(True, alpha=0.3, color=C_GRID); a5g.legend(fontsize=9)



f5.tight_layout()

p5 = os.path.join(RESULTS_DIR, "fig5_method2_pc_sum_ar_z03.png")

f5.savefig(p5, dpi=150); plt.close(f5)

print(f"  saved: {p5}")





# =====================================================================

# 14. Fig.6 — MLR, dual y

# =====================================================================

print("\n" + "=" * 70)

print("  10. Fig.6 plotting (MLR, W1/W2, dual y)")

print("=" * 70)



fig6, al = plt.subplots(figsize=(10, 6))

fig6.suptitle(f"Fig.6 — 방법2(PC=rate·(DA+RT)), MLR, rate={FIG36_RATE}", fontsize=13)

ar_ax = al.twinx()



l1 = al.axhline(mlr_nrmse, color=C_MLR_BASE, lw=2, alpha=0.5, label="MLR baseline nRMSE")

l2 = al.plot(x11[1:], f6n, marker="o", color=C_MLR_PROP, lw=2, label="제안모형 nRMSE")

l3 = ar_ax.axhline(mlr_bg, color=C_MLR_BASE, lw=2, alpha=0.5, ls="--", label="MLR baseline Gap")

l4 = ar_ax.plot(x11[1:], f6g, marker="s", color=C_MLR_PROP, lw=2, ls="--", label="제안모형 Gap")

l5 = al.axhline(PAPER_MLR_NRMSE, color=C_PAPER, lw=1.5, alpha=0.8, ls=":", label="논문 MLR nRMSE")

l6 = ar_ax.axhline(PAPER_MLR_GAP, color=C_PAPER, lw=1.5, alpha=0.8, ls="-.", label="논문 MLR Gap")



al.set_xticks(x11); al.set_xticklabels(FIG_LABELS)

al.set_xlabel("W1/W2"); al.set_ylabel("nRMSE (%)", color=C_MLR_BASE)

ar_ax.set_ylabel("Optimality Gap (%)", color=C_MLR_PROP)

al.tick_params(axis="y", labelcolor=C_MLR_BASE)

ar_ax.tick_params(axis="y", labelcolor=C_MLR_PROP)

al.grid(True, alpha=0.3, color=C_GRID)

al.legend([l1]+l2+[l3,l4]+[l5,l6],

          ["MLR baseline nRMSE","제안모형 nRMSE","MLR baseline Gap","제안모형 Gap",

           "논문 MLR nRMSE","논문 MLR Gap"],

          loc="upper left", fontsize=9)

fig6.tight_layout()

p6 = os.path.join(RESULTS_DIR, "fig6_method2_pc_sum_mlr_z03.png")

fig6.savefig(p6, dpi=150); plt.close(fig6)

print(f"  saved: {p6}")





# =====================================================================

# 15. Fig.8 — MLR, rate sweep (2분할)

# =====================================================================

print("\n" + "=" * 70)

print("  11. Fig.8 plotting (MLR, rate, 2분할)")

print("=" * 70)



f8, (a8n, a8g) = plt.subplots(1, 2, figsize=(13, 5))

f8.suptitle(f"Fig.8 — 방법2, MLR, W1=W2=1, rate 스윕", fontsize=13)



a8n.plot(x16, f8_bn, marker="o", color=C_MLR_BASE, lw=2, label="MLR baseline nRMSE")

a8n.plot(x16, f8_pn, marker="s", color=C_MLR_PROP, lw=2, label="제안모형 nRMSE")

a8n.axhline(PAPER_MLR_NRMSE, color=C_PAPER, lw=1.5, alpha=0.8, ls=":", label="논문 MLR nRMSE")

a8n.set_xticks(x16); a8n.set_xticklabels(lbl, rotation=45)

a8n.set_xlabel("벌금비용률"); a8n.set_ylabel("nRMSE (%)")

a8n.set_title("nRMSE"); a8n.grid(True, alpha=0.3, color=C_GRID); a8n.legend(fontsize=9)



a8g.plot(x16, f8_bg, marker="o", color=C_MLR_BASE, lw=2, label="MLR baseline Gap")

a8g.plot(x16, f8_pg, marker="s", color=C_MLR_PROP, lw=2, label="제안모형 Gap")

a8g.axhline(PAPER_MLR_GAP, color=C_PAPER, lw=1.5, alpha=0.8, ls=":", label="논문 MLR Gap")

a8g.axvline(hl, color=C_GRID, ls=":", lw=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")

a8g.set_xticks(x16); a8g.set_xticklabels(lbl, rotation=45)

a8g.set_xlabel("벌금비용률"); a8g.set_ylabel("Optimality Gap (%)")

a8g.set_title("Optimality Gap"); a8g.grid(True, alpha=0.3, color=C_GRID); a8g.legend(fontsize=9)



f8.tight_layout()

p8 = os.path.join(RESULTS_DIR, "fig8_method2_pc_sum_mlr_z03.png")

f8.savefig(p8, dpi=150); plt.close(f8)

print(f"  saved: {p8}")



print("\n" + "=" * 70)

print("  완료 — 방법2 PC=rate*(DA+RT) AR+MLR 통합 (z03/블록18)")

print("=" * 70)

print(f"  Fig.3: {p3}")

print(f"  Fig.5: {p5}")

print(f"  Fig.6: {p6}")

print(f"  Fig.8: {p8}")

print(f"  Grid:  {gc}")

