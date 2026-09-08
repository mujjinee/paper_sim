# -*- coding: utf-8 -*-
# =====================================================================
# integrated_pc_da_plus_rt_fig3568_AR_MLR.py
#
# PC = rate·(DA + RT) 통합 실행
#   - 이익함수: DA·x + RP·y+ - PC·y-  (3항, 논문 원문)
#   - PC = rate·(DA + RT)  (방법 2)
#   - oracle: 3후보 {0, actual, 1.0}
#   - AR(시간대별 12개 모델) + MLR(pooled 1개 모델)
#   - 대상: z03/블록18 (2013년 가을)
#
# 출력: Fig.3(AR W1/W2), Fig.5(AR rate), Fig.6(MLR W1/W2), Fig.8(MLR rate)
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
# 0. 한글 폰트 + 색상
# =====================================================================
korean_font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic"]
for font in fm.fontManager.ttflist:
    if font.name in korean_font_candidates:
        plt.rcParams["font.family"] = font.name
        break
plt.rcParams["axes.unicode_minus"] = False

C_AR     = "#4C72B0"
C_MLR    = "#CC4654"
C_PROP   = "#55A868"
C_PAPER  = "#FF7F00"
C_GRID   = "#DDDDDD"

# =====================================================================
# 1. 설정
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z03.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "simulation_output")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOCAL_HOUR_START, LOCAL_HOUR_END, HOURS_PER_DAY = 9, 21, 12

TRAIN_START  = pd.Timestamp("2013-08-25")
TRAIN_END    = pd.Timestamp("2013-11-22")
TEST_START   = pd.Timestamp("2013-11-23")
TEST_END     = pd.Timestamp("2013-12-22")
HISTORY_DATE = pd.Timestamp("2013-08-24")

CAPACITY_MW = 30.0; DURATION_HOURS = 1.0
scale = CAPACITY_MW * DURATION_HOURS

KPI_RATE = 0.5

W_RATIOS = [(1,20),(1,10),(1,5),(1,2),(1,1)]
W_LABELS = ["1/20","1/10","1/5","1/2","1/1"]
RATES    = [0.5, 0.6, 0.7, 0.8]

# 논문 기준값 (4항식, z03/블럭18)
PAPER_AR_BASE_NRMSE = 48.84;   PAPER_AR_BASE_GAP   = 16.48
PAPER_MLR_BASE_NRMSE = 39.73;  PAPER_MLR_BASE_GAP   = 14.09


# =====================================================================
# 2. 데이터 로딩
# =====================================================================
print("=" * 70)
print("  1. 데이터 로딩")
print("=" * 70)
raw = pd.read_csv(MERGED_FILE); raw["local_date"] = pd.to_datetime(raw["local_date"])
day = raw[(raw["local_hour"]>=LOCAL_HOUR_START)&(raw["local_hour"]<LOCAL_HOUR_END)].copy()
day["hour_idx"] = day["local_hour"] - LOCAL_HOUR_START

# AR 배열
his = day[day["local_date"]==HISTORY_DATE].sort_values("hour_idx")
history_solar = np.zeros((1, HOURS_PER_DAY))
for _, r in his.iterrows(): history_solar[0, r["hour_idx"]] = r["solar_power"]

tr = day[(day["local_date"]>=TRAIN_START)&(day["local_date"]<=TRAIN_END)].sort_values(["local_date","hour_idx"])
te = day[(day["local_date"]>=TEST_START)&(day["local_date"]<=TEST_END)].sort_values(["local_date","hour_idx"])

n_tr_days = len(sorted(tr["local_date"].unique()))
n_te_days = len(sorted(te["local_date"].unique()))
n_tr_obs = len(tr); n_te_obs = len(te)

tr_s = np.zeros((n_tr_days, HOURS_PER_DAY)); tr_d = np.zeros_like(tr_s); tr_r = np.zeros_like(tr_s)
for _, r in tr.iterrows():
    d = (r["local_date"]-TRAIN_START).days; h = r["hour_idx"]
    tr_s[d,h]=r["solar_power"]; tr_d[d,h]=r["da_price"]; tr_r[d,h]=r["rt_price"]

te_s = np.zeros((n_te_days, HOURS_PER_DAY)); te_d = np.zeros_like(te_s); te_r = np.zeros_like(te_s)
for _, r in te.iterrows():
    d = (r["local_date"]-TEST_START).days; h = r["hour_idx"]
    te_s[d,h]=r["solar_power"]; te_d[d,h]=r["da_price"]; te_r[d,h]=r["rt_price"]

# MLR 배열
mlr_tr_s = tr["solar_power"].to_numpy(); mlr_tr_d = tr["da_price"].to_numpy(); mlr_tr_r = tr["rt_price"].to_numpy()
mlr_te_s = te["solar_power"].to_numpy(); mlr_te_d = te["da_price"].to_numpy(); mlr_te_r = te["rt_price"].to_numpy()

n_feat_mlr = 4
X_tr = np.column_stack([np.ones(n_tr_obs), tr["dssrd"].to_numpy(), tr["dtsr"].to_numpy(), tr["hour_idx"].to_numpy(dtype=float)])
X_te = np.column_stack([np.ones(n_te_obs), te["dssrd"].to_numpy(), te["dtsr"].to_numpy(), te["hour_idx"].to_numpy(dtype=float)])

# AR 설계행렬
n_feat_ar = 13
ht = np.vstack([history_solar, tr_s])
ar_int = np.ones((n_tr_days, 1))
ar_lag = np.zeros((n_tr_days, HOURS_PER_DAY))
for d in range(n_tr_days): ar_lag[d] = ht[d][::-1]
ar_X = np.hstack([ar_int, ar_lag])

ar_act = te_s.flatten(); ar_da = te_d.flatten(); ar_rt = te_r.flatten()
print(f"  train: {n_tr_days}일({n_tr_obs}행), test: {n_te_days}일({n_te_obs}행)")


# =====================================================================
# 3. 유틸리티
# =====================================================================
def nrmse_fn(p, a):
    return 100.0 * np.sqrt(np.mean((a-p)**2)) / np.mean(a)

def chunk_matmul(A, b, chunk=100):
    n = A.shape[0]
    if n <= chunk: return A @ b
    out = np.empty(n)
    for i in range(0, n, chunk): out[i:i+chunk] = A[i:i+chunk] @ b
    return out

def compute_gap(pc_rate, pred, actual, da, rt):
    """PC = rate*(DA + RT) 기반 optimality gap"""
    s_r = 0.0; s_o = 0.0
    for i in range(len(pred)):
        a=actual[i]; x=pred[i]; dp=da[i]; rp=rt[i]
        pc = pc_rate * (dp + rp)
        m = a - x; yp = max(m, 0); ym = max(-m, 0)
        s_r += scale * (dp * x + rp * yp - pc * ym)
        p0 = scale * rp * a; pa = scale * dp * a
        s1 = max(a-1.0, 0); y1 = max(1.0-a, 0)
        p1 = scale * (dp*1.0 + rp*s1 - pc*y1)
        s_o += max(p0, pa, p1)
    return 100.0 * (s_o - s_r) / s_o if s_o > 1e-10 else 0.0


# =====================================================================
# 4. AR baseline (bounded LAD)
# =====================================================================
print("\n" + "=" * 70)
print("  2. AR baseline (bounded LAD)")
print("=" * 70)
ar_coeff = np.zeros((HOURS_PER_DAY, n_feat_ar))
for h in range(HOURS_PER_DAY):
    y_h = tr_s[:, h]
    Xs = sparse.csr_matrix(ar_X); In = sparse.eye(n_tr_days, format="csr")
    A = sparse.vstack([sparse.hstack([Xs, -In]), sparse.hstack([-Xs, -In])], format="csr")
    b = np.concatenate([y_h, -y_h])
    c = np.concatenate([np.zeros(n_feat_ar), np.ones(n_tr_days)/n_tr_days])
    vb = [(None,None)]*n_feat_ar + [(0.0,None)]*n_tr_days
    ar_coeff[h] = linprog(c, A_ub=A, b_ub=b, bounds=vb, method="highs").x[:n_feat_ar]

ar_fc = np.zeros((n_te_days, HOURS_PER_DAY)); prev = tr_s[-1]
for d in range(n_te_days):
    fv = np.concatenate([[1.0], prev[::-1]])
    for h in range(HOURS_PER_DAY): ar_fc[d,h] = np.clip(np.dot(ar_coeff[h], fv), 0, 1)
    prev = te_s[d]
ar_pred = ar_fc.flatten()
ar_nrmse = nrmse_fn(ar_pred, ar_act)
print(f"  AR baseline nRMSE = {ar_nrmse:.2f}%")


# =====================================================================
# 5. MLR baseline (bounded LAD, pooled)
# =====================================================================
print("\n" + "=" * 70)
print("  3. MLR baseline (bounded LAD, pooled)")
print("=" * 70)
Xs = sparse.csr_matrix(X_tr); In = sparse.eye(n_tr_obs, format="csr")
A = sparse.vstack([sparse.hstack([Xs, -In]), sparse.hstack([-Xs, -In])], format="csr")
b = np.concatenate([mlr_tr_s, -mlr_tr_s])
c = np.concatenate([np.zeros(n_feat_mlr), np.ones(n_tr_obs)/n_tr_obs])
vb = [(None,None)]*n_feat_mlr + [(0.0,None)]*n_tr_obs
mlr_coeff = linprog(c, A_ub=A, b_ub=b, bounds=vb, method="highs").x[:n_feat_mlr]
mlr_pred = np.clip(chunk_matmul(X_te, mlr_coeff), 0, 1)
mlr_nrmse = nrmse_fn(mlr_pred, mlr_te_s)
print(f"  MLR baseline nRMSE = {mlr_nrmse:.2f}%")


# =====================================================================
# 6. 제안모형 MILP — AR (PC = rate*(DA+RT), 정규화 제거 + 등식 제약)
# =====================================================================
_cache = {}

def solve_ar(pc_rate, W1, W2):
    k = (pc_rate, W1, W2, "ar")
    if k in _cache: return _cache[k]
    coeffs = np.zeros((HOURS_PER_DAY, n_feat_ar))
    for hour in range(HOURS_PER_DAY):
        y_h = tr_s[:, hour]; da_h = tr_d[:, hour]; rt_h = tr_r[:, hour]
        n_obs = n_tr_days

        sc = np.zeros(n_obs); yc = np.zeros(n_obs)
        for i in range(n_obs):
            pc = pc_rate * (da_h[i] + rt_h[i])
            sc[i] = (-W1 * rt_h[i]) + W2        # 논문 Eq.(9a), 정규화 제거
            yc[i] = (W1 * pc) + W2              # 논문 Eq.(9a), 정규화 제거

        bl = [i for i in range(n_obs) if sc[i]+yc[i] < 0]
        ba = np.array(bl, dtype=int); nb = len(ba)
        bs=0; xs=n_feat_ar; yps=n_feat_ar+n_obs; yms=n_feat_ar+2*n_obs; zs=n_feat_ar+3*n_obs
        nv = n_feat_ar + 3*n_obs + nb

        obj = np.zeros(nv)
        for i in range(n_obs):
            obj[xs+i] = -W1 * da_h[i]           # 정규화 제거
            obj[yps+i] = sc[i]; obj[yms+i] = yc[i]

        Xs = sparse.csr_matrix(ar_X); In = sparse.eye(n_obs, format="csr")
        ea = sparse.lil_matrix((n_obs, nv))
        ea[:, bs:bs+n_feat_ar] = -Xs; ea[:, xs:xs+n_obs] = In
        eb = sparse.lil_matrix((n_obs, nv))
        eb[:, xs:xs+n_obs] = In; eb[:, yps:yps+n_obs] = In; eb[:, yms:yms+n_obs] = -In
        ae = sparse.vstack([ea, eb], format="csr")
        rhs = np.concatenate([np.zeros(n_obs), y_h])
        con = [LinearConstraint(ae, rhs, rhs)]

        if nb > 0:
            cm = sparse.lil_matrix((2*nb, nv))
            for k2 in range(nb):
                r = ba[k2]; cm[k2, yps+r]=1.0; cm[k2, zs+k2]=1.0
                cm[nb+k2, yms+r]=1.0; cm[nb+k2, zs+k2]=-1.0
            con.append(LinearConstraint(cm.tocsr(), np.full(2*nb,-np.inf),
                          np.concatenate([np.ones(nb), np.zeros(nb)])))

        lb = np.concatenate([np.full(n_feat_ar,-np.inf), np.zeros(3*n_obs+nb)])
        ub = np.concatenate([np.full(n_feat_ar,np.inf), np.ones(3*n_obs+nb)])
        ig = np.zeros(nv, dtype=int); ig[zs:zs+nb] = 1
        res = milp(c=obj, integrality=ig, bounds=Bounds(lb, ub),
                   constraints=con, options={"mip_rel_gap":1e-9, "time_limit":300})
        if not res.success:
            print(f"    [경고] AR h={hour}: {res.message} -> baseline 사용")
            coeffs[hour] = ar_coeff[hour]; continue
        coeffs[hour] = res.x[bs:bs+n_feat_ar]

    fc = np.zeros((n_te_days, HOURS_PER_DAY)); prev = tr_s[-1]
    for d in range(n_te_days):
        fv = np.concatenate([[1.0], prev[::-1]])
        for h in range(HOURS_PER_DAY): fc[d,h] = np.clip(np.dot(coeffs[h], fv), 0, 1)
        prev = te_s[d]
    pred = fc.flatten()
    val = (nrmse_fn(pred, ar_act), compute_gap(pc_rate, pred, ar_act, ar_da, ar_rt))
    _cache[k] = val; return val


# =====================================================================
# 7. 제안모형 MILP — MLR (PC = rate*(DA+RT), 정규화 제거 + 등식 제약)
# =====================================================================
def solve_mlr(pc_rate, W1, W2):
    k = (pc_rate, W1, W2, "mlr")
    if k in _cache: return _cache[k]
    n_obs = n_tr_obs

    sc = np.zeros(n_obs); yc = np.zeros(n_obs)
    for i in range(n_obs):
        pc = pc_rate * (mlr_tr_d[i] + mlr_tr_r[i])
        sc[i] = (-W1 * mlr_tr_r[i]) + W2
        yc[i] = (W1 * pc) + W2

    bl = [i for i in range(n_obs) if sc[i]+yc[i] < 0]
    ba = np.array(bl, dtype=int); nb = len(ba)
    bs=0; xs=n_feat_mlr; yps=n_feat_mlr+n_obs; yms=n_feat_mlr+2*n_obs; zs=n_feat_mlr+3*n_obs
    nv = n_feat_mlr + 3*n_obs + nb

    obj = np.zeros(nv)
    for i in range(n_obs):
        obj[xs+i] = -W1 * mlr_tr_d[i]
        obj[yps+i] = sc[i]; obj[yms+i] = yc[i]

    Xs = sparse.csr_matrix(X_tr); In = sparse.eye(n_obs, format="csr")
    ea = sparse.lil_matrix((n_obs, nv))
    ea[:, bs:bs+n_feat_mlr] = -Xs; ea[:, xs:xs+n_obs] = In
    eb = sparse.lil_matrix((n_obs, nv))
    eb[:, xs:xs+n_obs] = In; eb[:, yps:yps+n_obs] = In; eb[:, yms:yms+n_obs] = -In
    ae = sparse.vstack([ea, eb], format="csr")
    rhs = np.concatenate([np.zeros(n_obs), mlr_tr_s])
    con = [LinearConstraint(ae, rhs, rhs)]

    if nb > 0:
        cm = sparse.lil_matrix((2*nb, nv))
        for k2 in range(nb):
            r = ba[k2]; cm[k2, yps+r]=1.0; cm[k2, zs+k2]=1.0
            cm[nb+k2, yms+r]=1.0; cm[nb+k2, zs+k2]=-1.0
        con.append(LinearConstraint(cm.tocsr(), np.full(2*nb,-np.inf),
                      np.concatenate([np.ones(nb), np.zeros(nb)])))

    lb = np.concatenate([np.full(n_feat_mlr,-np.inf), np.zeros(3*n_obs+nb)])
    ub = np.concatenate([np.full(n_feat_mlr,np.inf), np.ones(3*n_obs+nb)])
    ig = np.zeros(nv, dtype=int); ig[zs:zs+nb] = 1
    res = milp(c=obj, integrality=ig, bounds=Bounds(lb, ub),
               constraints=con, options={"mip_rel_gap":1e-9, "time_limit":300})
    if not res.success:
        print(f"    [경고] MLR: {res.message} -> baseline 사용")
        pred = np.clip(chunk_matmul(X_te, mlr_coeff), 0, 1)
    else:
        pred = np.clip(chunk_matmul(X_te, res.x[bs:bs+n_feat_mlr]), 0, 1)
    val = (nrmse_fn(pred, mlr_te_s), compute_gap(pc_rate, pred, mlr_te_s, mlr_te_d, mlr_te_r))
    _cache[k] = val; return val


# =====================================================================
# 8. Fig3/6 데이터 — W1/W2 10개점 (rate=0.5)
# =====================================================================
print("\n" + "=" * 70)
print("  4. Fig.3/6 데이터 — W1/W2 10개점 (rate=" + str(KPI_RATE) + ")")
print("=" * 70)

f3n=[]; f3g=[]; f6n=[]; f6g=[]
for w1, w2 in W_RATIOS:
    print(f"  {w1}/{w2}: 계산 중...", end=" ", flush=True)
    an, ag = solve_ar(KPI_RATE, w1, w2)
    mn, mg = solve_mlr(KPI_RATE, w1, w2)
    f3n.append(an); f3g.append(ag); f6n.append(mn); f6g.append(mg)
    print(f"AR(nRMSE={an:.2f}%,Gap={ag:.2f}%) MLR(nRMSE={mn:.2f}%,Gap={mg:.2f}%)")


# =====================================================================
# 9. Fig5/8 데이터 — rate 0~150% (W1=W2=1)
# =====================================================================
print("\n" + "=" * 70)
print("  5. Fig.5/8 데이터 — rate 0~150% (W1=W2=1)")
print("=" * 70)

f5_pn=[]; f5_pg=[]; f8_pn=[]; f8_pg=[]
for rate in RATES:
    print(f"  rate={rate:.1f}: 계산 중...", end=" ", flush=True)
    an, ag = solve_ar(rate, 1, 1)
    mn, mg = solve_mlr(rate, 1, 1)
    f5_pn.append(an); f5_pg.append(ag)
    f8_pn.append(mn); f8_pg.append(mg)
    print(f"AR(nRMSE={an:.2f}%,Gap={ag:.2f}%) MLR(nRMSE={mn:.2f}%,Gap={mg:.2f}%)")


# =====================================================================
# 논문 KPI 조건 — W1=1, W2=20, penalty rate=50%
# =====================================================================
print("\n" + "=" * 70)
print("  논문 KPI 조건 — W1=1, W2=20, penalty rate=50%")
print("=" * 70)

ar_kpi_gap_base = compute_gap(KPI_RATE, ar_pred, ar_act, ar_da, ar_rt)
mlr_kpi_gap_base = compute_gap(KPI_RATE, mlr_pred, mlr_te_s, mlr_te_d, mlr_te_r)

print(f"")
print(f"  +-------------+---------------+---------------+---------------+---------------+")
print(f"  |     모델     |  nRMSE (%)     |   Gap (%)     |  nRMSE (%)     |   Gap (%)     |")
print(f"  |             |   제안모형     |    제안모형    |   baseline     |   baseline     |")
print(f"  +-------------+---------------+---------------+---------------+---------------+")
print(f"  | AR          | {f3n[0]:13.2f} | {f3g[0]:13.2f} | {ar_nrmse:13.2f} | {ar_kpi_gap_base:13.2f} |")
print(f"  | MLR         | {f6n[0]:13.2f} | {f6g[0]:13.2f} | {mlr_nrmse:13.2f} | {mlr_kpi_gap_base:13.2f} |")
print(f"  +-------------+---------------+---------------+---------------+---------------+")
print(f"")
print(f"  ※ 논문 참고값 (z03/블록18): AR nRMSE=34.76%, Gap=15.04% / MLR nRMSE=21.76%, Gap=12.59%")
print(f"")


# ── CSV 저장 ──
def save_csv(path, hdr, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        for r in rows: w.writerow(r)
    print(f"  saved: {path}")

save_csv(os.path.join(OUT_DIR, "fig3_pc_da_plus_rt_AR.csv"), ["Label","nRMSE","Gap"],
         [[W_LABELS[i], round(f3n[i],2), round(f3g[i],2)] for i in range(len(W_LABELS))])
save_csv(os.path.join(OUT_DIR, "fig5_pc_da_plus_rt_AR.csv"), ["Rate","nRMSE","Gap"],
         [[RATES[i], round(f5_pn[i],2), round(f5_pg[i],2)] for i in range(len(RATES))])
save_csv(os.path.join(OUT_DIR, "fig6_pc_da_plus_rt_MLR.csv"), ["Label","nRMSE","Gap"],
         [[W_LABELS[i], round(f6n[i],2), round(f6g[i],2)] for i in range(len(W_LABELS))])
save_csv(os.path.join(OUT_DIR, "fig8_pc_da_plus_rt_MLR.csv"), ["Rate","nRMSE","Gap"],
         [[RATES[i], round(f8_pn[i],2), round(f8_pg[i],2)] for i in range(len(RATES))])


# =====================================================================
# 10. Fig.3 — AR, W1/W2 스윕 (논문 기준 + 제안모형)
# =====================================================================
print("\n" + "=" * 70)
print("  6. Fig.3 그리기")
print("=" * 70)

x_w = list(range(len(W_LABELS)))
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Fig.3 — AR, W1/W2 스윕 (rate=0.5, PC=rate*(DA+RT))", fontsize=13)
ax_r = ax.twinx()

ax.axhline(PAPER_AR_BASE_NRMSE, color=C_PAPER, lw=2.5, label="논문 nRMSE", alpha=0.7)
ax_r.axhline(PAPER_AR_BASE_GAP, color=C_PAPER, lw=2.5, ls="--", label="논문 Gap", alpha=0.7)
ax.plot(x_w, f3n, marker="o", color=C_PROP, lw=2.5, label="제안모형 nRMSE")
ax_r.plot(x_w, f3g, marker="s", color=C_PROP, lw=2.5, ls="--", label="제안모형 Gap")

ax.set_xticks(x_w); ax.set_xticklabels(W_LABELS)
ax.set_xlabel("W1/W2"); ax.set_ylabel("nRMSE (%)", color=C_PAPER)
ax_r.set_ylabel("Optimality Gap (%)", color=C_PROP)
ax.tick_params(axis="y", labelcolor=C_PAPER); ax_r.tick_params(axis="y", labelcolor=C_PROP)
ax.grid(True, alpha=0.3, color=C_GRID)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax_r.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc="upper left", fontsize=10)
fig.tight_layout()
p3 = os.path.join(RESULTS_DIR, "fig3_pc_da_plus_rt_W1W2_AR.png")
fig.savefig(p3, dpi=150); plt.close(fig)
print(f"  saved: {p3}")


# =====================================================================
# 11. Fig.5 — AR, rate 스윕 (2분할)
# =====================================================================
print("\n" + "=" * 70)
print("  7. Fig.5 그리기")
print("=" * 70)

x_rates = list(range(len(RATES))); lbl = [f"{int(r*100)}%" for r in RATES]; hl = RATES.index(KPI_RATE)
fig5, (a5n, a5g) = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle("Fig.5 — AR, rate 스윕 (W1=W2=1, PC=rate*(DA+RT))", fontsize=13)

a5n.axhline(PAPER_AR_BASE_NRMSE, color=C_PAPER, lw=2.5, label="논문 nRMSE", alpha=0.7)
a5n.plot(x_rates, f5_pn, marker="s", color=C_PROP, lw=2.5, label="제안모형 nRMSE")
a5n.set_xticks(x_rates); a5n.set_xticklabels(lbl, rotation=45)
a5n.set_xlabel("벌금비용률"); a5n.set_ylabel("nRMSE (%)")
a5n.set_title("nRMSE"); a5n.grid(True, alpha=0.3, color=C_GRID); a5n.legend(fontsize=10)

a5g.axhline(PAPER_AR_BASE_GAP, color=C_PAPER, lw=2.5, ls="--", label="논문 Gap", alpha=0.7)
a5g.plot(x_rates, f5_pg, marker="s", color=C_PROP, lw=2.5, label="제안모형 Gap")
a5g.axvline(hl, color=C_GRID, ls=":", lw=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")
a5g.set_xticks(x_rates); a5g.set_xticklabels(lbl, rotation=45)
a5g.set_xlabel("벌금비용률"); a5g.set_ylabel("Optimality Gap (%)")
a5g.set_title("Optimality Gap"); a5g.grid(True, alpha=0.3, color=C_GRID); a5g.legend(fontsize=10)

fig5.tight_layout()
p5 = os.path.join(RESULTS_DIR, "fig5_pc_da_plus_rt_rate_AR.png")
fig5.savefig(p5, dpi=150); plt.close(fig5)
print(f"  saved: {p5}")


# =====================================================================
# 12. Fig.6 — MLR, W1/W2 스윕
# =====================================================================
print("\n" + "=" * 70)
print("  8. Fig.6 그리기")
print("=" * 70)

fig6, ax6 = plt.subplots(figsize=(10, 6))
fig6.suptitle("Fig.6 — MLR, W1/W2 스윕 (rate=0.5, PC=rate*(DA+RT))", fontsize=13)
ax6_r = ax6.twinx()

ax6.axhline(PAPER_MLR_BASE_NRMSE, color=C_PAPER, lw=2.5, label="논문 nRMSE", alpha=0.7)
ax6_r.axhline(PAPER_MLR_BASE_GAP, color=C_PAPER, lw=2.5, ls="--", label="논문 Gap", alpha=0.7)
ax6.plot(x_w, f6n, marker="o", color=C_PROP, lw=2.5, label="제안모형 nRMSE")
ax6_r.plot(x_w, f6g, marker="s", color=C_PROP, lw=2.5, ls="--", label="제안모형 Gap")

ax6.set_xticks(x_w); ax6.set_xticklabels(W_LABELS)
ax6.set_xlabel("W1/W2"); ax6.set_ylabel("nRMSE (%)", color=C_PAPER)
ax6_r.set_ylabel("Optimality Gap (%)", color=C_PROP)
ax6.tick_params(axis="y", labelcolor=C_PAPER); ax6_r.tick_params(axis="y", labelcolor=C_PROP)
ax6.grid(True, alpha=0.3, color=C_GRID)
h1, l1 = ax6.get_legend_handles_labels(); h2, l2 = ax6_r.get_legend_handles_labels()
ax6.legend(h1+h2, l1+l2, loc="upper left", fontsize=10)
fig6.tight_layout()
p6 = os.path.join(RESULTS_DIR, "fig6_pc_da_plus_rt_W1W2_MLR.png")
fig6.savefig(p6, dpi=150); plt.close(fig6)
print(f"  saved: {p6}")


# =====================================================================
# 13. Fig.8 — MLR, rate 스윕 (2분할)
# =====================================================================
print("\n" + "=" * 70)
print("  9. Fig.8 그리기")
print("=" * 70)

fig8, (a8n, a8g) = plt.subplots(1, 2, figsize=(13, 5))
fig8.suptitle("Fig.8 — MLR, rate 스윕 (W1=W2=1, PC=rate*(DA+RT))", fontsize=13)

a8n.axhline(PAPER_MLR_BASE_NRMSE, color=C_PAPER, lw=2.5, label="논문 nRMSE", alpha=0.7)
a8n.plot(x_rates, f8_pn, marker="s", color=C_PROP, lw=2.5, label="제안모형 nRMSE")
a8n.set_xticks(x_rates); a8n.set_xticklabels(lbl, rotation=45)
a8n.set_xlabel("벌금비용률"); a8n.set_ylabel("nRMSE (%)")
a8n.set_title("nRMSE"); a8n.grid(True, alpha=0.3, color=C_GRID); a8n.legend(fontsize=10)

a8g.axhline(PAPER_MLR_BASE_GAP, color=C_PAPER, lw=2.5, ls="--", label="논문 Gap", alpha=0.7)
a8g.plot(x_rates, f8_pg, marker="s", color=C_PROP, lw=2.5, label="제안모형 Gap")
a8g.axvline(hl, color=C_GRID, ls=":", lw=1.5, label=f"KPI rate={int(KPI_RATE*100)}%")
a8g.set_xticks(x_rates); a8g.set_xticklabels(lbl, rotation=45)
a8g.set_xlabel("벌금비용률"); a8g.set_ylabel("Optimality Gap (%)")
a8g.set_title("Optimality Gap"); a8g.grid(True, alpha=0.3, color=C_GRID); a8g.legend(fontsize=10)

fig8.tight_layout()
p8 = os.path.join(RESULTS_DIR, "fig8_pc_da_plus_rt_rate_MLR.png")
fig8.savefig(p8, dpi=150); plt.close(fig8)
print(f"  saved: {p8}")

print("\n" + "=" * 70)
print("  완료 — 제안모형 AR+MLR 그림 생성 (PC=rate*(DA+RT), z03/블록18)")
print("=" * 70)
print(f"  Fig.3 (AR W1/W2): {p3}")
print(f"  Fig.5 (AR rate):  {p5}")
print(f"  Fig.6 (MLR W1/W2): {p6}")
print(f"  Fig.8 (MLR rate): {p8}")