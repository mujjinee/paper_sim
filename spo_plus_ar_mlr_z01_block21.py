# -*- coding: utf-8 -*-

# =====================================================================

# spo_plus_ar_mlr_z01_block21.py

#

# 방법4: Smart SPO+ Loss (Elmachtoub & Grigas 2022)

#

# 4항식 이익함수(DA*x + RP*y+ - RP*y- - PC*y-) 기반 SPO+:

#   surplus_cost:  -W1·RT + W2/n          (과예측 손실)

#   shortage_cost: W1·RT + W1·PC·I[DA>RT] + W2/n   (SPO+ weight)

#

# 기존 4항식과 차이: shortage 시 PC 적용에 I[DA>RT] 조건 추가

#   - DA > RT: shortage 시 RT + PC 모두 손실 (동일)

#   - RT ≥ DA: shortage 시 RT만 손실 (PC 생략)

#

# 대상: z01/블럭21, AR+MLR 통합

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

# 0. 폰트 + 색상

# =====================================================================

korean_font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic"]

for font in fm.fontManager.ttflist:

    if font.name in korean_font_candidates:

        plt.rcParams["font.family"] = font.name

        break

plt.rcParams["axes.unicode_minus"] = False



C_AR_BASE  = "#4C72B0"

C_AR_PROP  = "#55A868"

C_MLR_BASE = "#CC4654"

C_MLR_PROP = "#8CAED6"

C_GRID     = "#DDDDDD"



# =====================================================================

# 1. 설정

# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MERGED_FILE = os.path.join(BASE_DIR, "merged_for_simulation_z01.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

OUT_DIR = os.path.join(RESULTS_DIR, "simulation_output")

os.makedirs(OUT_DIR, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)



LOCAL_HOUR_START, LOCAL_HOUR_END, HOURS_PER_DAY = 9, 21, 12

TRAIN_START  = pd.Timestamp("2013-01-26")

TRAIN_END    = pd.Timestamp("2013-04-25")

TEST_START   = pd.Timestamp("2013-04-26")

TEST_END     = pd.Timestamp("2013-05-25")

HISTORY_DATE = pd.Timestamp("2013-01-25")

CAPACITY_MW, DURATION_HOURS = 30.0, 1.0

scale = CAPACITY_MW * DURATION_HOURS



KPI_W1, KPI_W2, KPI_RATE = 1, 20, 0.5



GRID_W_RATIOS = [(1,20),(1,10),(1,5),(1,1),(1,0)]

GRID_RATES = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5]



FIG_W_RATIOS = [(1,20),(1,10),(1,5),(1,2),(1,1),

                (2,1),(5,1),(10,1),(20,1),(1,0)]

FIG_LABELS = ["AR/MLR","1/20","1/10","1/5","1/2",

              "1/1","2/1","5/1","10/1","20/1","1/0"]

FIG_RATES = [round(0.1*i, 1) for i in range(16)]





# =====================================================================

# 2. 데이터 로딩

# =====================================================================

print("="*70)

print("  1. 데이터 로딩")

print("="*70)

raw = pd.read_csv(MERGED_FILE)

raw["local_date"] = pd.to_datetime(raw["local_date"])

dl = raw[(raw["local_hour"]>=9)&(raw["local_hour"]<21)].copy()

dl["hour_idx"] = dl["local_hour"] - 9



# AR: (날짜 x 12)

hr = dl[dl["local_date"]==HISTORY_DATE].sort_values("hour_idx")

hist_sol = np.zeros((1, 12))

for _, r in hr.iterrows(): hist_sol[0, r["hour_idx"]] = r["solar_power"]



tr = dl[(dl["local_date"]>=TRAIN_START)&(dl["local_date"]<=TRAIN_END)].sort_values(["local_date","hour_idx"])

te = dl[(dl["local_date"]>=TEST_START)&(dl["local_date"]<=TEST_END)].sort_values(["local_date","hour_idx"])



tds = sorted(tr["local_date"].unique())

nds = len(tds)

tes = sorted(te["local_date"].unique())

nds_t = len(tes)



ts = np.zeros((nds,12)); tda = np.zeros((nds,12)); trt = np.zeros((nds,12))

for _, r in tr.iterrows():

    d=(r["local_date"]-TRAIN_START).days; h=r["hour_idx"]

    ts[d,h]=r["solar_power"]; tda[d,h]=r["da_price"]; trt[d,h]=r["rt_price"]



tss = np.zeros((nds_t,12)); tdsa = np.zeros((nds_t,12)); trts = np.zeros((nds_t,12))

for _, r in te.iterrows():

    d=(r["local_date"]-TEST_START).days; h=r["hour_idx"]

    tss[d,h]=r["solar_power"]; tdsa[d,h]=r["da_price"]; trts[d,h]=r["rt_price"]



# MLR flat

ntr = len(tr); nte = len(te)

m_ts = tr["solar_power"].to_numpy(); m_da = tr["da_price"].to_numpy(); m_rt = tr["rt_price"].to_numpy()

m_xtr = np.column_stack([np.ones(ntr), tr["dssrd"].to_numpy(), tr["dtsr"].to_numpy(), tr["hour_idx"].to_numpy(float)])

m_act = te["solar_power"].to_numpy(); m_da_t = te["da_price"].to_numpy(); m_rt_t = te["rt_price"].to_numpy()

m_xte = np.column_stack([np.ones(nte), te["dssrd"].to_numpy(), te["dtsr"].to_numpy(), te["hour_idx"].to_numpy(float)])



# AR design

nfa = 13

ht = np.vstack([hist_sol, ts])

ar_int = np.ones((nds,1))

ar_lag = np.zeros((nds,12))

for d in range(nds): ar_lag[d] = ht[d][::-1]

ar_des = np.hstack([ar_int, ar_lag])



ar_act = tss.flatten(); ar_da = tdsa.flatten(); ar_rt = trts.flatten()



print(f"  train: {nds}일({ntr}행), test: {nds_t}일({nte}행)")





# =====================================================================

# 3. Gap 계산 — 4항식, PC=rate*DA, oracle 3후보

# =====================================================================

def comp_gap(pr, pred, act, da, rt):

    sr=0.0; so=0.0

    for i in range(len(pred)):

        a=act[i]; x=pred[i]; dp=da[i]; rp=rt[i]

        pc = pr * dp

        m=a-x; yp=max(m,0); ym=max(-m,0)

        sr += scale*(dp*x + rp*yp - rp*ym - pc*ym)

        p0=scale*rp*a; pa=scale*dp*a

        s1=max(a-1,0); y1=max(1-a,0)

        p1=scale*(dp+rp*s1-rp*y1-pc*y1)

        so += max(p0,pa,p1)

    return 100*(so-sr)/so if so>1e-10 else 0.0



def nrmse_f(p, a): return 100*np.sqrt(np.mean((a-p)**2))/np.mean(a)





# =====================================================================

# 4-5. Baseline

# =====================================================================

print("\n"+"="*70)

print("  2. AR + MLR baseline (bounded LAD)")

print("="*70)



# AR

ar_c = np.zeros((12, nfa))

for h in range(12):

    yh=ts[:,h]; Xs=sparse.csr_matrix(ar_des); In=sparse.eye(nds,format="csr")

    A=sparse.vstack([sparse.hstack([Xs,-In]),sparse.hstack([-Xs,-In])],format="csr")

    b=np.concatenate([yh,-yh]); c=np.concatenate([np.zeros(nfa),np.ones(nds)/nds])

    vb=[(None,None)]*nfa+[(0.0,None)]*nds

    ar_c[h]=linprog(c,A_ub=A,b_ub=b,bounds=vb,method="highs").x[:nfa]



ar_fc=np.zeros((nds_t,12)); pv=ts[-1]

for d in range(nds_t):

    fv=np.concatenate([[1.0],pv[::-1]])

    for h in range(12): ar_fc[d,h]=np.clip(ar_c[h]@fv,0,1)

    pv=tss[d]

ar_pred=ar_fc.flatten(); ar_nr=nrmse_f(ar_pred,ar_act)

print(f"  AR  baseline nRMSE = {ar_nr:.2f}%")



# MLR

nfm=4; Xs=sparse.csr_matrix(m_xtr); In=sparse.eye(ntr,format="csr")

A=sparse.vstack([sparse.hstack([Xs,-In]),sparse.hstack([-Xs,-In])],format="csr")

b=np.concatenate([m_ts,-m_ts]); c=np.concatenate([np.zeros(nfm),np.ones(ntr)/ntr])

vb=[(None,None)]*nfm+[(0.0,None)]*ntr

mlr_c=linprog(c,A_ub=A,b_ub=b,bounds=vb,method="highs").x[:nfm]

mlr_pred=np.clip(m_xte@mlr_c,0,1); mlr_nr=nrmse_f(mlr_pred,m_act)

print(f"  MLR baseline nRMSE = {mlr_nr:.2f}%")





# =====================================================================

# 6. AR SPO+ MILP

# =====================================================================

def solve_ar_spo(pr, W1, W2):

    co=np.zeros((12,nfa))

    for hour in range(12):

        yh=ts[:,hour]; dh=tda[:,hour]; rh=trt[:,hour]; no=nds



        oc=np.zeros(no)

        for i in range(no):

            a=yh[i]; dp=dh[i]; rp=rh[i]; pc=pr*dp

            p0=scale*rp*a; pa=scale*dp*a

            s1=max(a-1,0); y1=max(1-a,0)

            p1=scale*(dp+rp*s1-rp*y1-pc*y1)

            oc[i]=max(p0,pa,p1)

        den=oc.sum()



        sc=np.zeros(no); yc=np.zeros(no)

        for i in range(no):

            pc=pr*dh[i]

            sc[i]=(-W1*scale*rh[i]/den)+(W2/no)

            # SPO+: PC에 I[DA>RT] 조건

            indicator = 1.0 if dh[i] > rh[i] else 0.0

            yc[i]=(W1*scale*rh[i]/den)+(W1*scale*pc*indicator/den)+(W2/no)



        bl=[i for i in range(no) if sc[i]+yc[i]<0]

        ba=np.array(bl,dtype=int); nb=len(ba)

        bs=0;xs=nfa;yps=nfa+no;yms=nfa+2*no;zs=nfa+3*no

        nv=nfa+3*no+nb



        obj=np.zeros(nv)

        for i in range(no):

            obj[xs+i]=-W1*scale*dh[i]/den

            obj[yps+i]=sc[i]; obj[yms+i]=yc[i]



        Xs=sparse.csr_matrix(ar_des); In=sparse.eye(no,format="csr")

        ea=sparse.lil_matrix((no,nv)); ea[:,bs:bs+nfa]=-Xs; ea[:,xs:xs+no]=In

        eb=sparse.lil_matrix((no,nv)); eb[:,xs:xs+no]=In

        eb[:,yps:yps+no]=In; eb[:,yms:yms+no]=-In

        rhs_ar=np.concatenate([np.zeros(no),yh])

        con=[LinearConstraint(sparse.vstack([ea,eb],format="csr"), rhs_ar, rhs_ar)]

        if nb>0:

            cm=sparse.lil_matrix((2*nb,nv))

            for k in range(nb):

                rr=ba[k]; cm[k,yps+rr]=1;cm[k,zs+k]=1

                cm[nb+k,yms+rr]=1;cm[nb+k,zs+k]=-1

            con.append(LinearConstraint(cm.tocsr(),np.full(2*nb,-np.inf),

                  np.concatenate([np.ones(nb),np.zeros(nb)])))

        lb=np.concatenate([np.full(nfa,-np.inf),np.zeros(3*no+nb)])

        ub=np.concatenate([np.full(nfa,np.inf),np.ones(3*no+nb)])

        ig=np.zeros(nv,dtype=int); ig[zs:zs+nb]=1

        r=milp(c=obj,integrality=ig,bounds=Bounds(lb,ub),

               constraints=con,options={"mip_rel_gap":1e-9})

        if not r.success: raise RuntimeError(f"AR MILP h={hour}: {r.message}")

        co[hour]=r.x[bs:bs+nfa]



    fc=np.zeros((nds_t,12)); pv=ts[-1]

    for d in range(nds_t):

        fv=np.concatenate([[1.0],pv[::-1]])

        for h in range(12): fc[d,h]=np.clip(co[h]@fv,0,1)

        pv=tss[d]

    return fc.flatten()





# =====================================================================

# 7. MLR SPO+ MILP

# =====================================================================

def solve_mlr_spo(pr, W1, W2):

    no=ntr

    oc=np.zeros(no)

    for i in range(no):

        a=m_ts[i];dp=m_da[i];rp=m_rt[i];pc=pr*dp

        p0=scale*rp*a;pa=scale*dp*a

        s1=max(a-1,0);y1=max(1-a,0)

        p1=scale*(dp+rp*s1-rp*y1-pc*y1)

        oc[i]=max(p0,pa,p1)

    den=oc.sum()



    sc=np.zeros(no); yc=np.zeros(no)

    for i in range(no):

        pc=pr*m_da[i]

        sc[i]=(-W1*scale*m_rt[i]/den)+(W2/no)

        indicator = 1.0 if m_da[i] > m_rt[i] else 0.0

        yc[i]=(W1*scale*m_rt[i]/den)+(W1*scale*pc*indicator/den)+(W2/no)



    bl=[i for i in range(no) if sc[i]+yc[i]<0]

    ba=np.array(bl,dtype=int); nb=len(ba)

    bs=0;xs=nfm;yps=nfm+no;yms=nfm+2*no;zs=nfm+3*no

    nv=nfm+3*no+nb



    obj=np.zeros(nv)

    for i in range(no):

        obj[xs+i]=-W1*scale*m_da[i]/den

        obj[yps+i]=sc[i];obj[yms+i]=yc[i]



    Xs=sparse.csr_matrix(m_xtr); In=sparse.eye(no,format="csr")

    ea=sparse.lil_matrix((no,nv)); ea[:,bs:bs+nfm]=-Xs; ea[:,xs:xs+no]=In

    eb=sparse.lil_matrix((no,nv)); eb[:,xs:xs+no]=In

    eb[:,yps:yps+no]=In; eb[:,yms:yms+no]=-In

    rhs_mlr=np.concatenate([np.zeros(no),m_ts])

    con=[LinearConstraint(sparse.vstack([ea,eb],format="csr"), rhs_mlr, rhs_mlr)]

    if nb>0:

        cm=sparse.lil_matrix((2*nb,nv))

        for k in range(nb):

            rr=ba[k];cm[k,yps+rr]=1;cm[k,zs+k]=1

            cm[nb+k,yms+rr]=1;cm[nb+k,zs+k]=-1

        con.append(LinearConstraint(cm.tocsr(),np.full(2*nb,-np.inf),

              np.concatenate([np.ones(nb),np.zeros(nb)])))

    lb=np.concatenate([np.full(nfm,-np.inf),np.zeros(3*no+nb)])

    ub=np.concatenate([np.full(nfm,np.inf),np.ones(3*no+nb)])

    ig=np.zeros(nv,dtype=int); ig[zs:zs+nb]=1

    r=milp(c=obj,integrality=ig,bounds=Bounds(lb,ub),

           constraints=con,options={"mip_rel_gap":1e-9})

    if not r.success: raise RuntimeError(f"MLR MILP: {r.message}")

    return np.clip(m_xte@r.x[bs:bs+nfm],0,1)





# =====================================================================

# 8. 캐시

# =====================================================================

ca={}; cm={}

def get_ar(pr,w1,w2):

    k=(pr,w1,w2)

    if k not in ca:

        p=solve_ar_spo(pr,w1,w2)

        ca[k]=(nrmse_f(p,ar_act),comp_gap(pr,p,ar_act,ar_da,ar_rt))

    return ca[k]

def get_mlr(pr,w1,w2):

    k=(pr,w1,w2)

    if k not in cm:

        p=solve_mlr_spo(pr,w1,w2)

        cm[k]=(nrmse_f(p,m_act),comp_gap(pr,p,m_act,m_da_t,m_rt_t))

    return cm[k]





# =====================================================================

# 9. 그리드 스윕

# =====================================================================

print("\n"+"="*70)

print("  3. 그리드 스윕 (50조합, AR+MLR SPO+)")

print("="*70)



grid=[]

for pr in GRID_RATES:

    print(f"\n  ── rate={pr} ──")

    for W1,W2 in GRID_W_RATIOS:

        an,ag=get_ar(pr,W1,W2); mn,mg=get_mlr(pr,W1,W2)

        grid.append((pr,W1,W2,an,ag,mn,mg))

        print(f"  {W1}/{W2}: AR(nRMSE={an:.2f}%,Gap={ag:.2f}%) "

              f"MLR(nRMSE={mn:.2f}%,Gap={mg:.2f}%)")



gc=os.path.join(OUT_DIR,"grid_spo_plus_ar_mlr_z01.csv")

with open(gc,"w",newline="") as f:

    w=csv.writer(f); w.writerow(["rate","W1","W2","AR_nRMSE","AR_Gap","MLR_nRMSE","MLR_Gap"])

    for r in grid: w.writerow([r[0],r[1],r[2],round(r[3],2),round(r[4],2),round(r[5],2),round(r[6],2)])

print(f"\n  saved: {gc}")





# =====================================================================

# 10. KPI (W1=1, W2=20, rate=50%)

# =====================================================================

print("\n"+"="*70)

print(f"  4. KPI — W1={KPI_W1}, W2={KPI_W2}, rate={KPI_RATE}")

print("="*70)



akn,akg=get_ar(KPI_RATE,KPI_W1,KPI_W2)

mkn,mkg=get_mlr(KPI_RATE,KPI_W1,KPI_W2)

abg=comp_gap(KPI_RATE,ar_pred,ar_act,ar_da,ar_rt)

mbg=comp_gap(KPI_RATE,mlr_pred,m_act,m_da_t,m_rt_t)



print(f"""

  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐

  │     모델          │  nRMSE (%)   │  Gap (%)     │  nRMSE (%)   │  Gap (%)     │

  │                  │  (SPO+)      │  (SPO+)      │  (baseline)  │  (baseline)  │

  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤

  │  AR  (z01/블럭21) │ {akn:>10.2f}   │ {akg:>10.2f}   │ {ar_nr:>10.2f}   │ {abg:>10.2f}   │

  │  MLR (z01/블럭21) │ {mkn:>10.2f}   │ {mkg:>10.2f}   │ {mlr_nr:>10.2f}   │ {mbg:>10.2f}   │

  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘



  ※ 4항식 결과 비교:

      AR  KPI: nRMSE=48.93%, Gap=16.23%

      MLR KPI: nRMSE=43.03%, Gap=14.78%

""")





# =====================================================================

# 11. Fig3/6 (W1/W2 10개, rate=0.5) + Fig5/8 (rate sweep, W1=W2=1)

# =====================================================================

FR=KPI_RATE

print("\n"+"="*70)

print(f"  5. Fig.3/6 (rate={FR}) + Fig.5/8 (W1=W2=1)")

print("="*70)



f3n=[];f3g=[];f6n=[];f6g=[]

for W1,W2 in FIG_W_RATIOS:

    an,ag=get_ar(FR,W1,W2); mn,mg=get_mlr(FR,W1,W2)

    f3n.append(an);f3g.append(ag);f6n.append(mn);f6g.append(mg)

    print(f"  {W1}/{W2}: AR({an:.2f}/{ag:.2f}) MLR({mn:.2f}/{mg:.2f})")



f5bn=[];f5bg=[];f5pn=[];f5pg=[]

f8bn=[];f8bg=[];f8pn=[];f8pg=[]

for rate in FIG_RATES:

    ab=comp_gap(rate,ar_pred,ar_act,ar_da,ar_rt)

    mb=comp_gap(rate,mlr_pred,m_act,m_da_t,m_rt_t)

    f5bn.append(ar_nr);f5bg.append(ab);f8bn.append(mlr_nr);f8bg.append(mb)

    an,ag=get_ar(rate,1,1); mn,mg=get_mlr(rate,1,1)

    f5pn.append(an);f5pg.append(ag);f8pn.append(mn);f8pg.append(mg)

    print(f"  rate={rate:.1f}: AR(b={ab:.2f},p={an:.2f}/{ag:.2f}) "

          f"MLR(b={mb:.2f},p={mn:.2f}/{mg:.2f})")



def sv(p,h,r):

    with open(p,"w",newline="") as f:

        csv.writer(f).writerow(h)

        for row in r: csv.writer(f).writerow(row)

    print(f"  saved: {p}")



sv(os.path.join(OUT_DIR,"fig3_spo_z01.csv"),["Label","nRMSE","Gap"],

   [[FIG_LABELS[i+1],round(f3n[i],2),round(f3g[i],2)] for i in range(10)])

sv(os.path.join(OUT_DIR,"fig5_spo_z01.csv"),["Rate","b_nRMSE","b_Gap","p_nRMSE","p_Gap"],

   [[FIG_RATES[i],round(f5bn[i],2),round(f5bg[i],2),round(f5pn[i],2),round(f5pg[i],2)] for i in range(16)])

sv(os.path.join(OUT_DIR,"fig6_spo_z01.csv"),["Label","nRMSE","Gap"],

   [[FIG_LABELS[i+1],round(f6n[i],2),round(f6g[i],2)] for i in range(10)])

sv(os.path.join(OUT_DIR,"fig8_spo_z01.csv"),["Rate","b_nRMSE","b_Gap","p_nRMSE","p_Gap"],

   [[FIG_RATES[i],round(f8bn[i],2),round(f8bg[i],2),round(f8pn[i],2),round(f8pg[i],2)] for i in range(16)])





# =====================================================================

# 12-15. Fig.3/5/6/8 plotting

# =====================================================================

x11=list(range(11)); x16=list(range(16))

lbl=[f"{int(r*100)}%" for r in FIG_RATES]

hl=FIG_RATES.index(KPI_RATE)



def plot_dual(fig_id, model, bn, bg, pn, pg, cbase, cprop, title):

    fig,al=plt.subplots(figsize=(10,6))

    fig.suptitle(title,fontsize=13)

    ax=al.twinx()

    l1=al.axhline(bn,color=cbase,lw=2,alpha=0.5,label=f"{model} baseline nRMSE")

    l2=al.plot(x11[1:],pn,marker="o",color=cprop,lw=2,label="SPO+ nRMSE")

    l3=ax.axhline(bg,color=cbase,lw=2,alpha=0.5,ls="--",label=f"{model} baseline Gap")

    l4=ax.plot(x11[1:],pg,marker="s",color=cprop,lw=2,ls="--",label="SPO+ Gap")

    al.set_xticks(x11);al.set_xticklabels(FIG_LABELS)

    al.set_xlabel("W1/W2");al.set_ylabel("nRMSE (%)",color=cbase)

    ax.set_ylabel("Optimality Gap (%)",color=cprop)

    al.tick_params(axis="y",labelcolor=cbase);ax.tick_params(axis="y",labelcolor=cprop)

    al.grid(True,alpha=0.3,color=C_GRID)

    al.legend([l1]+l2+[l3,l4],[f"{model} baseline nRMSE","SPO+ nRMSE",

              f"{model} baseline Gap","SPO+ Gap"],loc="upper left",fontsize=9)

    fig.tight_layout()

    p=os.path.join(RESULTS_DIR,f"{fig_id}_spo_z01.png")

    fig.savefig(p,dpi=150);plt.close(fig)

    print(f"  saved: {p}"); return p



def plot_split(fig_id, model, btn, btg, ptn, ptg, cbase, cprop, title):

    fig,(an,ag)=plt.subplots(1,2,figsize=(13,5))

    fig.suptitle(title,fontsize=13)

    an.plot(x16,btn,marker="o",color=cbase,lw=2,label=f"{model} baseline nRMSE")

    an.plot(x16,ptn,marker="s",color=cprop,lw=2,label="SPO+ nRMSE")

    an.set_xticks(x16);an.set_xticklabels(lbl,rotation=45)

    an.set_xlabel("벌금비용률");an.set_ylabel("nRMSE (%)")

    an.set_title("nRMSE");an.grid(True,alpha=0.3,color=C_GRID);an.legend(fontsize=9)

    ag.plot(x16,btg,marker="o",color=cbase,lw=2,label=f"{model} baseline Gap")

    ag.plot(x16,ptg,marker="s",color=cprop,lw=2,label="SPO+ Gap")

    ag.axvline(hl,color=C_GRID,ls=":",lw=1.5,label=f"KPI rate={int(KPI_RATE*100)}%")

    ag.set_xticks(x16);ag.set_xticklabels(lbl,rotation=45)

    ag.set_xlabel("벌금비용률");ag.set_ylabel("Optimality Gap (%)")

    ag.set_title("Optimality Gap");ag.grid(True,alpha=0.3,color=C_GRID);ag.legend(fontsize=9)

    fig.tight_layout()

    p=os.path.join(RESULTS_DIR,f"{fig_id}_spo_z01.png")

    fig.savefig(p,dpi=150);plt.close(fig)

    print(f"  saved: {p}"); return p



print("\n"+"="*70)

print("  6. Plotting")

print("="*70)



p3=plot_dual("fig3","AR",ar_nr,abg,f3n,f3g,C_AR_BASE,C_AR_PROP,

             f"Fig.3 — SPO+, AR, rate={FR}")

p5=plot_split("fig5","AR",f5bn,f5bg,f5pn,f5pg,C_AR_BASE,C_AR_PROP,

              "Fig.5 — SPO+, AR, W1=W2=1, rate 스윕")

p6=plot_dual("fig6","MLR",mlr_nr,mbg,f6n,f6g,C_MLR_BASE,C_MLR_PROP,

             f"Fig.6 — SPO+, MLR, rate={FR}")

p8=plot_split("fig8","MLR",f8bn,f8bg,f8pn,f8pg,C_MLR_BASE,C_MLR_PROP,

              "Fig.8 — SPO+, MLR, W1=W2=1, rate 스윕")



print("\n"+"="*70)

print("  완료 — SPO+ AR+MLR (z01/블럭21)")

print("="*70)

print(f"  Fig.3: {p3}\n  Fig.5: {p5}\n  Fig.6: {p6}\n  Fig.8: {p8}\n  Grid: {gc}")