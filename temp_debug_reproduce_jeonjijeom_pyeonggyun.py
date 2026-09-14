# -*- coding: utf-8 -*-
# 5.9.3 순위표의 "전 지점 평균" 참고열이 어떤 계산식으로 나온 건지 역추적.
# 방법1(목표 153.8 = Fig3 39.5+Fig5 52.2+Fig6 34.5+Fig8 27.6)과
# 방법6(목표 21.0 = Fig3 5.1+Fig5 9.3+Fig6 3.0+Fig8 3.5)에
# 여러 방식(10점 전체평균/11점/5점 GRID세트)을 적용해 기존 숫자를 재현하는지 확인.
import pandas as pd
import numpy as np

pd.set_option('display.width', 200)

PAPER_PROP_NRMSE = np.array([34.89,35.14,36.28,41.09,44.95,46.11,48.27,49.21,49.61,50.07])
PAPER_PROP_GAP   = np.array([13.91,13.42,12.71,11.88,11.44,11.38,11.38,11.36,11.36,11.36])
PAPER_PROP_MLR_NRMSE = np.array([21.92,21.84,21.75,21.66,22.01,23.32,27.75,30.62,35.76,37.67])
PAPER_PROP_MLR_GAP   = np.array([11.91,11.68,11.15,10.65,10.28,9.91,9.51,9.32,9.27,9.28])
FIG5_PAPER_PROP_NRMSE = np.array([46,34,35,36,40,44.45,50,55,61,64,67])
FIG5_PAPER_PROP_GAP   = np.array([8,10,11,11,11,11.44,11,11.5,11.5,11.5,11.5])
FIG8_PAPER_PROP_NRMSE = np.array([23.92,22.56,21.76,21.41,21.67,22.01,22.45,23.14,23.76,24.98,26.10])
FIG8_PAPER_PROP_GAP   = np.array([7.59,8.53,9.17,9.56,9.94,10.28,10.52,10.68,10.85,10.96,10.93])

GRID5 = [0,1,2,4,9]  # 1/20,1/10,1/5,1/1,1/0 (인덱스, FIG_W_RATIOS 10점 중)


def dev(n, p, idx=None):
    n = np.asarray(n); p = np.asarray(p)
    if idx is not None:
        n = n[idx]; p = p[idx]
    return np.abs(n - p) + np.abs(p - p) * 0 + np.abs(n - p) * 0 + np.abs(n - p)  # just |n-p|


def avg_dev(n, p, idx=None):
    n = np.asarray(n); p = np.asarray(p)
    if idx is not None:
        n = n[idx]; p = p[idx]
    return np.abs(n - p)


def method_avg(n_col, g_col, p_n, p_g, idx=None):
    dn = avg_dev(n_col, p_n, idx)
    dg = avg_dev(g_col, p_g, idx)
    return (dn + dg).mean()


print("=" * 70)
print("방법1 재현 시도 (목표: Fig3 39.5+Fig5 52.2+Fig6 34.5+Fig8 27.6 = 153.8)")
print("=" * 70)
f3 = pd.read_csv("results/fig3_paper_AR.csv")
f6 = pd.read_csv("results/fig6_paper_MLR.csv")
f5 = pd.read_csv("results/fig5_paper_AR.csv")
f8 = pd.read_csv("results/fig8_paper_MLR.csv")

n3 = f3["Code_nRMSE"].values[:10]; g3 = f3["Code_Gap"].values[:10]  # 1/20..1/0 (10개, 0/1 제외)
n6 = f6["Code_nRMSE"].values[:10]; g6 = f6["Code_Gap"].values[:10]
n5 = f5["Code_nRMSE"].values[:11]; g5 = f5["Code_Gap"].values[:11]  # rate 0~100% (11개, 1.1~1.5 제외)
n8 = f8["Code_nRMSE"].values[:11]; g8 = f8["Code_Gap"].values[:11]

for label, idx3, idx5 in [("10점 전체", None, None), ("5점 GRID(1/20,10,5,1,0)", GRID5, None)]:
    a3 = method_avg(n3, g3, PAPER_PROP_NRMSE, PAPER_PROP_GAP, idx3)
    a6 = method_avg(n6, g6, PAPER_PROP_MLR_NRMSE, PAPER_PROP_MLR_GAP, idx3)
    a5 = method_avg(n5, g5, FIG5_PAPER_PROP_NRMSE, FIG5_PAPER_PROP_GAP, idx5)
    a8 = method_avg(n8, g8, FIG8_PAPER_PROP_NRMSE, FIG8_PAPER_PROP_GAP, idx5)
    print(f"[{label}] Fig3={a3:.1f} Fig5={a5:.1f} Fig6={a6:.1f} Fig8={a8:.1f}  합계={a3+a5+a6+a8:.1f}")
print("목표: Fig3=39.5 Fig5=52.2 Fig6=34.5 Fig8=27.6 합계=153.8")

print()
print("=" * 70)
print("방법6 재현 시도 (목표: Fig3 5.1+Fig5 9.3+Fig6 3.0+Fig8 3.5 = 21.0)")
print("=" * 70)
f3b = pd.read_csv("results/simulation_output/fig3_method6_AR.csv")
f6b = pd.read_csv("results/simulation_output/fig6_method6_MLR.csv")
f5b = pd.read_csv("results/simulation_output/fig5_method6_AR.csv")
f8b = pd.read_csv("results/simulation_output/fig8_method6_MLR.csv")

n3b = f3b["nRMSE"].values[1:]; g3b = f3b["Gap"].values[1:]  # skip AR/MLR baseline row -> 10 pts
n6b = f6b["nRMSE"].values[1:]; g6b = f6b["Gap"].values[1:]
n5b = f5b["AR_prop_nRMSE"].values; g5b = f5b["AR_prop_Gap"].values  # 11 pts (rate 0~100%)
n8b = f8b["MLR_prop_nRMSE"].values; g8b = f8b["MLR_prop_Gap"].values

for label, idx3, idx5 in [("10점 전체", None, None), ("5점 GRID(1/20,10,5,1,0)", GRID5, None)]:
    a3 = method_avg(n3b, g3b, PAPER_PROP_NRMSE, PAPER_PROP_GAP, idx3)
    a6 = method_avg(n6b, g6b, PAPER_PROP_MLR_NRMSE, PAPER_PROP_MLR_GAP, idx3)
    a5 = method_avg(n5b, g5b, FIG5_PAPER_PROP_NRMSE, FIG5_PAPER_PROP_GAP, idx5)
    a8 = method_avg(n8b, g8b, FIG8_PAPER_PROP_NRMSE, FIG8_PAPER_PROP_GAP, idx5)
    print(f"[{label}] Fig3={a3:.1f} Fig5={a5:.1f} Fig6={a6:.1f} Fig8={a8:.1f}  합계={a3+a5+a6+a8:.1f}")
print("목표: Fig3=5.1 Fig5=9.3 Fig6=3.0 Fig8=3.5 합계=21.0")
