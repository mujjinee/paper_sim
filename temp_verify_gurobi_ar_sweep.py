# -*- coding: utf-8 -*-
# temp_verify_gurobi_ar_sweep.py
# AR 전체 스윕(Fig.3류 W1/W2 10점 @rate=0.5, Fig.5류 rate 11점 @W1=W2=1) 비교.
import io, os, time
import numpy as np


def load_ar_section(path, cut_start_marker, cut_end_marker, func_end_marker):
    lines = io.open(path, encoding="utf-8").read().split("\n")
    idx_cut_start = next(i for i, l in enumerate(lines) if cut_start_marker in l)
    idx_cut_end = next(i for i, l in enumerate(lines) if cut_end_marker in l)
    idx_func_end = next(i for i, l in enumerate(lines) if func_end_marker in l)
    part1 = lines[:idx_cut_start]
    part2 = lines[idx_cut_end:idx_func_end + 1]
    return "\n".join(part1 + part2)


def exec_ns(path, cut_start, cut_end, func_end):
    src = load_ar_section(path, cut_start, cut_end, func_end)
    ns = {"__file__": os.path.abspath(path), "__name__": "temp_verify"}
    exec(compile(src, path, "exec"), ns)
    return ns


def compare(name, scipy_path, gurobi_path, cut_start, cut_end, func_end,
            solve_fn_name, w_ratios, rates, kind):
    print("=" * 78); print(name); print("=" * 78)
    t0 = time.time()
    ns_s = exec_ns(scipy_path, cut_start, cut_end, func_end)
    ns_g = exec_ns(gurobi_path, cut_start, cut_end, func_end)
    print(f"  (로딩+baseline 소요: {time.time()-t0:.1f}s)")

    if kind == "tuple":  # 방법4/5 solve_ar(pc_rate,W1,W2) -> (nrmse,gap)
        solve_s = ns_s[solve_fn_name]; solve_g = ns_g[solve_fn_name]
        n_diffs = []; g_diffs = []
        for w1, w2 in w_ratios:
            ns_val, gs_val = solve_s(0.5, w1, w2)
            ng_val, gg_val = solve_g(0.5, w1, w2)
            n_diffs.append(abs(ns_val - ng_val)); g_diffs.append(abs(gs_val - gg_val))
        print(f"  Fig3류(W1/W2 {len(w_ratios)}점,rate=0.5): max|dNRMSE|={max(n_diffs):.2e}  max|dGap|={max(g_diffs):.2e}")
        n_diffs2 = []; g_diffs2 = []
        for r in rates:
            ns_val, gs_val = solve_s(r, 1, 1)
            ng_val, gg_val = solve_g(r, 1, 1)
            n_diffs2.append(abs(ns_val - ng_val)); g_diffs2.append(abs(gs_val - gg_val))
        print(f"  Fig5류(rate {len(rates)}점,W1=W2=1):    max|dNRMSE|={max(n_diffs2):.2e}  max|dGap|={max(g_diffs2):.2e}")
    else:  # 방법6/8 solve_ar_proposed(rate,W1,W2) -> pred array
        solve_s = ns_s[solve_fn_name]; solve_g = ns_g[solve_fn_name]
        nrmse_s = ns_s.get("nrmse", ns_s.get("calc_nrmse")); nrmse_g = ns_g.get("nrmse", ns_g.get("calc_nrmse"))
        gap_s = ns_s["compute_gap_4term"]; gap_g = ns_g["compute_gap_4term"]
        actual = ns_s["actual_flat_all"]; da = ns_s["da_flat_all"]; rt = ns_s["rt_flat_all"]
        n_diffs = []; g_diffs = []; pred_diffs = []
        for w1, w2 in w_ratios:
            p_s = solve_s(0.5, w1, w2); p_g = solve_g(0.5, w1, w2)
            pred_diffs.append(np.max(np.abs(p_s - p_g)))
            n_diffs.append(abs(nrmse_s(p_s, actual) - nrmse_g(p_g, actual)))
            g_diffs.append(abs(gap_s(p_s, 0.5, actual, da, rt) - gap_g(p_g, 0.5, actual, da, rt)))
        print(f"  Fig3류(W1/W2 {len(w_ratios)}점,rate=0.5): max|dpred|={max(pred_diffs):.2e}  max|dNRMSE|={max(n_diffs):.2e}  max|dGap|={max(g_diffs):.2e}")
        n_diffs2 = []; g_diffs2 = []; pred_diffs2 = []
        for r in rates:
            p_s = solve_s(r, 1, 1); p_g = solve_g(r, 1, 1)
            pred_diffs2.append(np.max(np.abs(p_s - p_g)))
            n_diffs2.append(abs(nrmse_s(p_s, actual) - nrmse_g(p_g, actual)))
            g_diffs2.append(abs(gap_s(p_s, r, actual, da, rt) - gap_g(p_g, r, actual, da, rt)))
        print(f"  Fig5류(rate {len(rates)}점,W1=W2=1):    max|dpred|={max(pred_diffs2):.2e}  max|dNRMSE|={max(n_diffs2):.2e}  max|dGap|={max(g_diffs2):.2e}")
    print(f"  (총 소요: {time.time()-t0:.1f}s)\n")


FIG_W_RATIOS = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
FIG_RATES = [round(0.1 * i, 1) for i in range(11)]
W_RATIOS_45 = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]
RATES_45 = [round(0.1 * i, 1) for i in range(11)]

compare("1) 방법8_w2x4_ar_mlr_z03_block18 — 전체 AR 스윕",
        "방법2_4term_ar_mlr_z03_block18.py", "방법8_w2x4_ar_mlr_z03_block18_gurobi.py",
        "# 5. MLR baseline", "# 6. 제안모형 MILP — AR", "return fc.flatten()",
        "solve_ar_proposed", FIG_W_RATIOS, FIG_RATES, kind="pred")

compare("2) 방법6_pc_conditional_ar_mlr — 전체 AR 스윕",
        "방법6_pc_conditional_ar_mlr.py", "방법6_pc_conditional_ar_mlr_gurobi.py",
        "# 5. MLR baseline", "# 6. 제안모형 MILP — AR", "return fc.flatten()",
        "solve_ar_proposed", FIG_W_RATIOS, FIG_RATES, kind="pred")

compare("3) 방법4_pc_max_rt_dt_AR_MLR — 전체 AR 스윕",
        "방법4_pc_max_rt_dt_AR_MLR.py", "방법4_pc_max_rt_dt_AR_MLR_gurobi.py",
        "# 5. MLR baseline", "# 6. 제안모형 MILP — AR", "_cache[k] = val; return val",
        "solve_ar", W_RATIOS_45, RATES_45, kind="tuple")

compare("4) 방법5_pc_da_plus_rt_AR_MLR — 전체 AR 스윕",
        "방법5_pc_da_plus_rt_AR_MLR.py", "방법5_pc_da_plus_rt_AR_MLR_gurobi.py",
        "# 5. MLR baseline", "# 6. 제안모형 MILP — AR", "_cache[k] = val; return val",
        "solve_ar", W_RATIOS_45, RATES_45, kind="tuple")

print("완료")
