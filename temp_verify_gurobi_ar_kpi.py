# -*- coding: utf-8 -*-
# temp_verify_gurobi_ar_kpi.py
#
# 4개 Gurobi 변환본의 AR baseline + KPI 지점(1개)을, 같은 원본 SciPy 파일의
# AR 관련 코드만 exec로 잘라서 돌린 결과와 직접 비교한다. MLR은 건드리지
# 않는다(자유 라이선스 크기 제한으로 별도 확인 필요).
import io
import time
import numpy as np

def load_ar_section(path, cut_start_marker, cut_end_marker, func_end_marker):
    """파일을 읽어서 [처음:MLR baseline 시작) + [AR 함수 시작:AR 함수 끝] 만 이어붙여 반환."""
    lines = io.open(path, encoding="utf-8").read().split("\n")
    idx_cut_start = next(i for i, l in enumerate(lines) if cut_start_marker in l)
    idx_cut_end = next(i for i, l in enumerate(lines) if cut_end_marker in l)
    idx_func_end = next(i for i, l in enumerate(lines) if func_end_marker in l)
    part1 = lines[:idx_cut_start]
    part2 = lines[idx_cut_end:idx_func_end + 1]
    return "\n".join(part1 + part2)


def run_case(name, scipy_path, gurobi_path, cut_start, cut_end, func_end,
             solve_fn_name, rate, w1, w2, gap_fn_name="compute_gap_4term",
             gap_extra_args=None):
    print("=" * 78)
    print(name)
    print("=" * 78)

    import os
    src_scipy = load_ar_section(scipy_path, cut_start, cut_end, func_end)
    ns_scipy = {"__file__": os.path.abspath(scipy_path), "__name__": "temp_verify_scipy"}
    t0 = time.time()
    exec(compile(src_scipy, scipy_path, "exec"), ns_scipy)
    t_scipy = time.time() - t0

    src_gurobi = load_ar_section(gurobi_path, cut_start, cut_end, func_end)
    ns_gurobi = {"__file__": os.path.abspath(gurobi_path), "__name__": "temp_verify_gurobi"}
    t0 = time.time()
    exec(compile(src_gurobi, gurobi_path, "exec"), ns_gurobi)
    t_gurobi = time.time() - t0

    base_n_s = ns_scipy["ar_nrmse"]
    base_n_g = ns_gurobi["ar_nrmse"]
    print(f"  baseline AR nRMSE:  scipy={base_n_s:.6f}%   gurobi={base_n_g:.6f}%   diff={abs(base_n_s-base_n_g):.2e}")

    pred_s = ns_scipy[solve_fn_name](rate, w1, w2)
    pred_g = ns_gurobi[solve_fn_name](rate, w1, w2)
    if isinstance(pred_s, tuple):  # 방법4/5의 solve_ar()는 (nrmse, gap) 튜플을 바로 반환
        n_s, g_s = pred_s
        n_g, g_g = pred_g
        print(f"  KPI(rate={rate},W1={w1},W2={w2}) nRMSE:  scipy={n_s:.6f}%  gurobi={n_g:.6f}%  diff={abs(n_s-n_g):.2e}")
        print(f"  KPI(rate={rate},W1={w1},W2={w2}) Gap  :  scipy={g_s:.6f}%  gurobi={g_g:.6f}%  diff={abs(g_s-g_g):.2e}")
        max_pred_diff = None
    else:  # 방법6/8의 solve_ar_proposed()는 예측치 배열을 반환 -> nrmse/gap은 직접 계산
        max_pred_diff = float(np.max(np.abs(pred_s - pred_g)))
        nrmse_fn_s = ns_scipy.get("nrmse", ns_scipy.get("calc_nrmse"))
        nrmse_fn_g = ns_gurobi.get("nrmse", ns_gurobi.get("calc_nrmse"))
        actual = ns_scipy["actual_flat_all"]
        da = ns_scipy["da_flat_all"]; rt = ns_scipy["rt_flat_all"]
        n_s = nrmse_fn_s(pred_s, actual); n_g = nrmse_fn_g(pred_g, actual)
        gap_fn_s = ns_scipy[gap_fn_name]; gap_fn_g = ns_gurobi[gap_fn_name]
        g_s = gap_fn_s(pred_s, rate, actual, da, rt)
        g_g = gap_fn_g(pred_g, rate, actual, da, rt)
        print(f"  예측값 최대절대차(pred_s vs pred_g): {max_pred_diff:.3e}")
        print(f"  KPI(rate={rate},W1={w1},W2={w2}) nRMSE:  scipy={n_s:.6f}%  gurobi={n_g:.6f}%  diff={abs(n_s-n_g):.2e}")
        print(f"  KPI(rate={rate},W1={w1},W2={w2}) Gap  :  scipy={g_s:.6f}%  gurobi={g_g:.6f}%  diff={abs(g_s-g_g):.2e}")
    print(f"  실행시간: scipy={t_scipy:.1f}s  gurobi={t_gurobi:.1f}s")
    print()


# ── 1) 방법8 (방법2_4term_ar_mlr_z03_block18) ──
run_case(
    "1) 방법8_w2x4_ar_mlr_z03_block18",
    "방법2_4term_ar_mlr_z03_block18.py",
    "방법8_w2x4_ar_mlr_z03_block18_gurobi.py",
    cut_start="# 5. MLR baseline", cut_end="# 6. 제안모형 MILP — AR",
    func_end="return fc.flatten()",
    solve_fn_name="solve_ar_proposed", rate=0.5, w1=1, w2=20,
)

# ── 2) 방법6 ──
run_case(
    "2) 방법6_pc_conditional_ar_mlr",
    "방법6_pc_conditional_ar_mlr.py",
    "방법6_pc_conditional_ar_mlr_gurobi.py",
    cut_start="# 5. MLR baseline", cut_end="# 6. 제안모형 MILP — AR",
    func_end="return fc.flatten()",
    solve_fn_name="solve_ar_proposed", rate=0.5, w1=1, w2=20,
)

# ── 3) 방법4 ──
run_case(
    "3) 방법4_pc_max_rt_dt_AR_MLR",
    "방법4_pc_max_rt_dt_AR_MLR.py",
    "방법4_pc_max_rt_dt_AR_MLR_gurobi.py",
    cut_start="# 5. MLR baseline", cut_end="# 6. 제안모형 MILP — AR",
    func_end="_cache[k] = val; return val",
    solve_fn_name="solve_ar", rate=0.5, w1=1, w2=20,
)

# ── 4) 방법5 ──
run_case(
    "4) 방법5_pc_da_plus_rt_AR_MLR",
    "방법5_pc_da_plus_rt_AR_MLR.py",
    "방법5_pc_da_plus_rt_AR_MLR_gurobi.py",
    cut_start="# 5. MLR baseline", cut_end="# 6. 제안모형 MILP — AR",
    func_end="_cache[k] = val; return val",
    solve_fn_name="solve_ar", rate=0.5, w1=1, w2=20,
)

print("완료")
