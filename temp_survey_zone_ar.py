# -*- coding: utf-8 -*-
# temp_survey_zone_ar.py
#
# GEFCom2014 zone1/2/3 각각에 대해 90일 학습 + 30일 테스트로 구성된
# 120일 롤링 블록을 30일 간격으로 전수 생성하고, 각 블록마다 AR baseline
# (시간대별 12개, bounded LAD 회귀)의 nRMSE를 계산한다.
#
# 그 다음 각 블록의 "호주 계절"(테스트 구간 다수 월 기준 — 여름 12~2월,
# 가을 3~5월, 겨울 6~8월, 봄 9~11월)을 매겨서, 계절별 nRMSE 평균/최소/최대/
# 표준편차를 집계한다 — 보고서 4.1.1절 "계절별 요약 통계" 표를 재현하기 위함.
#
# AR baseline 방법론(설계행렬·bounded LAD·rolling 예측)은 기존 통합 스크립트
# (예: 방법2_4term_ar_mlr_z03_block18.py)의 "AR baseline" 절과 동일하게 맞췄다.

import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21
HOURS_PER_DAY = LOCAL_HOUR_END - LOCAL_HOUR_START  # 12

TRAIN_DAYS = 90
TEST_DAYS = 30
BLOCK_STEP_DAYS = 30  # 블록 시작을 30일씩 밀며 롤링

ZONES = {
    "Z1": "merged_for_simulation_z01.csv",
    "Z2": "merged_for_simulation_z02.csv",
    "Z3": "merged_for_simulation_z03.csv",
}

# 호주(남반구) 기준 계절 — 월 -> 계절
MONTH_TO_SEASON = {
    12: "여름", 1: "여름", 2: "여름",
    3: "가을", 4: "가을", 5: "가을",
    6: "겨울", 7: "겨울", 8: "겨울",
    9: "봄", 10: "봄", 11: "봄",
}


def nrmse(pred_flat, actual_flat):
    rmse = np.sqrt(np.mean((actual_flat - pred_flat) ** 2))
    denom = np.mean(actual_flat)
    return 100.0 * rmse / denom if denom > 1e-10 else np.nan


def fit_ar_baseline(train_solar, history_row):
    """시간대별(12개) bounded LAD 회귀. 설계행렬 = 절편 + 직전 하루 12시간(역순).
    train_solar: (n_train_days, 12), history_row: (12,) — train 첫날 바로 전날."""
    n_train_days = train_solar.shape[0]
    history_and_train = np.vstack([history_row.reshape(1, -1), train_solar])
    n_features_ar = HOURS_PER_DAY + 1  # 1(intercept) + 12(lag)

    ar_intercept = np.ones((n_train_days, 1))
    ar_lag = np.zeros((n_train_days, HOURS_PER_DAY))
    for d in range(n_train_days):
        ar_lag[d] = history_and_train[d][::-1]
    ar_design = np.hstack([ar_intercept, ar_lag])

    ar_coefficients = np.zeros((HOURS_PER_DAY, n_features_ar))
    X_sp = sparse.csr_matrix(ar_design)
    I_n = sparse.eye(n_train_days, format="csr")
    A = sparse.vstack([sparse.hstack([X_sp, -I_n]),
                        sparse.hstack([-X_sp, -I_n])], format="csr")
    c = np.concatenate([np.zeros(n_features_ar), np.ones(n_train_days) / n_train_days])
    vb = [(None, None)] * n_features_ar + [(0.0, None)] * n_train_days

    for h in range(HOURS_PER_DAY):
        y_h = train_solar[:, h]
        b = np.concatenate([y_h, -y_h])
        res = linprog(c, A_ub=A, b_ub=b, bounds=vb, method="highs")
        ar_coefficients[h] = res.x[:n_features_ar]
    return ar_coefficients


def predict_ar(ar_coefficients, test_solar, last_train_day):
    """rolling 1-step 예측: 전날 실제값으로 다음날을 예측 (0~1 clip)."""
    n_test_days = test_solar.shape[0]
    pred = np.zeros((n_test_days, HOURS_PER_DAY))
    prev_day = last_train_day
    for d in range(n_test_days):
        feat = np.concatenate([[1.0], prev_day[::-1]])
        for h in range(HOURS_PER_DAY):
            pred[d, h] = np.clip(np.dot(ar_coefficients[h], feat), 0.0, 1.0)
        prev_day = test_solar[d]
    return pred


def build_daylight_matrix(df, dates):
    """지정된 날짜 목록에 대해 (일수 x 12) 발전량 행렬을 만든다. 빠진 (날짜,시간)은 0으로 채움."""
    mat = np.zeros((len(dates), HOURS_PER_DAY))
    date_to_idx = {d: i for i, d in enumerate(dates)}
    sub = df[df["local_date"].isin(dates)]
    for _, row in sub.iterrows():
        d_idx = date_to_idx.get(row["local_date"])
        if d_idx is None:
            continue
        h_idx = int(row["local_hour"]) - LOCAL_HOUR_START
        if 0 <= h_idx < HOURS_PER_DAY:
            mat[d_idx, h_idx] = row["solar_power"]
    return mat


def survey_zone(zone_label, csv_path):
    print("=" * 70)
    print(f"  {zone_label}: {csv_path}")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    df["local_date"] = pd.to_datetime(df["local_date"])
    df = df[(df["local_hour"] >= LOCAL_HOUR_START) & (df["local_hour"] < LOCAL_HOUR_END)].copy()

    all_dates = sorted(df["local_date"].unique())
    first_date = all_dates[0]
    last_date = all_dates[-1]
    total_days = (last_date - first_date).days + 1

    rows = []
    block_no = 0
    # 블록 i: history(1일) + train(90일) + test(30일), 시작을 30일씩 밀며 롤링
    start_offset = 1  # history 하루 확보를 위해 첫날은 history로 쓰고 1일부터 시작
    while True:
        history_date = first_date + pd.Timedelta(days=start_offset - 1 + block_no * BLOCK_STEP_DAYS)
        train_start = history_date + pd.Timedelta(days=1)
        train_end = train_start + pd.Timedelta(days=TRAIN_DAYS - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=TEST_DAYS - 1)
        if test_end > last_date:
            break
        block_no += 1

        train_dates = pd.date_range(train_start, train_end, freq="D")
        test_dates = pd.date_range(test_start, test_end, freq="D")

        history_mat = build_daylight_matrix(df, [history_date])
        train_mat = build_daylight_matrix(df, list(train_dates))
        test_mat = build_daylight_matrix(df, list(test_dates))

        # 발전 시간대(합이 0보다 큰 시간)만 nRMSE 계산에 사용
        ar_coef = fit_ar_baseline(train_mat, history_mat[0])
        pred_mat = predict_ar(ar_coef, test_mat, train_mat[-1])

        daylight_mask = test_mat.sum(axis=0) > 0.01
        actual_flat = test_mat[:, daylight_mask].flatten()
        pred_flat = pred_mat[:, daylight_mask].flatten()
        block_nrmse = nrmse(pred_flat, actual_flat)

        # 테스트 구간에서 가장 많이 등장하는 월 -> 호주 계절
        month_counts = pd.Series(test_dates.month).value_counts()
        majority_month = int(month_counts.idxmax())
        season = MONTH_TO_SEASON[majority_month]

        rows.append({
            "zone": zone_label,
            "block": block_no,
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "season": season,
            "ar_nrmse": round(block_nrmse, 2),
        })
        print(f"  블록{block_no:>2} | 테스트 {test_start.date()}~{test_end.date()} "
              f"| {season} | AR nRMSE={block_nrmse:.2f}%")

    result = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f"season_block_survey_ar_{zone_label.lower()}_lad.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  -> 저장: {out_path} ({len(result)}개 블록)\n")
    return result


def main():
    all_results = []
    for zone_label, fname in ZONES.items():
        csv_path = os.path.join(BASE_DIR, fname)
        all_results.append(survey_zone(zone_label, csv_path))

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "season_block_survey_ar_all_lad.csv")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"통합 결과 저장: {combined_path} (총 {len(combined)}개 블록)")

    print("\n" + "=" * 70)
    print("  4.1.1 계절별 요약 통계 (재현)")
    print("=" * 70)
    season_order = ["여름", "겨울", "봄", "가을"]
    summary = (combined.groupby("season")["ar_nrmse"]
               .agg(개수="count", 평균="mean", 최소="min", 최대="max", 표준편차="std")
               .reindex(season_order))
    summary = summary.round(1)
    print(summary.to_string())
    summary_path = os.path.join(OUT_DIR, "season_summary_stats_ar_lad.csv")
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print(f"\n요약 통계 저장: {summary_path}")

    print("\n" + "=" * 70)
    print("  전체 블록 중 AR nRMSE 상위 10개")
    print("=" * 70)
    top10 = combined.sort_values("ar_nrmse").head(10)
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
