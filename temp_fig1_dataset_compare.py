# -*- coding: utf-8 -*-
# temp_fig1_dataset_compare.py
# 4.1.2절 계절별 대표 블록 4개(여름 z03/18, 겨울 z01/1, 봄 z03/3, 가을 z02/21)에 대해
# 논문 Fig.1(=DA/RT 가격이 하루 중 시간대별로 어떻게 다른지) 스타일 그래프를 그리고,
# 태양광 발전 프로파일도 같이 그려서 "논문의 데이터셋 특성"(DA > RT at all hours)과 비교한다.

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트 (있으면 사용)
for name in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    if any(name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# 4개 대표 블록: (라벨, zone파일, 전체구간 시작, 전체구간 끝)
BLOCKS = [
    ("여름 (z03/블록18)", "z03", "2013-08-25", "2013-12-22"),
    ("겨울 (z01/블록1)",  "z01", "2012-04-02", "2012-07-30"),
    ("봄 (z03/블록3)",    "z03", "2012-06-01", "2012-09-28"),
    ("가을 (z01/블록21)", "z01", "2013-01-26", "2013-05-25"),
]

data = {}
for label, zone, start, end in BLOCKS:
    df = pd.read_csv(f"merged_for_simulation_{zone}.csv", parse_dates=["local_date"])
    mask = (df["local_date"] >= start) & (df["local_date"] <= end)
    blk = df.loc[mask].copy()
    data[label] = blk
    print(f"{label}: {zone}, {start}~{end}, n={len(blk)}행 (기대값 120일×24시간=2880 근방)")

# ---- 시간대별 평균 DA/RT/발전량 (요약 통계는 기존과 동일하게 평균 기준) ----
# 위 행 그래프만 논문 Fig.1과 같은 "시간대별 박스플롯"으로 교체한다.
COLOR_DA = "#4472C4"   # 논문 Fig.1의 파랑(Day-ahead price)과 동일 계열
COLOR_RT = "#ED7D31"   # 논문 Fig.1의 주황(Real-time price)과 동일 계열

summary_rows = []
fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=False)

for i, (label, zone, start, end) in enumerate(BLOCKS):
    blk = data[label]
    hourly = blk.groupby("local_hour").agg(
        da=("da_price", "mean"),
        rt=("rt_price", "mean"),
        solar=("solar_power", "mean"),
    ).reindex(range(24))

    n_hours_da_gt_rt = (hourly["da"] > hourly["rt"]).sum()
    avg_gap = (hourly["da"] - hourly["rt"]).mean()
    summary_rows.append({
        "block": label, "zone": zone,
        "avg_DA": hourly["da"].mean(), "avg_RT": hourly["rt"].mean(),
        "avg_DA_minus_RT": avg_gap,
        "hours_DA_gt_RT": n_hours_da_gt_rt,
        "peak_solar_hour": int(hourly["solar"].idxmax()),
        "peak_solar_value": hourly["solar"].max(),
        "daylight_hours": int((hourly["solar"] > 0.01).sum()),
    })

    # ── 위 행: 논문 Fig.1과 같은 형식 — 시간대별 DA/RT 박스플롯 ──
    # (x축은 이 프로젝트의 기존 관례인 local_hour 0~23을 그대로 쓴다 — 나머지 절에서도 hour=0~23로 얘기하므로
    #  전체 보고서와 축 정의를 통일하기 위함이고, "박스플롯 형식"만 논문 Fig.1을 따른 것이다.)
    ax1 = axes[0, i]
    da_by_hour = [blk.loc[blk["local_hour"] == h, "da_price"].dropna().values for h in range(24)]
    rt_by_hour = [blk.loc[blk["local_hour"] == h, "rt_price"].dropna().values for h in range(24)]
    positions_da = [h - 0.18 for h in range(24)]
    positions_rt = [h + 0.18 for h in range(24)]

    flier_style = dict(marker="x", markersize=2.5, markeredgecolor="#888888", alpha=0.5)
    bp_da = ax1.boxplot(da_by_hour, positions=positions_da, widths=0.32, patch_artist=True,
                         showfliers=False, flierprops=flier_style)
    bp_rt = ax1.boxplot(rt_by_hour, positions=positions_rt, widths=0.32, patch_artist=True,
                         showfliers=False, flierprops=flier_style)
    for box in bp_da["boxes"]:
        box.set(facecolor=COLOR_DA, edgecolor="black", linewidth=0.6)
    for box in bp_rt["boxes"]:
        box.set(facecolor=COLOR_RT, edgecolor="black", linewidth=0.6)
    for bp in (bp_da, bp_rt):
        for part in ("whiskers", "caps", "medians"):
            for line in bp[part]:
                line.set(color="black", linewidth=0.6)

    ax1.set_xticks(list(range(0, 24, 2)), labels=[str(h) for h in range(0, 24, 2)])
    ax1.set_xlim(-0.7, 23.7)
    ax1.set_title(label, fontsize=11)
    ax1.set_xlabel("Hours")
    if i == 0:
        ax1.set_ylabel("Price")
    ax1.grid(alpha=0.3, axis="y")

    ax2 = axes[1, i]
    ax2.plot(hourly.index, hourly["solar"], "-^", color="#e67e22", label="태양광 발전량(정규화)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(-0.7, 23.7)
    ax2.set_xlabel("hour")
    if i == 0:
        ax2.set_ylabel("평균 발전량")
    ax2.grid(alpha=0.3)

# 논문처럼 범례를 그림 맨 위 중앙에 한 번만 표시
handles = [bp_da["boxes"][0], bp_rt["boxes"][0]]
fig.legend(handles, ["Day-ahead price (DA)", "Real-time price (RT)"],
           loc="upper center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))

fig.suptitle("계절별 대표 블록 4개 — 시간대별 DA/RT 가격 박스플롯(위, 논문 Fig.1 형식) vs 태양광 발전 프로파일(아래)",
             fontsize=13, y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(OUT_DIR, "fig1_dataset_season_compare.png")
fig.savefig(out_path, dpi=150)
print("saved:", out_path)

summary = pd.DataFrame(summary_rows)
pd.set_option("display.width", 160)
print()
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT_DIR, "fig1_dataset_season_compare_summary.csv"), index=False)
