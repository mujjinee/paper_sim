
# -*- coding: utf-8 -*-



# =====================================================================

# method1_pc_max_da_rt_mlr_fig6_fig8_그리기.py

# method1_pc_max_da_rt_mlr_fig6_fig8.py 가 만든 두 CSV

# (fig6_method1_pc_max_da_rt_mlr.csv, fig8_method1_pc_max_da_rt_mlr.csv)를

# 읽어서 Fig.6, Fig.8을 그린다 (CodefromJiWon/fig3_fig5_그리기.py 참고).

#

#   - Fig.6: 한 그래프에 nRMSE(왼쪽 y축)와 Gap(오른쪽 y축)을 같이 그리는

#            dual y축 방식 - fig3_fig5_그리기.py의 Fig.3과 동일한 형태

#   - Fig.8: nRMSE/Gap을 좌우 두 그래프로 나눔 (MLR과 제안모형을 한

#            그래프 안에 같이 표시). 논문 그림8은 수치가 표로 없어서

#            (fig6_fig8.py 8절 참고) 재현값만 그린다 - 논문 곡선 없음.

#

# 코딩 스타일: class, def(함수)를 쓰지 않는다. 위에서 아래로 순서대로

#             실행되는 코드만 쓴다(naive 스타일).

# =====================================================================



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

import os

import csv





# =====================================================================

# 0. 한글이 깨지지 않도록 시스템에 있는 한글 폰트를 찾아서 지정

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





# =====================================================================

# 1. 경로 설정

# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FIG6_CSV_PATH = os.path.join(BASE_DIR, "results", "simulation_output", "fig6_method1_pc_max_da_rt_mlr.csv")

FIG8_CSV_PATH = os.path.join(BASE_DIR, "results", "simulation_output", "fig8_method1_pc_max_da_rt_mlr.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)





# =====================================================================

# 2. Fig.6 CSV 읽기

#    columns: Label, nRMSE, Gap, Paper_nRMSE, Paper_Gap

# =====================================================================

fig6_labels = []

fig6_nrmse_list = []

fig6_gap_list = []

fig6_paper_nrmse = []

fig6_paper_gap = []



with open(FIG6_CSV_PATH, "r", newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        fig6_labels.append(row["Label"])

        fig6_nrmse_list.append(float(row["nRMSE"]))

        fig6_gap_list.append(float(row["Gap"]))

        fig6_paper_nrmse.append(float(row["Paper_nRMSE"]))

        fig6_paper_gap.append(float(row["Paper_Gap"]))



x6_positions = list(range(len(fig6_labels)))





# =====================================================================

# 3. Fig.8 CSV 읽기

#    columns: Rate, MLR_nRMSE, MLR_Gap, Proposed_nRMSE, Proposed_Gap

# =====================================================================

fig8_rates = []

fig8_mlr_nrmse_list = []

fig8_mlr_gap_list = []

fig8_prop_nrmse_list = []

fig8_prop_gap_list = []



with open(FIG8_CSV_PATH, "r", newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        fig8_rates.append(float(row["Rate"]))

        fig8_mlr_nrmse_list.append(float(row["MLR_nRMSE"]))

        fig8_mlr_gap_list.append(float(row["MLR_Gap"]))

        fig8_prop_nrmse_list.append(float(row["Proposed_nRMSE"]))

        fig8_prop_gap_list.append(float(row["Proposed_Gap"]))



fig8_x_labels = [f"{int(round(r * 100))}%" for r in fig8_rates]

x8_positions = list(range(len(fig8_x_labels)))





# =====================================================================

# 4. Fig.6 그리기 — 하나의 그래프에 nRMSE(왼쪽 y축) + Gap(오른쪽 y축)

#    (fig3_fig5_그리기.py 의 Fig.3 dual y축 방식과 동일)

# =====================================================================

figure6, axis_left = plt.subplots(figsize=(10, 6))

figure6.suptitle("Fig.6 재현 — MLR 제안 모형, 벌금율 50%, W1/W2 스윕 (dual y축)", fontsize=13)



axis_right = axis_left.twinx()



# --- nRMSE는 왼쪽 y축(파랑 계열), 실선 + 동그라미 마커 ---

line1 = axis_left.plot(x6_positions, fig6_paper_nrmse, marker="o", color="#2a78d6",

                        linestyle="-", label="논문 nRMSE")

line2 = axis_left.plot(x6_positions, fig6_nrmse_list, marker="o", color="#0b3d78",

                        linestyle="-", label="재현 nRMSE")



# --- Gap은 오른쪽 y축(주황/빨강 계열), 점선 + 네모 마커 ---

line3 = axis_right.plot(x6_positions, fig6_paper_gap, marker="s", color="#eb9834",

                         linestyle="--", label="논문 Gap")

line4 = axis_right.plot(x6_positions, fig6_gap_list, marker="s", color="#b34700",

                         linestyle="--", label="재현 Gap")



axis_left.set_xticks(x6_positions)

axis_left.set_xticklabels(fig6_labels)

axis_left.set_xlabel("W1/W2")

axis_left.set_ylabel("nRMSE (%)", color="#0b3d78")

axis_right.set_ylabel("Optimality Gap (%)", color="#b34700")

axis_left.tick_params(axis="y", labelcolor="#0b3d78")

axis_right.tick_params(axis="y", labelcolor="#b34700")

axis_left.grid(True, alpha=0.3)



all_lines = line1 + line2 + line3 + line4

all_labels = [one_line.get_label() for one_line in all_lines]

axis_left.legend(all_lines, all_labels, loc="upper left", fontsize=9)



figure6.tight_layout()

fig6_output_path = os.path.join(OUTPUT_DIR, "fig6_method1_pc_max_da_rt_mlr.png")

figure6.savefig(fig6_output_path, dpi=150)

print("저장 완료:", fig6_output_path)





# =====================================================================

# 5. Fig.8 그리기 — nRMSE / Gap 을 좌우 두 그래프로 나눠서 (MLR + 제안모형)

#    논문 수치가 표로 없어서 재현값만 그림 (논문 곡선 없음)

# =====================================================================

figure8, (axis_nrmse, axis_gap) = plt.subplots(1, 2, figsize=(13, 5))

figure8.suptitle("Fig.8 재현 — MLR 제안 모형, W1/W2=1/1, 벌금비용률 스윕", fontsize=14)



# --- 왼쪽 그래프: nRMSE (%) ---

axis_nrmse.plot(x8_positions, fig8_mlr_nrmse_list, marker="o", color="#2a78d6", label="재현 MLR")

axis_nrmse.plot(x8_positions, fig8_prop_nrmse_list, marker="o", color="#eb6834", label="재현 제안모형")

axis_nrmse.set_xticks(x8_positions)

axis_nrmse.set_xticklabels(fig8_x_labels)

axis_nrmse.set_xlabel("벌금비용률(penalty cost rate)")

axis_nrmse.set_ylabel("nRMSE (%)")

axis_nrmse.set_title("nRMSE 비교")

axis_nrmse.grid(True, alpha=0.3)

axis_nrmse.legend(fontsize=9)



# --- 오른쪽 그래프: Optimality Gap (%) ---

axis_gap.plot(x8_positions, fig8_mlr_gap_list, marker="o", color="#2a78d6", label="재현 MLR")

axis_gap.plot(x8_positions, fig8_prop_gap_list, marker="o", color="#eb6834", label="재현 제안모형")

axis_gap.set_xticks(x8_positions)

axis_gap.set_xticklabels(fig8_x_labels)

axis_gap.set_xlabel("벌금비용률(penalty cost rate)")

axis_gap.set_ylabel("Optimality Gap (%)")

axis_gap.set_title("Optimality Gap 비교")

axis_gap.grid(True, alpha=0.3)

axis_gap.legend(fontsize=9)



figure8.tight_layout()

fig8_output_path = os.path.join(OUTPUT_DIR, "fig8_method1_pc_max_da_rt_mlr.png")

figure8.savefig(fig8_output_path, dpi=150)

print("저장 완료:", fig8_output_path)
