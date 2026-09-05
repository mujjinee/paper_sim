# -*- coding: utf-8 -*-

# =====================================================================
# model_baseline_ar_profit_change_v2_그리기.py
#
# 목적: model_baseline_ar_profit_change_v2.py 가 만든 두 CSV
#       (model_baseline_ar_profit_change_v2_fig3.csv,
#        model_baseline_ar_profit_change_v2_fig5.csv)를 읽어서
#       "기본 모형 AR"(4항 이익함수) 를 논문값과 나란히 그린다.
#
#   - 기본 모형 AR은 제안 모형(MILP)과 달리 W1/W2 손잡이가 없어서, 원본
#     fig3_fig5_그리기.py 처럼 W1/W2 축을 따라 여러 점을 잇는 선 그래프를
#     그릴 수 없다. 대신:
#       · Fig.3: 벌금율 50%에서 "논문 AR" vs "재현 AR" 딱 한 점을
#                nRMSE/Gap 두 항목에 대해 막대그래프로 비교한다.
#       · Fig.5: 원본과 동일하게 nRMSE/Gap을 좌우 두 그래프로 나누되,
#                AR 곡선(논문 vs 재현)만 그린다 (제안모형 곡선 없음).
#
# 코딩 스타일: class, def(함수)를 쓰지 않는다. 위에서 아래로 순서대로
#             실행되는 코드만 쓴다(naive 스타일). 거의 모든 줄에 주석을 단다.
# =====================================================================

import matplotlib                             # 그래프를 그리는 라이브러리
matplotlib.use("Agg")                          # 화면 없이 파일로만 저장하는 백엔드 지정
import matplotlib.pyplot as plt                # 실제 그리기 함수들이 들어있는 서브모듈
import matplotlib.font_manager as fm           # 한글 폰트를 찾아 쓰기 위한 서브모듈
import os                                      # 파일 경로를 다루는 표준 라이브러리
import csv                                     # CSV 파일을 읽기 위한 표준 라이브러리


# =====================================================================
# 0. 한글이 깨지지 않도록 시스템에 있는 한글 폰트를 찾아서 지정
# =====================================================================
korean_font_candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic"]  # 흔한 한글 폰트 이름들
found_font_name = None                            # 찾은 폰트 이름을 담을 변수(못 찾으면 None)
for font in fm.fontManager.ttflist:                 # 시스템에 설치된 폰트를 하나씩 확인
    if font.name in korean_font_candidates:            # 후보 이름과 일치하면
        found_font_name = font.name                       # 그 이름을 저장
        break                                               # 더 볼 필요 없으니 반복 종료
if found_font_name:                                # 한글 폰트를 찾았으면
    plt.rcParams["font.family"] = found_font_name       # 전역 폰트로 지정
plt.rcParams["axes.unicode_minus"] = False           # 마이너스 기호가 깨지는 것 방지


# =====================================================================
# 1. 경로 설정
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                        # 이 파일이 있는 폴더
FIG3_CSV_PATH = os.path.join(BASE_DIR, "results", "model_baseline_ar_profit_change_v2_fig3.csv")
FIG5_CSV_PATH = os.path.join(BASE_DIR, "results", "model_baseline_ar_profit_change_v2_fig5.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 2. Fig.3 CSV 읽기 (model_baseline_ar_profit_change_v2.py 가 만든 파일)
#    columns: Label, nRMSE, Gap, Paper_nRMSE, Paper_Gap - "AR" 한 줄뿐
# =====================================================================
with open(FIG3_CSV_PATH, "r", newline="") as f:          # Fig.3 CSV를 읽기 모드로 염
    reader = csv.DictReader(f)                              # 첫 줄을 열 이름으로 쓰는 방식으로 읽음
    fig3_row = next(reader)                                   # AR 한 줄뿐이므로 첫 줄만 꺼냄

fig3_ar_nrmse = float(fig3_row["nRMSE"])                  # 재현 AR nRMSE
fig3_ar_gap = float(fig3_row["Gap"])                        # 재현 AR gap
fig3_paper_nrmse = float(fig3_row["Paper_nRMSE"])             # 논문 AR nRMSE
fig3_paper_gap = float(fig3_row["Paper_Gap"])                   # 논문 AR gap


# =====================================================================
# 3. Fig.5 CSV 읽기 (model_baseline_ar_profit_change_v2.py 가 만든 파일)
#    columns: Rate, AR_nRMSE, AR_Gap
# =====================================================================
fig5_rates = []                # 벌금비용률(0.0~1.0) 값을 담을 빈 리스트
fig5_ar_nrmse_list = []         # 재현 AR nRMSE 값을 담을 빈 리스트
fig5_ar_gap_list = []            # 재현 AR gap 값을 담을 빈 리스트

with open(FIG5_CSV_PATH, "r", newline="") as f:           # Fig.5 CSV를 읽기 모드로 염
    reader = csv.DictReader(f)                               # 첫 줄을 열 이름으로 쓰는 방식으로 읽음
    for row in reader:                                          # 한 줄씩 순서대로 확인
        fig5_rates.append(float(row["Rate"]))                     # 벌금비용률을 추가
        fig5_ar_nrmse_list.append(float(row["AR_nRMSE"]))           # 재현 AR nRMSE를 추가
        fig5_ar_gap_list.append(float(row["AR_Gap"]))                 # 재현 AR gap을 추가

fig5_x_labels = [f"{int(round(r * 100))}%" for r in fig5_rates]   # x축에 쓸 "0%","10%",... 형태의 문자열 목록
x5_positions = list(range(len(fig5_x_labels)))              # x축에 쓸 위치(0,1,2,...,10) 목록


# =====================================================================
# 3-1. 논문 Fig.5 AR 참고값 (fig3_fig5_그리기.py와 동일한 값 - 육안 판독,
#      50%만 Table 3 실측치이고 나머지는 근사치)
# =====================================================================
fig5_paper_ar_nrmse = [34.76] * 11
fig5_paper_ar_gap = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]


# =====================================================================
# 4. Fig.3 그리기 — "논문 AR" vs "재현 AR" 막대그래프 (nRMSE, Gap 각각)
# =====================================================================
figure3, (axis_nrmse3, axis_gap3) = plt.subplots(1, 2, figsize=(9, 5))   # 1행 2열 그래프판 생성
figure3.suptitle("Fig.3 재현 — 기본 모형 AR(4항 이익함수), 벌금율 50%", fontsize=13)

bar_positions = [0, 1]                                     # 막대 두 개(논문, 재현)의 x 위치
bar_labels = ["논문 AR", "재현 AR"]                            # 막대 아래 표시할 이름

# --- 왼쪽: nRMSE 막대 ---
axis_nrmse3.bar(bar_positions, [fig3_paper_nrmse, fig3_ar_nrmse],
                 color=["#2a78d6", "#0b3d78"], width=0.5)
axis_nrmse3.set_xticks(bar_positions)
axis_nrmse3.set_xticklabels(bar_labels)
axis_nrmse3.set_ylabel("nRMSE (%)")
axis_nrmse3.set_title("nRMSE 비교")
axis_nrmse3.grid(True, axis="y", alpha=0.3)
for x, y in zip(bar_positions, [fig3_paper_nrmse, fig3_ar_nrmse]):        # 막대 위에 값 라벨 표시
    axis_nrmse3.text(x, y, f"{y:.2f}%", ha="center", va="bottom", fontsize=9)

# --- 오른쪽: Gap 막대 ---
axis_gap3.bar(bar_positions, [fig3_paper_gap, fig3_ar_gap],
               color=["#eb9834", "#b34700"], width=0.5)
axis_gap3.set_xticks(bar_positions)
axis_gap3.set_xticklabels(bar_labels)
axis_gap3.set_ylabel("Optimality Gap (%)")
axis_gap3.set_title("Optimality Gap 비교")
axis_gap3.grid(True, axis="y", alpha=0.3)
for x, y in zip(bar_positions, [fig3_paper_gap, fig3_ar_gap]):           # 막대 위에 값 라벨 표시
    axis_gap3.text(x, y, f"{y:.2f}%", ha="center", va="bottom", fontsize=9)

figure3.tight_layout()                                             # 여백을 자동으로 깔끔하게 정리
fig3_output_path = os.path.join(OUTPUT_DIR, "model_baseline_ar_profit_change_v2_fig3.png")
figure3.savefig(fig3_output_path, dpi=150)                            # 150dpi 해상도로 이미지 저장
print("저장 완료:", fig3_output_path)                                    # 저장된 경로를 화면에 출력


# =====================================================================
# 5. Fig.5 그리기 — nRMSE / Gap 을 좌우 두 그래프로 나눠서 (AR만)
# =====================================================================
figure5, (axis_nrmse5, axis_gap5) = plt.subplots(1, 2, figsize=(13, 5))   # 1행 2열 그래프판 생성
figure5.suptitle("Fig.5 재현 — 기본 모형 AR(4항 이익함수), 벌금비용률 스윕", fontsize=14)

# --- 왼쪽 그래프: nRMSE (%) - 논문(점선) + 재현(실선) ---
axis_nrmse5.plot(x5_positions, fig5_paper_ar_nrmse, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 AR")
axis_nrmse5.plot(x5_positions, fig5_ar_nrmse_list, marker="o", color="#2a78d6", label="재현 AR")
axis_nrmse5.set_xticks(x5_positions)                     # x축 눈금 위치 지정
axis_nrmse5.set_xticklabels(fig5_x_labels)                 # x축 눈금 이름(라벨) 지정
axis_nrmse5.set_xlabel("벌금비용률(penalty cost rate)")       # x축 이름
axis_nrmse5.set_ylabel("nRMSE (%)")                          # y축 이름
axis_nrmse5.set_title("nRMSE 비교")                            # 이 그래프의 소제목
axis_nrmse5.grid(True, alpha=0.3)                              # 격자선을 옅게 표시
axis_nrmse5.legend(fontsize=9)                                    # 범례 표시

# --- 오른쪽 그래프: Optimality Gap (%) - 논문(점선) + 재현(실선) ---
axis_gap5.plot(x5_positions, fig5_paper_ar_gap, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 AR")
axis_gap5.plot(x5_positions, fig5_ar_gap_list, marker="o", color="#2a78d6", label="재현 AR")
axis_gap5.set_xticks(x5_positions)                       # x축 눈금 위치 지정
axis_gap5.set_xticklabels(fig5_x_labels)                   # x축 눈금 이름(라벨) 지정
axis_gap5.set_xlabel("벌금비용률(penalty cost rate)")           # x축 이름
axis_gap5.set_ylabel("Optimality Gap (%)")                    # y축 이름
axis_gap5.set_title("Optimality Gap 비교")                      # 이 그래프의 소제목
axis_gap5.grid(True, alpha=0.3)                                # 격자선을 옅게 표시
axis_gap5.legend(fontsize=9)                                      # 범례 표시

figure5.tight_layout()                                            # 여백을 자동으로 깔끔하게 정리
fig5_output_path = os.path.join(OUTPUT_DIR, "model_baseline_ar_profit_change_v2_fig5.png")
figure5.savefig(fig5_output_path, dpi=150)                           # 150dpi 해상도로 이미지 저장
print("저장 완료:", fig5_output_path)                                   # 저장된 경로를 화면에 출력
