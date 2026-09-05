# -*- coding: utf-8 -*-

# =====================================================================
# fig3_fig5_그리기.py
#
# 목적: model_proposed_ar_ver2.py 가 만든 두 CSV
#       (model_proposed_ar_ver2_fig3.csv, model_proposed_ar_ver2_fig5.csv)를
#       읽어서 Fig.3, Fig.5를 논문값과 나란히 그린다.
#
#   - Fig.3: 한 그래프에 nRMSE(왼쪽 y축)와 Gap(오른쪽 y축)을 같이 그리는
#            dual y축 방식 (fig3_그리기.py는 nRMSE/Gap을 좌우 두 그래프로
#            나눴지만, 이 스크립트는 하나로 합침)
#   - Fig.5: fig5_그리기.py와 동일하게 nRMSE/Gap을 좌우 두 그래프로 나눔
#            (AR과 제안모형을 한 그래프 안에 같이 표시)
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
FIG3_CSV_PATH = os.path.join(BASE_DIR, "results", "model_proposed_ar_ver2_fig3.csv")
FIG5_CSV_PATH = os.path.join(BASE_DIR, "results", "model_proposed_ar_ver2_fig5.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 2. Fig.3 CSV 읽기 (model_proposed_ar_ver2.py 가 만든 파일)
#    columns: Label, nRMSE, Gap, Paper_nRMSE, Paper_Gap
# =====================================================================
fig3_labels = []              # x축에 쓸 "AR","1/20",... 라벨 목록
fig3_nrmse_list = []           # 재현 nRMSE 값을 담을 빈 리스트
fig3_gap_list = []              # 재현 gap 값을 담을 빈 리스트

with open(FIG3_CSV_PATH, "r", newline="") as f:          # Fig.3 CSV를 읽기 모드로 염
    reader = csv.DictReader(f)                              # 첫 줄을 열 이름으로 쓰는 방식으로 읽음
    for row in reader:                                         # 한 줄씩 순서대로 확인
        fig3_labels.append(row["Label"])                         # 라벨(W1/W2 표기)을 추가
        fig3_nrmse_list.append(float(row["nRMSE"]))                # 재현 nRMSE를 추가
        fig3_gap_list.append(float(row["Gap"]))                      # 재현 gap을 추가

x3_positions = list(range(len(fig3_labels)))              # x축에 쓸 위치(0,1,2,...,10) 목록


# =====================================================================
# 2-1. 논문 Fig.3 참고값 (fig3_그리기.py와 동일한 값 - 논문 Table 3/Fig.3)
# =====================================================================
fig3_paper_nrmse = [34.76, 34.89, 35.14, 36.28, 41.09, 44.95, 46.11, 48.27, 49.21, 49.61, 50.07]
fig3_paper_gap   = [15.04, 13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36]


# =====================================================================
# 3. Fig.5 CSV 읽기 (model_proposed_ar_ver2.py 가 만든 파일)
#    columns: Rate, AR_nRMSE, AR_Gap, Proposed_nRMSE, Proposed_Gap
# =====================================================================
fig5_rates = []                # 벌금비용률(0.0~1.0) 값을 담을 빈 리스트
fig5_ar_nrmse_list = []         # 재현 AR nRMSE 값을 담을 빈 리스트
fig5_ar_gap_list = []            # 재현 AR gap 값을 담을 빈 리스트
fig5_prop_nrmse_list = []         # 재현 제안모형 nRMSE 값을 담을 빈 리스트
fig5_prop_gap_list = []            # 재현 제안모형 gap 값을 담을 빈 리스트

with open(FIG5_CSV_PATH, "r", newline="") as f:           # Fig.5 CSV를 읽기 모드로 염
    reader = csv.DictReader(f)                               # 첫 줄을 열 이름으로 쓰는 방식으로 읽음
    for row in reader:                                          # 한 줄씩 순서대로 확인
        fig5_rates.append(float(row["Rate"]))                     # 벌금비용률을 추가
        fig5_ar_nrmse_list.append(float(row["AR_nRMSE"]))           # 재현 AR nRMSE를 추가
        fig5_ar_gap_list.append(float(row["AR_Gap"]))                 # 재현 AR gap을 추가
        fig5_prop_nrmse_list.append(float(row["Proposed_nRMSE"]))       # 재현 제안모형 nRMSE를 추가
        fig5_prop_gap_list.append(float(row["Proposed_Gap"]))             # 재현 제안모형 gap을 추가

fig5_x_labels = [f"{int(round(r * 100))}%" for r in fig5_rates]   # x축에 쓸 "0%","10%",... 형태의 문자열 목록
x5_positions = list(range(len(fig5_x_labels)))              # x축에 쓸 위치(0,1,2,...,10) 목록


# =====================================================================
# 3-1. 논문 Fig.5 참고값 (fig5_그리기.py와 동일한 값 - 육안 판독, 50%만
#      Table 3 실측치이고 나머지는 근사치)
# =====================================================================
fig5_paper_ar_nrmse   = [34.76] * 11
fig5_paper_ar_gap     = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]
fig5_paper_prop_nrmse = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]
fig5_paper_prop_gap   = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]


# =====================================================================
# 4. Fig.3 그리기 — 하나의 그래프에 nRMSE(왼쪽 y축) + Gap(오른쪽 y축)
# =====================================================================
figure3, axis_left = plt.subplots(figsize=(10, 6))          # 그래프판 하나 생성 (왼쪽 y축)
figure3.suptitle("Fig.3 재현 — 논문 제안 모형 AR, 벌금율 50%, W1/W2 스윕 (dual y축)", fontsize=13)

axis_right = axis_left.twinx()                                 # 같은 x축을 공유하는 오른쪽 y축 생성

# --- nRMSE는 왼쪽 y축(파랑 계열), 실선 + 동그라미 마커 ---
line1 = axis_left.plot(x3_positions, fig3_paper_nrmse, marker="o", color="#2a78d6",
                        linestyle="-", label="논문 nRMSE")
line2 = axis_left.plot(x3_positions, fig3_nrmse_list, marker="o", color="#0b3d78",
                        linestyle="-", label="재현 nRMSE")

# --- Gap은 오른쪽 y축(주황/빨강 계열), 점선 + 네모 마커 (nRMSE와 시각적으로 구분) ---
line3 = axis_right.plot(x3_positions, fig3_paper_gap, marker="s", color="#eb9834",
                         linestyle="--", label="논문 Gap")
line4 = axis_right.plot(x3_positions, fig3_gap_list, marker="s", color="#b34700",
                         linestyle="--", label="재현 Gap")

axis_left.set_xticks(x3_positions)                             # x축 눈금 위치 지정
axis_left.set_xticklabels(fig3_labels)                           # x축 눈금 이름(라벨) 지정
axis_left.set_xlabel("W1/W2")                                       # x축 이름
axis_left.set_ylabel("nRMSE (%)", color="#0b3d78")                    # 왼쪽 y축 이름(파랑)
axis_right.set_ylabel("Optimality Gap (%)", color="#b34700")            # 오른쪽 y축 이름(주황)
axis_left.tick_params(axis="y", labelcolor="#0b3d78")                     # 왼쪽 y축 눈금 색
axis_right.tick_params(axis="y", labelcolor="#b34700")                      # 오른쪽 y축 눈금 색
axis_left.grid(True, alpha=0.3)                                              # 격자선을 옅게 표시

all_lines = line1 + line2 + line3 + line4                       # 네 선의 핸들을 하나로 합침
all_labels = [one_line.get_label() for one_line in all_lines]     # 각 선의 범례 이름을 뽑음
axis_left.legend(all_lines, all_labels, loc="upper left", fontsize=9)   # 왼쪽 위에 통합 범례 표시

figure3.tight_layout()                                             # 여백을 자동으로 깔끔하게 정리
fig3_output_path = os.path.join(OUTPUT_DIR, "model_proposed_ar_ver2_fig3.png")   # 저장할 파일 경로
figure3.savefig(fig3_output_path, dpi=150)                            # 150dpi 해상도로 이미지 저장
print("저장 완료:", fig3_output_path)                                    # 저장된 경로를 화면에 출력


# =====================================================================
# 5. Fig.5 그리기 — nRMSE / Gap 을 좌우 두 그래프로 나눠서 (AR + 제안모형)
# =====================================================================
figure5, (axis_nrmse, axis_gap) = plt.subplots(1, 2, figsize=(13, 5))   # 1행 2열 그래프판 생성
figure5.suptitle("Fig.5 재현 — 논문 제안 모형 AR, W1/W2=1/1, 벌금비용률 스윕", fontsize=14)

# --- 왼쪽 그래프: nRMSE (%) - 논문(점선) + 재현(실선) ---
axis_nrmse.plot(x5_positions, fig5_paper_ar_nrmse, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 AR")
axis_nrmse.plot(x5_positions, fig5_paper_prop_nrmse, linestyle="--", color="#eb6834", alpha=0.6, label="논문 제안모형")
axis_nrmse.plot(x5_positions, fig5_ar_nrmse_list, marker="o", color="#2a78d6", label="재현 AR")
axis_nrmse.plot(x5_positions, fig5_prop_nrmse_list, marker="o", color="#eb6834", label="재현 제안모형")
axis_nrmse.set_xticks(x5_positions)                     # x축 눈금 위치 지정
axis_nrmse.set_xticklabels(fig5_x_labels)                 # x축 눈금 이름(라벨) 지정
axis_nrmse.set_xlabel("벌금비용률(penalty cost rate)")       # x축 이름
axis_nrmse.set_ylabel("nRMSE (%)")                          # y축 이름
axis_nrmse.set_title("nRMSE 비교")                            # 이 그래프의 소제목
axis_nrmse.grid(True, alpha=0.3)                              # 격자선을 옅게 표시
axis_nrmse.legend(fontsize=8)                                    # 범례 표시

# --- 오른쪽 그래프: Optimality Gap (%) - 논문(점선) + 재현(실선) ---
axis_gap.plot(x5_positions, fig5_paper_ar_gap, linestyle="--", color="#2a78d6", alpha=0.6, label="논문 AR")
axis_gap.plot(x5_positions, fig5_paper_prop_gap, linestyle="--", color="#eb6834", alpha=0.6, label="논문 제안모형")
axis_gap.plot(x5_positions, fig5_ar_gap_list, marker="o", color="#2a78d6", label="재현 AR")
axis_gap.plot(x5_positions, fig5_prop_gap_list, marker="o", color="#eb6834", label="재현 제안모형")
axis_gap.set_xticks(x5_positions)                       # x축 눈금 위치 지정
axis_gap.set_xticklabels(fig5_x_labels)                   # x축 눈금 이름(라벨) 지정
axis_gap.set_xlabel("벌금비용률(penalty cost rate)")           # x축 이름
axis_gap.set_ylabel("Optimality Gap (%)")                    # y축 이름
axis_gap.set_title("Optimality Gap 비교")                      # 이 그래프의 소제목
axis_gap.grid(True, alpha=0.3)                                # 격자선을 옅게 표시
axis_gap.legend(fontsize=8)                                      # 범례 표시

figure5.tight_layout()                                            # 여백을 자동으로 깔끔하게 정리
fig5_output_path = os.path.join(OUTPUT_DIR, "model_proposed_ar_ver2_fig5.png")   # 저장할 파일 경로
figure5.savefig(fig5_output_path, dpi=150)                           # 150dpi 해상도로 이미지 저장
print("저장 완료:", fig5_output_path)                                   # 저장된 경로를 화면에 출력
