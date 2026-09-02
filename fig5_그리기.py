# -*- coding: utf-8 -*-

# =====================================================================
# fig5_그리기.py
#
# 목적: 논문 Fig.5(벌금비용률 0~100%에 따른 AR·제안모형의 nRMSE·
#       optimality gap)를, temp_논문AR_수정_fig5데이터추출.py 가 만든
#       결과 CSV(results/temp_fig5_논문AR수정_결과.csv)를 읽어서 그린다.
#
# 코딩 스타일: class, def(함수) 를 전혀 쓰지 않는다. 위에서 아래로
#             순서대로 실행되는 코드만 쓴다(naive 스타일). 거의 모든
#             줄에 그 줄이 뭘 하는지 주석을 단다.
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
# 1. 결과 CSV 읽기 (temp_논문AR_수정_fig5데이터추출.py 가 만든 파일)
# =====================================================================
csv_path = os.path.join(os.getcwd(), "results", "temp_fig5_논문AR수정_결과.csv")  # 읽어올 CSV 경로

penalty_list = []              # 벌금율(0.0~1.0) 값을 담을 빈 리스트
ar_nrmse_list = []              # AR nRMSE 값을 담을 빈 리스트
ar_gap_list = []                 # AR gap 값을 담을 빈 리스트
prop_nrmse_list = []              # 제안모형 nRMSE 값을 담을 빈 리스트
prop_gap_list = []                 # 제안모형 gap 값을 담을 빈 리스트

with open(csv_path, "r", newline="") as f:            # CSV 파일을 읽기 모드로 염
    reader = csv.DictReader(f)                            # 첫 줄을 열 이름으로 쓰는 방식으로 읽음
    for row in reader:                                       # 한 줄씩 순서대로 확인
        model_name = row["Model"]                               # 이 줄의 모형 이름("AR" 또는 "Proposed")
        penalty_value = float(row["Penalty_Rate"])                # 이 줄의 벌금율(0.0~1.0)
        nrmse_value = float(row["nRMSE"])                           # 이 줄의 nRMSE(%)
        gap_value = float(row["Gap"])                                 # 이 줄의 gap(%)

        if model_name == "AR":                                # AR 모형 줄이면
            penalty_list.append(penalty_value)                    # 벌금율을 x축 목록에 추가(AR 기준으로만 채움)
            ar_nrmse_list.append(nrmse_value)                       # AR nRMSE 값을 추가
            ar_gap_list.append(gap_value)                             # AR gap 값을 추가
        elif model_name == "Proposed":                        # 제안모형 줄이면
            prop_nrmse_list.append(nrmse_value)                     # 제안모형 nRMSE 값을 추가
            prop_gap_list.append(gap_value)                           # 제안모형 gap 값을 추가

x_labels = [f"{int(p*100)}%" for p in penalty_list]     # x축에 쓸 "0%","10%",... 형태의 문자열 목록
x_positions = list(range(len(x_labels)))                   # x축에 쓸 위치(0,1,2,...,10) 목록


# =====================================================================
# 1-1. 논문 Fig.5 그림을 육안으로 읽은 참고값 (Table 3에서 확인되는
#      50% 지점만 정확한 값이고, 나머지는 그래프 판독 근사치)
# =====================================================================
paper_ar_nrmse_list = [34.76] * 11                                        # 논문 AR nRMSE: 벌금율과 무관하게 고정
paper_ar_gap_list = [9, 11, 12, 13, 14, 15.04, 16, 18, 19, 20, 22]           # 논문 AR gap(육안, 50%만 Table3 실측)
paper_prop_nrmse_list = [46, 34, 35, 36, 40, 44.45, 50, 55, 61, 64, 67]        # 논문 제안모형 nRMSE(육안, 50%만 실측)
paper_prop_gap_list = [8, 10, 11, 11, 11, 11.44, 11, 11.5, 11.5, 11.5, 11.5]     # 논문 제안모형 gap(육안, 50%만 실측)


# =====================================================================
# 2. 그림판(figure) 준비 - 가로로 2개(nRMSE, gap) 나란히
# =====================================================================
figure, (axis_nrmse, axis_gap) = plt.subplots(1, 2, figsize=(13, 5))   # 1행 2열 그래프판 생성
figure.suptitle("Fig.5 재현 vs 논문 — 벌금비용률 스윕 (논문AR_수정, W1=W2=1)", fontsize=14)  # 전체 제목


# =====================================================================
# 3. 왼쪽 그래프: nRMSE (%) - 논문(점선) + 재현(실선)
# =====================================================================
axis_nrmse.plot(x_positions, paper_ar_nrmse_list, linestyle="--", marker="", color="#2a78d6", alpha=0.6, label="논문 AR")        # 논문 AR nRMSE(파랑 점선)
axis_nrmse.plot(x_positions, paper_prop_nrmse_list, linestyle="--", marker="", color="#eb6834", alpha=0.6, label="논문 제안모형")   # 논문 제안모형 nRMSE(주황 점선)
axis_nrmse.plot(x_positions, ar_nrmse_list, marker="o", color="#2a78d6", label="재현 AR")           # 재현 AR nRMSE 선(파랑 실선)
axis_nrmse.plot(x_positions, prop_nrmse_list, marker="o", color="#eb6834", label="재현 제안모형")      # 재현 제안모형 nRMSE 선(주황 실선)
axis_nrmse.set_xticks(x_positions)                     # x축 눈금 위치 지정
axis_nrmse.set_xticklabels(x_labels)                     # x축 눈금 이름(라벨) 지정
axis_nrmse.set_xlabel("벌금비용률(penalty cost rate)")       # x축 이름
axis_nrmse.set_ylabel("nRMSE (%)")                          # y축 이름
axis_nrmse.set_title("nRMSE 비교")                            # 이 그래프의 소제목
axis_nrmse.grid(True, alpha=0.3)                              # 격자선을 옅게 표시
axis_nrmse.legend(fontsize=8)                                    # 범례(논문/재현 x AR/제안모형) 표시


# =====================================================================
# 4. 오른쪽 그래프: Optimality Gap (%) - 논문(점선) + 재현(실선)
# =====================================================================
axis_gap.plot(x_positions, paper_ar_gap_list, linestyle="--", marker="", color="#2a78d6", alpha=0.6, label="논문 AR")        # 논문 AR gap(파랑 점선)
axis_gap.plot(x_positions, paper_prop_gap_list, linestyle="--", marker="", color="#eb6834", alpha=0.6, label="논문 제안모형")   # 논문 제안모형 gap(주황 점선)
axis_gap.plot(x_positions, ar_gap_list, marker="o", color="#2a78d6", label="재현 AR")               # 재현 AR gap 선(파랑 실선)
axis_gap.plot(x_positions, prop_gap_list, marker="o", color="#eb6834", label="재현 제안모형")          # 재현 제안모형 gap 선(주황 실선)
axis_gap.set_xticks(x_positions)                       # x축 눈금 위치 지정
axis_gap.set_xticklabels(x_labels)                       # x축 눈금 이름(라벨) 지정
axis_gap.set_xlabel("벌금비용률(penalty cost rate)")           # x축 이름
axis_gap.set_ylabel("Optimality Gap (%)")                    # y축 이름
axis_gap.set_title("Optimality Gap 비교")                      # 이 그래프의 소제목
axis_gap.grid(True, alpha=0.3)                                # 격자선을 옅게 표시
axis_gap.legend(fontsize=8)                                      # 범례(논문/재현 x AR/제안모형) 표시


# =====================================================================
# 5. 그림을 파일로 저장
# =====================================================================
figure.tight_layout()                                    # 여백을 자동으로 깔끔하게 정리
output_path = os.path.join(os.getcwd(), "results", "fig5_재현.png")  # 저장할 파일 경로
os.makedirs(os.path.dirname(output_path), exist_ok=True)     # results 폴더가 없으면 만듦
figure.savefig(output_path, dpi=150)                            # 150dpi 해상도로 이미지 저장

print("저장 완료:", output_path)                          # 저장된 경로를 화면에 출력
