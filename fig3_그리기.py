# -*- coding: utf-8 -*-

# =====================================================================
# fig3_그리기.py
#
# 목적: 논문 Fig.3(AR 제안모형의 W1/W2 비율별 nRMSE·optimality gap)을
#       논문값과 재현값을 나란히 놓고 그린다. 사용자가 직접 붙여넣은
#       표(AR, 1/20, 1/10, ..., 1/0 총 11개 지점) 데이터를 그대로 쓴다.
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
# 1. 사용자가 붙여넣은 표 데이터를 그대로 입력 (AR 제안모형, W1/W2 스윕)
# =====================================================================
labels = ["AR", "1/20", "1/10", "1/5", "1/2", "1/1", "2/1", "5/1", "10/1", "20/1", "1/0"]  # x축 눈금 이름

ours_nrmse = [36.11, 35.69, 35.68, 35.29, 36.98, 37.46, 39.92, 45.10, 54.75, 63.15, 67.84]   # 재현 nRMSE(%)
ours_gap   = [14.94, 14.54, 14.35, 14.09, 14.18, 13.70, 12.95, 11.50, 10.94, 10.33, 10.69]     # 재현 gap(%)

paper_nrmse = [34.76, 34.89, 35.14, 36.28, 41.09, 44.95, 46.11, 48.27, 49.21, 49.61, 50.07]    # 논문 nRMSE(%)
paper_gap   = [15.04, 13.91, 13.42, 12.71, 11.88, 11.44, 11.38, 11.38, 11.36, 11.36, 11.36]     # 논문 gap(%)

x_positions = list(range(len(labels)))            # x축에 쓸 위치(0,1,2,...,10) 목록


# =====================================================================
# 2. 그림판(figure) 준비 - 가로로 2개(nRMSE, gap) 나란히
# =====================================================================
figure, (axis_nrmse, axis_gap) = plt.subplots(1, 2, figsize=(13, 5))   # 1행 2열 그래프판 생성
figure.suptitle("Fig.3 재현 — AR 제안모형 W1/W2 스윕 (벌금율 50%)", fontsize=14)  # 전체 제목


# =====================================================================
# 3. 왼쪽 그래프: nRMSE (%)
# =====================================================================
axis_nrmse.plot(x_positions, paper_nrmse, marker="o", color="#2a78d6", label="논문")   # 논문 nRMSE 선(파랑)
axis_nrmse.plot(x_positions, ours_nrmse, marker="o", color="#eb6834", label="재현")     # 재현 nRMSE 선(주황)
axis_nrmse.set_xticks(x_positions)                     # x축 눈금 위치 지정
axis_nrmse.set_xticklabels(labels)                       # x축 눈금 이름(라벨) 지정
axis_nrmse.set_xlabel("W1/W2")                             # x축 이름
axis_nrmse.set_ylabel("nRMSE (%)")                          # y축 이름
axis_nrmse.set_title("nRMSE 비교")                            # 이 그래프의 소제목
axis_nrmse.grid(True, alpha=0.3)                              # 격자선을 옅게 표시
axis_nrmse.legend()                                              # 범례(논문/재현) 표시


# =====================================================================
# 4. 오른쪽 그래프: Optimality Gap (%)
# =====================================================================
axis_gap.plot(x_positions, paper_gap, marker="o", color="#2a78d6", label="논문")        # 논문 gap 선(파랑)
axis_gap.plot(x_positions, ours_gap, marker="o", color="#eb6834", label="재현")          # 재현 gap 선(주황)
axis_gap.set_xticks(x_positions)                       # x축 눈금 위치 지정
axis_gap.set_xticklabels(labels)                         # x축 눈금 이름(라벨) 지정
axis_gap.set_xlabel("W1/W2")                               # x축 이름
axis_gap.set_ylabel("Optimality Gap (%)")                    # y축 이름
axis_gap.set_title("Optimality Gap 비교")                      # 이 그래프의 소제목
axis_gap.grid(True, alpha=0.3)                                # 격자선을 옅게 표시
axis_gap.legend()                                                # 범례(논문/재현) 표시


# =====================================================================
# 5. 그림을 파일로 저장
# =====================================================================
figure.tight_layout()                                    # 여백을 자동으로 깔끔하게 정리
output_path = os.path.join(os.getcwd(), "results", "fig3_재현.png")  # 저장할 파일 경로
os.makedirs(os.path.dirname(output_path), exist_ok=True)     # results 폴더가 없으면 만듦
figure.savefig(output_path, dpi=150)                            # 150dpi 해상도로 이미지 저장

print("저장 완료:", output_path)                          # 저장된 경로를 화면에 출력
