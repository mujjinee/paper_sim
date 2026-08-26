

이제 data set이 정확하게 추출되었는지 확인해
- extract_****.py로 data를 추출했어.
그런 다음에 기본 ar, mlr이 논문하고 차이나는 이유가 무엇인지 파악해.
- 절대 추측하지 마
- 곰곰히 자세히 생각해야 돼 (시간이 오래 걸려도 좋아)

==========
검증을 끝냈습니다. 결론부터 말하면, 현재 데이터는 “추출 코드가 선택한 원본값을 잘못 옮긴 데이터”는 아닙니다. 다만 “논문 실험과 동일한 데이터셋”이라고 증명할 수도 없습니다.
상세 검증은 [13_데이터셋_추출_및_성능차이_검증.md](/D:/03_JiWon/JiWonProject/references/13_데이터셋_추출_및_성능차이_검증.md)에 기록했고, 재현 가능한 읽기 전용 검증기는 [audit_data_pipeline.py](/D:/03_JiWon/JiWonProject/audit_data_pipeline.py)입니다.
확인된 데이터 추출 결과는 다음과 같습니다.
항목	결과
Solar	원본 Task 15/predictors15.csv의 Zone 1·2014-01~04 값과 정확히 일치, 2,880행
Weather	같은 원본의 12개 VAR 열과 정확히 일치, 2,880행
POWER	train15.csv 및 Task 15 solution의 실제 POWER와 일치
DA / RT 가격	사용 가능한 MISO xls를 다시 파싱한 결과와 일치
병합 파일	현재 merge_for_simulation.py의 join·forward-fill 규칙 결과와 일치


즉, 현 CSV는 추출 파이프라인 기준으로 정확합니다.
다만 논문 기본 AR/MLR과 결과가 다른 확인된 이유는 다음입니다.
1. 실험 표본이 다릅니다.
   - 논문: 학습 300일, 테스트 100일.
   - 현재 run_simulation.py: 학습 90일(1~3월), 테스트 30일(4월).
따라서 논문 nRMSE 34.76% / 21.76%와 현재 결과를 같은 실험의 결과처럼 비교할 수 없습니다. 논문의 정확한 날짜 범위, Solar zone, MISO node는 현재 자료만으로 동일하다고 확인되지 않았습니다.
2. 기존 AR에는 테스트 이력 버그가 있습니다.
   - 기존 AR은 테스트 일자의 실제 발전량을 이력에 추가하지만, 예측 때 그 값을 읽지 않습니다.
   - 같은 데이터에서 이 버그만 고친 결과:
AR	nRMSE	Gap
기존 run_simulation.py	86.55%	28.16%
이력 인덱스 수정 후	64.54%	19.80%


AR 성능 차이에는 이 구현 오류가 실제로 기여했다고 확인했습니다.
3. 기존 MLR은 논문 MLR과 구조가 다릅니다.
   - 논문 최종 변수: SSRD, TSR, Hour.
   - 기존 코드는 시간대별 12개 회귀모델을 학습합니다.
   - 각 모델 안에서 Hour는 상수이므로 회귀계수로 식별되지 않습니다.
따라서 기존 MLR(sw-3vars)는 같은 변수 이름을 써도 논문 MLR 식과 동일하지 않습니다. 이 구조 차이가 성능 차이에 얼마나 기여하는지는 현재 자료만으로 수치 분해할 수 없습니다.
4. 현재 시간 창은 전체 일조 시간을 보존하지 않습니다.
   - 실행 시 UTC 00~11시만 사용합니다.
   - 원본 예시에서 제외되는 UTC 20~23시에도 평균 발전량 0.2169가 존재합니다.
   - 즉 “낮 시간 전체”를 쓴다는 주석과 달리, 일부 유의미한 발전 시간대가 빠집니다.
논문이 정확히 어떤 timezone 기준의 12시간 창을 썼는지는 확인되지 않았으므로, 이것을 논문과의 불일치라고 단정하지는 않습니다. 하지만 nRMSE 대상 자체를 바꾸는 중요한 미확인 항목입니다.
5. RT 가격의 마지막 하루가 잘못 대체됩니다.
   - 2014-04-30 RT 가격 24개가 없고, 병합 시 4월 29일 23시 가격 하나로 모두 forward-fill됩니다.
   - 이는 optimality gap에는 영향을 줍니다.
   - 하지만 nRMSE는 가격을 쓰지 않으므로 AR/MLR nRMSE 차이의 원인은 아닙니다.
다음 단계는 명확합니다. 논문의 300/100 표본 크기와 동일하게 Solar·Weather 전체 기간으로 AR/MLR nRMSE를 재실행하고, 시간 창을 논문 기준으로 확정해야 합니다. Optimality gap까지 논문과 비교하려면 MISO 가격도 같은 기간·동일 node·시간대 기준으로 확보해야 합니다.

Edited 2 files+181-0Review changesUndoReview

audit_data_pipeline.pyaudit_data_pipeline.py+81-0
references/13_데이터셋_추출_및_성능차이_검증.mdreferences/13_데이터셋_추출_및_성능차이_검증.md+100-0





9:41 PM


============

위 내용들도 모두 md 파일로 저장하고, 관련 py 파일 정보도 정리 내용에 넣어 주고.
- 모든 시험결과는 표로 비교 해줘

설정이 다른 경우에도 구분 열을 두어, 숫자를 동등 비교하는 것처럼 보이지 않게 정리하겠습니다.




Edited a file, ran commands



You've hit your usage limit. Upgrade your plan to continue, or try again at Sep 25, 2026, 8:32 PM.
