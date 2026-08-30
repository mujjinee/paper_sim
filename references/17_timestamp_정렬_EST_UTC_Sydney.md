# predictors15.csv ↔ da_lmp_prices.csv timestamp 정렬 정리

## 작업 날짜

2026-08-29 ~ 2026-08-30

## 배경

`predictors15.csv`(GEFCom2014 태양광/기상)와 `da_lmp_prices.csv`/`rt_lmp_prices.csv`
(MISO 가격)를 `timestamp` 기준으로 병합해야 하는데, 두 데이터가 서로 다른 시간대
(timezone)로 되어 있어서 그냥 합치면 안 됐다. 이 문서는 이 문제를 어떻게 찾고
고쳤는지 정리한다. **서로 다른 두 가지 문제**를 순서대로 고쳤다는 점이 핵심이다:

1. 가격↔태양광을 같은 시간 축으로 맞추는 문제 (EST → UTC)
2. "하루 중 낮 시간대가 몇 시냐"를 정하는 문제 (UTC → Sydney 현지시간)

---

## 1단계: 두 데이터가 각각 어떤 시간대인지 확인

| 데이터 | 시간대 |
|---|---|
| `predictors15.csv` (GEFCom2014 태양광/기상) | **UTC** (프로젝트 관례상 그렇게 취급) |
| `da_lmp_prices.csv`, `rt_lmp_prices.csv` (MISO 가격) | **고정 EST (UTC-5, 서머타임 미적용)** |

MISO가 EST 고정이라는 건 추측이 아니라 **원본 엑셀 헤더를 직접 확인**해서
나온 사실이다. `data_raw/miso/raw/`의 DA/RT 엑셀 파일을 열어보면:

```
Peak Hour: HE 21 (EST)
Minimum Hour: HE 04 (EST)
```

DA/RT 엑셀 둘 다 이렇게 명시돼 있다. MISO는 여름에도 시계를 안 바꾸는 EST 고정
발표 방식을 쓴다 — 이게 원본 엑셀이 서머타임과 무관하게 항상 "Hour 01~Hour 24"
24개 고정 포맷인 이유이기도 하다 (실제 서머타임 전환일인 2014-03-09를 열어봐도
24개 그대로였음을 확인).

---

## 2단계: 가격 timestamp에 +5시간을 더해서 UTC로 맞춤

`build_merged_for_simulation.py`의 `load_prices()`에서:

```python
EST_TO_UTC = pd.Timedelta(hours=5)   # MISO는 고정 EST(UTC-5) 발표, DST 미반영

da["timestamp"] = da["timestamp"] + EST_TO_UTC   # EST -> UTC
rt["timestamp"] = rt["timestamp"] + EST_TO_UTC   # EST -> UTC
```

이렇게 하면 두 데이터의 `timestamp` 열이 **같은 기준(UTC)**이 되고, 그제서야
`pd.merge(..., on="timestamp")`로 안전하게 합칠 수 있다.

### 보정 전에는 무엇이 문제였나

보정 전에는 가격과 태양광/기상이 **5시간 어긋난 채로 merge**되고 있었다.
AR nRMSE처럼 가격을 아예 안 쓰는 계산엔 영향이 없었지만, optimality gap처럼
가격이 필요한 계산에서는 실제로 다른 시각의 가격과 짝지어지는 문제가 있었다
(`check_price_timezone_impact.py`로 보정 전/후 gap을 비교해 확인함).

---

## 3단계: 보정으로 생긴 앞쪽 결측 처리

가격을 +5시간 밀었기 때문에, 태양광 구간 맨 앞 몇 시간(확장 가격 파일 기준
2012-11-01 00:00~04:00)은 대응하는 과거 가격이 없어진다. **forward-fill로
채우지 않고 명시적으로 그 행들을 제거**했다:

```python
merged = merged.dropna(subset=["da_price", "rt_price"]).reset_index(drop=True)
```

---

## 4단계 (별개의 문제): "낮 시간대"를 어떤 시간대 기준으로 볼 것인가

1~3단계로 **가격과 태양광이 같은 UTC 축 위에서 정확히 merge**되긴 했지만, 그
다음 "하루 중 몇 시가 낮 시간대냐"는 **또 다른 시간대 문제**였다. 논문은
"호주 현지시간 9시~21시"라고 서술하는데, 처음엔 그냥 UTC 0~11시로 근사했다.
나중에 이걸 실제 Sydney 현지시간으로 다시 계산했다:

```python
local_ts = merged["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Australia/Sydney")
merged["local_date"] = local_ts.dt.date
merged["local_hour"] = local_ts.dt.hour
```

이건 merge 자체와는 무관하고, merge가 끝난 뒤 **UTC로 통일된 timestamp를 다시
Sydney 시간으로 해석**해서 어느 시간이 "낮"인지 고르는 별도 단계다. 이 보정을
적용한 뒤 AR/MLR을 다시 돌려보니 nRMSE가 실제로 개선됐다 (`sydney_local_
block_experiment.py` 실험, AR -2~2.7%p, MLR -0.5~3.6%p 개선).

---

## 전체 흐름 요약

```
predictors15.csv (UTC, 원본 그대로)
                                          →  같은 UTC 축에서 merge  →  merged_for_simulation_zXX.csv
da_lmp_prices.csv (EST) + 5시간 = UTC                                   (timestamp=UTC, local_date/local_hour=Sydney)
```

- **"가격↔태양광 정렬"**은 EST→UTC 보정 (2~3단계)
- **"낮 시간대 선택"**은 UTC→Sydney 변환 (4단계)

서로 다른 두 번의 시간대 처리가 들어갔다는 점이 핵심이며, 이 둘을 섞어서
생각하면 혼란스러울 수 있어 따로 구분해서 기록해둔다.

## 관련 파일

| 파일 | 역할 |
|---|---|
| `build_merged_for_simulation.py` | EST→UTC 보정 + Sydney local_date/local_hour 열 생성이 반영된 메인 병합 스크립트 |
| `check_price_timezone_impact.py` | EST→UTC 보정 전/후 optimality gap 비교 |
| `sydney_local_block_experiment.py` | UTC 0~11시 vs Sydney 9~20시 낮 시간대 선택 전/후 AR·MLR 성능 비교 |



###  추가로 진행된 것

 세 가지 검증 결과 모두 완전히 일치합니다.

1. da_lmp_prices_extended.csv ↔ da_lmp_prices_extended_full.csv
항목	결과
extended 구간	2012-11-01 ~ 2014-04-30 (13,104행)
extended_full 구간	2012-04-01 ~ 2014-04-30 (18,240행, 앞으로 더 확장됨)
공통 timestamp	13,104개 (extended 전체가 그대로 포함됨)
DA_LMP 값 불일치	0건
old에만 있는 timestamp	0개
중복 timestamp	0개(둘 다)
2. rt_lmp_prices_extended.csv ↔ rt_lmp_prices_extended_full.csv
항목	결과
extended 구간	2012-10-31 ~ 2014-04-30 (13,128행)
extended_full 구간	2012-04-01 ~ 2014-04-30 (18,240행)
공통 timestamp	13,128개 (전체 포함됨)
RT_LMP 값 불일치	0건
old에만 있는 timestamp	0개
extended_full 내부 시간 갭	DA/RT 둘 다 0건 (완전 연속)
→ 두 확장 파일 모두 기존 _extended 값을 그대로 유지하면서 앞쪽(2012-04-01까지)으로만 넓어졌고, 내부에 결측/중복도 없습니다.

3. merged_for_simulation_z01 - 복사본.csv ↔ merged_for_simulation_z01.csv
항목	결과
복사본(old)	2012-11-01 05:00 ~ 2014-04-30 23:00 (13,099행)
새 파일(new)	2012-11-01 00:00 ~ 2014-04-30 23:00 (13,104행)
공통 timestamp	13,099개 (복사본 전체가 그대로 포함됨)
new에만 있는 timestamp	5개 — 정확히 2012-11-01 00:00~04:00 (아래 설명)
21개 열 전체 값 불일치	0건 (solar_power, da_price, rt_price, 기상 12변수, dssrd/dtsr, local_timestamp/date/hour 전부 포함)
앞쪽 5시간 차이의 이유
예전엔 da/rt_lmp_prices_extended.csv(2012-11-01부터)를 썼는데, EST→UTC로 +5시간 밀면서 맨 앞 5시간(2012-11-01 00:00~04:00)의 가격이 없어서 그 5시간을 드롭했었습니다. 이번엔 _full 버전(2012-04-01부터, 훨씬 이전 구간까지 커버)을 써서 그 5시간의 가격도 확보됐고, 그래서 새 파일이 정확히 그 5시간만큼 더 많습니다 — 이건 오류가 아니라 의도된 개선입니다.

결론: 두 가격 파일 확장과 merged_for_simulation_z01.csv 재생성 모두 기존 데이터를 훼손 없이 그대로 포함하면서 구간만 확장된 것으로 확인됐습니다.


### 최종 final data
변경 내용
경로 기준: BASE = os.path.dirname(os.path.abspath(__file__)) → BASE = os.getcwd()
__file__이 없는 Jupyter/Colab 셀 환경에서도 그대로 동작하도록 (이 .py 파일 자체를 그대로 노트북에 붙여넣어도 됨)
단, 실행 시 현재 작업 디렉터리가 data/ 폴더가 있는 프로젝트 루트여야 함 — docstring에 명시해둠
가격 파일 교체: da/rt_lmp_prices_extended.csv → da/rt_lmp_prices_extended_final.csv (predictors15.csv와 동일 구간까지 확장된 버전)
구간 확장: START_DATE/END_DATE를 predictors15.csv 전체 구간(2012-04-01 ~ 2014-07-01 00:00:00)에 맞춤
실행 결과 (python build_merged_for_simulation.py)
Zone	행 수	기간
z01/z02/z03	19,700	2012-04-01 05:00 ~ 2014-07-01 00:00
EST→UTC 보정으로 맨 앞 4시간만 드롭되고, predictors15.csv의 전체 구간이 이제 거의 다 커버됩니다.