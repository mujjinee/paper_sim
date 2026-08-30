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
