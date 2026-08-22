
# 데이터 정제 작업 요약



## 개요



원본(raw) 데이터를 논문 실험에 사용할 수 있는 폴더 구조로 정제했습니다.



## 작업 날짜



2026-08-22



## 사용한 원본 데이터



### GEFCom2014 Solar (태양광 + 기상)



공식 원본 파일 **하나**에서 **두 가지** dataset을 뽑아냈습니다.



- 원본 파일: `data_raw/gefcom2014/GEFCom2014/GEFCom2014 Data/GEFCom2014-S_V2/Solar/Task 15/predictors15.csv`

- 행 수: 59,112행, 기간: 2012-04-01 ~ 2014-07-01, Zone 1~3

- 컬럼: `ZONEID, VAR78, VAR79, VAR134, VAR157, VAR164, VAR165, VAR166, VAR167, VAR169, VAR175, VAR178, VAR228, POWER`

- `data_raw/solar/gefcom2014-solar.csv`는 위 공식 원본과 같은 데이터입니다. TIMESTAMP만 인덱스에서 열로 펼쳤을 뿐입니다.



| 결과 파일 | `predictors15.csv`에서 뽑은 컬럼 |

|-----------|----------------------------------|

| `solar/solar-energy-generation.csv` | TIMESTAMP + **POWER** |

| `weather/weather_data.csv` | TIMESTAMP + **VAR78~VAR228** (12개) |



GEFCom2014 Solar Track은 3종류의 파일로 나뉩니다:

- `train*.csv`: ZONEID + POWER 만 (발전량만)

- `predictors*.csv`: ZONEID + 기상변수 12개 + POWER (기상 + 발전량) ← **이걸 사용**

- `benchmark*.csv`: ZONEID + 확률분포 (0.01~0.99 퍼센타일)



### MISO 가격



- `data_raw/miso/raw/2014*_da_pr_xls/`: MISO day-ahead 가격 엑셀 파일 (일별, 1~4월)

- `data_raw/miso/raw/2014*_rt_pr_xls/`: MISO real-time 가격 엑셀 파일 (일별, 1~4월)



**파일 구조**: 각 폴더에 1일당 1개 엑셀 파일 들어 있음. ex) `20140101_da_pr.xls`



**엑셀 파일 내부**: 행마다 시간(Hour 01~24) × 지역(허브) 조합에 대한 LMP 가격



| 지역(허브) | 내용 |

|-----------|------|

| MISO System | 전체 계통 평균 ← **이것만 사용** |

| Illinois Hub | 일리노이 지역 |

| Michigan Hub | 미시간 지역 |

| Minnesota Hub | 미네소타 지역 |

| Indiana Hub | 인디애나 지역 |

| Arkansas Hub | 아칸소 지역 |

| Louisiana Hub | 루이지애나 지역 |

| Texas Hub | 텍사스 지역 |



**논문 기준 적합성**:

- 논문은 **어떤 지역(허브)을 썼는지 밝히지 않음** — "data are available on request"라고만 기술

- 논문은 **어떤 기간을 썼는지 밝히지 않음**

- 따라서 현재 `MISO System`(전체 평균)을 선택한 것은 가장 타당한 판단입니다.

- 정확한 수치를 논문 결과와 비교하려면 저자에게 실제 사용한 처리된 데이터를 요청해야 합니다.



**다른 파일이 필요할까?**:

- GEFCom2014 Price 트랙(`GEFCom2014-P_V2/Price/`)은 **사용 불가** — 2011~2013 기간(2014랑 다름), Zonal Price만 있고 real-time 없음

- 현재 MISO day-ahead + real-time 파일이 있는 그대로 쓸 수 있는 최선입니다.



## 생성된 결과물



```text

data/

├── solar/

│   └── solar-energy-generation.csv      (2880행)

├── prices/

│   ├── day_ahead_prices.csv              (2880행)

│   └── real_time_prices.csv              (2856행)

└── weather/

    └── weather_data.csv                  (2880행)

```



## 각 파일 내용



### solar/solar-energy-generation.csv



- 출처: GEFCom2014 Solar Track `predictors15.csv` → POWER 컬럼만 추출

- 기간: 2014-01-01 ~ 2014-04-30 23:00 (1시간 단위, 총 2880시간)

- 선택된 Zone: Zone 1

- 컬럼: TIMESTAMP, solar_power

- solar_power는 이미 0~1 사이 정규화됨 (30MW로 나누지 않음)



### prices/day_ahead_prices.csv



- 출처: MISO Settlement Point Prices

- 기간: 2014-01-01 ~ 2014-04-30 23:00 (1시간 단위, 총 2880시간)

- 선택된 허브: MISO System

- 컬럼: TIMESTAMP, DA_LMP



### prices/real_time_prices.csv



- 출처: MISO Settlement Point Prices

- 기간: 2014-01-01 ~ 2014-04-29 23:00 (1시간 단위, 총 2856시간)

- 선택된 허브: MISO System

- 컬럼: TIMESTAMP, RT_LMP

- **참고**: 4월 30일 하루 누락 (MISO는 다음날 real-time 가격을 공개하므로 마지막 날 데이터 없음)



### weather/weather_data.csv



- 출처: GEFCom2014 Solar Track `predictors15.csv` → VAR78~VAR228 (12개 기상변수) 컬럼만 추출

- 기간: 2014-01-01 ~ 2014-04-30 23:00 (1시간 단위, 총 2880시간)

- 선택된 Zone: Zone 1

- 컬럼: TIMESTAMP + 12개 기상 변수

- 기상 변수: VAR78, VAR79, VAR134, VAR157, VAR164, VAR165, VAR166, VAR167, VAR169, VAR175, VAR178, VAR228

- 기상 변수는 ECMWF 일전 예보로, 자정(UTC) 기준 다음 24시간치 제공

- 기상 변수명은 GEFCom2014에서 VAR78~VAR228로 익명화됨 (논문은 SSRD, TSR, Hour 3개만 최종 선택)



## 정제 과정



1. **MISO 엑셀 파일 구조 확인**: 1월 파일과 2~4월 파일에서 'MISO System' 행 위치가 다름 (1월: row 13, 2~4월: row 14)

2. **동적 행 탐색**: 파일마다 'MISO System' 문자를 찾아 가격 데이터 시작 행을 결정

3. **Market Date 추출**: 각 파일의 row 1에서 날짜를 분리, 파일 이름이 아닌 실제 시장 날짜 사용

4. **GEFCom 데이터 필터링**: ZoneID==1, 2014년 1~4월 기간만 선택



## 검증 결과



- Solar, DA Price, Weather: 2880행 (2014-01-01 ~ 2014-04-30 23:00)

- RT Price: 2856행 (4월 30일 하루 누락)

- 모든 파일 중복 없음

- Solar+DA+Weather 공통 시점: 2880

- 모두 포함 시 공통 시점: 2856



## 사용 도구



- Python + pandas + xlrd

- 전체 스크립트: `process_data.py`

- Solar + Weather 전용 스크립트: `extract_solar_weather.py`

