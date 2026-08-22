
"""

데이터 4개를 하나로 합쳐 시뮬레이션에 쓸 수 있게 함.



출력: data/merged_for_simulation.csv



GEFCom2014 기상변수 익명명 → 실제 ECMWF 명칭 매핑:

  VAR78  → tclw  (Total Column Water)

  VAR79  → tciw  (Total Column Ice Water)

  VAR134 → sp    (Surface Pressure)

  VAR157 → r     (Relative Humidity)

  VAR164 → tcc   (Total Cloud Cover)

  VAR165 → u10   (10m U Wind)

  VAR166 → v10   (10m V Wind)

  VAR167 → t2m   (2m Temperature)

  VAR169 → sssrd (Surface Solar Radiation Downwards)

  VAR175 → strd  (Surface Thermal Radiation Downwards)

  VAR178 → tsr   (Thermodynamic Surface Radiation)

  VAR228 → tp    (Total Precipitation)

"""

import pandas as pd

import os



#BASE = r"D:\marco\AIBM\PaperModel\.claude\skills\paper-agent\data"

BASE = r"D:\03_JiWon\JiWonProject\data"


# 1. 파일 읽기

solar  = pd.read_csv(os.path.join(BASE, "solar", "solar-energy-generation.csv"))

#da     = pd.read_csv(os.path.join(BASE, "prices", "day_ahead_prices.csv"))
da     = pd.read_csv(os.path.join(BASE, "prices", "da_lmp_prices.csv"))

#rt     = pd.read_csv(os.path.join(BASE, "prices", "real_time_prices.csv"))
rt     = pd.read_csv(os.path.join(BASE, "prices", "rt_lmp_prices.csv"))

weath  = pd.read_csv(os.path.join(BASE, "weather", "weather_data.csv"))



# 2. 컬럼명 정리

solar.columns = ['timestamp', 'solar_power']

da.columns    = ['timestamp', 'da_price']

rt.columns    = ['timestamp', 'rt_price']



# weather 컬럼: TIMESTAMP → timestamp, VAR → 실제 명칭

weath = weath.rename(columns={'TIMESTAMP': 'timestamp'})

var_map = {

    'VAR78':  'tclw', 'VAR79':  'tciw', 'VAR134': 'sp',

    'VAR157': 'r',    'VAR164': 'tcc',  'VAR165': 'u10',

    'VAR166': 'v10',  'VAR167': 't2m',  'VAR169': 'ssrd',

    'VAR175': 'strd', 'VAR178': 'tsr',  'VAR228': 'tp',

}

weath = weath.rename(columns=var_map)



# 3. 병합

merged = pd.merge(solar, da, on='timestamp', how='left')

merged = pd.merge(merged, rt, on='timestamp', how='left')

merged = pd.merge(merged, weath, on='timestamp', how='left')



# 4. RT price 마지막 날 누락 → forward fill

merged['rt_price'] = merged['rt_price'].ffill()



# 5. 정렬

merged = merged.sort_values('timestamp').reset_index(drop=True)



# 6. 저장

outpath = os.path.join(BASE, "merged_for_simulation.csv")

merged.to_csv(outpath, index=False)

print(f"통합 완료: {outpath}")

print(f"행: {len(merged)}, 기간: {merged['timestamp'].iloc[0]} ~ {merged['timestamp'].iloc[-1]}")

print("컬럼:", list(merged.columns))

print(merged.head(3))