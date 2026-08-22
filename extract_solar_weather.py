"""

공식 GEFCom2014 Solar `predictors15.csv` 파일 하나에서

solar + weather 두 dataset을 뽑아냄.



출력:

  data/solar/solar-energy-generation.csv   (TIMESTAMP + POWER)

  data/weather/weather_data.csv            (TIMESTAMP + 기상 12개)



Usage:

    python extract_solar_weather.py

"""

import os

import pandas as pd



#BASE = r"D:\marco\AIBM\PaperModel\.claude\skills\paper-agent"
BASE = r"D:\03_JiWon\JiWonProject"

SOURCE = os.path.join(

    BASE,

    "data_raw", "gefcom2014", "GEFCom2014",

    "GEFCom2014 Data", "GEFCom2014-S_V2", "Solar",

    "Task 15", "predictors15.csv",

)

OUT_SOLAR = os.path.join(BASE, "data", "solar", "solar-energy-generation.csv")

OUT_WEATHER = os.path.join(BASE, "data", "weather", "weather_data.csv")



WEATHER_VARS = [

    "VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165",

    "VAR166", "VAR167", "VAR169", "VAR175", "VAR178", "VAR228",

]



START_DATE = "2014-01-01"

END_DATE   = "2014-04-30 23:00:00"





def main():

    df = pd.read_csv(SOURCE, index_col=1, parse_dates=True)

    df.index.name = "TIMESTAMP"



    # Zone 1 만

    df = df[df["ZONEID"] == 1].copy()



    # 2014년 1~4월 만

    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)].copy()

    df = df.reset_index()



    # --- Solar: TIMESTAMP + POWER → solar_power ---

    solar = df[["TIMESTAMP", "POWER"]].rename(columns={"POWER": "solar_power"})

    os.makedirs(os.path.dirname(OUT_SOLAR), exist_ok=True)

    solar.to_csv(OUT_SOLAR, index=False)

    print(f"[Solar]   {len(solar)}행 -> {OUT_SOLAR}")



    # --- Weather: TIMESTAMP + 기상 12개 ---

    weather = df[["TIMESTAMP"] + WEATHER_VARS]

    os.makedirs(os.path.dirname(OUT_WEATHER), exist_ok=True)

    weather.to_csv(OUT_WEATHER, index=False)

    print(f"[Weather] {len(weather)}행, {len(WEATHER_VARS)}변수 -> {OUT_WEATHER}")





if __name__ == "__main__":

    main()

