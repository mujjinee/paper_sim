import os
import pandas as pd

def fetch_and_preprocess_weather():
    # GEFCom2014 Solar 기상 예보 정제 데이터셋 (GitHub Open Data)
    url = "https://raw.githubusercontent.com/akylasstrat/df-forecast-comb/main/data/weather_gefcom2014.csv"
    
    print("Downloading GEFCom2014 Weather dataset (ECMWF)...")
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"1차 URL 로드 실패 ({e}), 미러 URL 접속을 시도합니다.")
        alt_url = "https://raw.githubusercontent.com/greenlytics/mqe-forecast/master/data/gefcom2014-weather-raw.csv"
        df = pd.read_csv(alt_url)

    # 타임스탬프 파싱
    time_col = next((c for c in df.columns if 'TIME' in c.upper() or 'DATE' in c.upper()), df.columns[0])
    df['TIMESTAMP'] = pd.to_datetime(df[time_col])

    # 2014년 1월 1일 ~ 4월 30일 타임라인 필터링
    start_date = '2014-01-01 00:00:00'
    end_date = '2014-04-30 23:00:00'
    filtered_df = df[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date)].copy()

    # 정렬 및 저장
    filtered_df = filtered_df.sort_values('TIMESTAMP').reset_index(drop=True)
    
    output_dir = "weather"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "weather_data.csv")
    
    filtered_df.to_csv(output_path, index=False)
    
    print(f"\n 기상 데이터 저장 완료: {output_path}")
    print(f"- 레코드 수: {len(filtered_df):,} 개 (시간)")
    print(f"- 기상 변수 개수: {len(filtered_df.columns) - 1} 개")
    print(filtered_df.head())

if __name__ == "__main__":
    fetch_and_preprocess_weather()