
"""

논문(Karimi & Kwon, 2022) 실험 재현 - MISO 가격 없이 GEFCom2014 전체 기간 사용.



목적:

  1. 학습 데이터 부족 문제(90일 vs 300일)가 얼마나 영향을 미치는지 확인

  2. nRMSE 계산 방식 3가지 비교 (전체 합산 / 하루별 평균 / 시간대별 평균)

  3. MLR backward stepwise (ssrd, tsr, hour) 효과 확인



데이터:

  - GEFCom2014 Solar Track predictors15.csv 전체 기간(2012.04~2014.07) 사용

  - MISO 가격 없음 → nRMSE 만 평가 (optimality gap 불가)

  - Zone 1, UTC 00:00~11:00 12시간만 사용



학습/테스트:

  - 학습: 300일, 테스트: 100일

  - 2012.04~2014.07 전체 400일 이상 사용 가능



모델:

  1. AR: direct multi-step, 과거 12일 같은 시간

  2. MLR(12vars): 기상 변수 12개 전 변수

  3. MLR(sw-3vars): ssrd, tsr, hour 3개 (논문 backward stepwise)

"""

import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LinearRegression



BASE = r"D:\marco\AIBM\PaperModel\.claude\skills\paper-agent"

# GEFCom2014 원본 파일 직접 읽기 (MISO 가격 제외)

SOURCE = os.path.join(

    BASE,

    "data_raw", "gefcom2014", "GEFCom2014",

    "GEFCom2014 Data", "GEFCom2014-S_V2", "Solar",

    "Task 15", "predictors15.csv",

)

OUTPUT_DIR = os.path.join(BASE, "results", "simulation_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)



N_HOURS = 12  # 하루 12시간 (UTC 00:00~11:00)

TRAIN_DAYS = 300

TEST_DAYS = 100





# ============================================================

# 1. 데이터 로딩

# ============================================================

def load_gefcom_full():

    """predictors15.csv 전체 기간을 읽어 Zone 1, UTC 00:00~11:00 만 추출.



    반환:

        DataFrame (TIMESTAMP, solar_power, 기상 변수 12개, date, hour_idx)

    """

    df = pd.read_csv(SOURCE, index_col=1, parse_dates=True)

    df.index.name = "TIMESTAMP"



    # Zone 1 만

    df = df[df["ZONEID"] == 1].copy()

    df = df.reset_index()



    # UTC 00:00~11:00 만 (하루 12시간)

    df = df[df['TIMESTAMP'].dt.hour.between(0, 11)].reset_index(drop=True)



    df['date'] = df['TIMESTAMP'].dt.date

    df['hour_idx'] = df['TIMESTAMP'].dt.hour



    # POWER → solar_power

    df = df.rename(columns={"POWER": "solar_power"})

    df = df.sort_values('TIMESTAMP').reset_index(drop=True)



    print(f"전체 데이터: {len(df)}행 ({len(df)//N_HOURS}일)")

    print(f"기간: {df['TIMESTAMP'].min()} ~ {df['TIMESTAMP'].max()}")

    return df





def to_daily_arrays(df):

    """DataFrame → (n_days, 12) 배열 변환."""

    dates = sorted(df['date'].unique())

    n_days = len(dates)

    solar_arr = np.zeros((n_days, N_HOURS))

    for i, date in enumerate(dates):

        day_df = df[df['date'] == date]

        for _, row in day_df.iterrows():

            h = int(row['hour_idx'])

            if 0 <= h < N_HOURS:

                solar_arr[i, h] = row['solar_power']

    return solar_arr





# ============================================================

# 2. nRMSE 계산 방식 3가지

# ============================================================

def nrmse_overall(actual, predicted):

    """전체 테스트 기간을 하나로 합쳐 nRMSE 계산."""

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return 100 * rmse / np.mean(actual) if np.mean(actual) > 0 else 0





def nrmse_per_day_average(daily_actual, daily_predicted):

    """하루 12시간에 대해 각 날마다 nRMSE 계산 후 전체 평균.



    nRMSE_day = 100 × RMSE(day) / 평균 발전량(day)

    결과 = mean(nRMSE_day)

    """

    nrmse_list = []

    for day in range(daily_actual.shape[0]):

        day_nrmse = nrmse_overall(daily_actual[day], daily_predicted[day])

        nrmse_list.append(day_nrmse)

    return np.mean(nrmse_list)





def nrmse_per_hour_average(daily_actual, daily_predicted):

    """시간대(h=0~11)별로 nRMSE 계산 후 평균.



    논문 Eq.(11)에 nRMSE_h 표기가 있으므로 이 방식이 가장 가까울 가능성 있음.

    nRMSE_h = 100 × RMSE(h) / 평균 발전량(h)

    결과 = mean(nRMSE_h)

    """

    nrmse_list = []

    for h in range(daily_actual.shape[1]):

        h_actual = daily_actual[:, h]  # 테스트 기간 전체의 시간 h 데이터

        h_pred = daily_predicted[:, h]

        h_nrmse = nrmse_overall(h_actual, h_pred)

        nrmse_list.append(h_nrmse)

    return np.mean(nrmse_list)





# ============================================================

# 3. AR 모델 (direct multi-step)

# ============================================================

class ARModel:

    def __init__(self, n_lags=12):

        self.n_lags = n_lags

        self.models = {}



    def fit(self, daily_solar):

        n_days = daily_solar.shape[0]

        for h in range(daily_solar.shape[1]):

            X, y = [], []

            for day in range(self.n_lags, n_days):

                features = [daily_solar[day - i - 1, h] for i in range(self.n_lags)]

                X.append(features)

                y.append(daily_solar[day, h])

            X, y = np.array(X), np.array(y)

            if len(X) > 5:

                model = LinearRegression()

                model.fit(X, y)

                self.models[h] = model



    def predict_next_day(self, past_daily):

        n_hours = past_daily.shape[1]

        forecast = np.zeros(n_hours)

        for h in range(n_hours):

            if h in self.models:

                features = [past_daily[self.n_lags - i - 1, h] for i in range(self.n_lags)]

                pred = self.models[h].predict([features])[0]

            else:

                pred = np.mean(past_daily[:, h])

            forecast[h] = np.clip(pred, 0, 1)

        return forecast





# ============================================================

# 4. MLR 모델

# ============================================================

class MLRModel:

    def __init__(self, weather_vars, use_stepwise=False):

        self.weather_vars = weather_vars

        self.use_stepwise = use_stepwise

        self.models = {}



    def _get_vars(self, hour_idx):

        if self.use_stepwise:

            return ['ssrd', 'tsr', 'hour']

        return self.weather_vars



    def fit(self, daily_solar, train_df, n_hours):

        dates = sorted(train_df['date'].unique())

        for h in range(n_hours):

            X, y = [], []

            for day_idx in range(len(dates)):

                date = dates[day_idx]

                row = train_df[(train_df['date'] == date) & (train_df['hour_idx'] == h)]

                if len(row):

                    vars = self._get_vars(h)

                    feature_row = []

                    for var in vars:

                        if var == 'hour':

                            feature_row.append(h)

                        else:

                            feature_row.append(row.iloc[0][var])

                    X.append(feature_row)

                    y.append(daily_solar[day_idx, h])

            X, y = np.array(X), np.array(y)

            if len(X) > 5:

                model = LinearRegression()

                model.fit(X, y)

                self.models[h] = model



    def predict_next_day(self, test_day_df, n_hours):

        forecast = np.zeros(n_hours)

        for h in range(n_hours):

            if h in self.models:

                row = test_day_df[test_day_df['hour_idx'] == h]

                if len(row):

                    vars = self._get_vars(h)

                    feature_row = []

                    for var in vars:

                        if var == 'hour':

                            feature_row.append(h)

                        else:

                            feature_row.append(row.iloc[0][var])

                    pred = self.models[h].predict(np.array(feature_row).reshape(1, -1))[0]

                else:

                    pred = 0.5

            else:

                pred = 0.5

            forecast[h] = np.clip(pred, 0, 1)

        return forecast





# ============================================================

# 5. 실행

# ============================================================

def run_experiment():

    print("GEFCom2014 전체 데이터 로딩 (MISO 없이 solar+weather 만)...")

    df = load_gefcom_full()



    # 기상 변수명 매핑 (VAR78 → tclw 등)

    VAR_MAP = {

        'VAR78': 'tclw', 'VAR79': 'tciw', 'VAR134': 'sp',

        'VAR157': 'r', 'VAR164': 'tcc', 'VAR165': 'u10',

        'VAR166': 'v10', 'VAR167': 't2m', 'VAR169': 'ssrd',

        'VAR175': 'strd', 'VAR178': 'tsr', 'VAR228': 'tp',

    }

    df = df.rename(columns=VAR_MAP)

    weather_vars = ['tclw', 'tciw', 'sp', 'r', 'tcc', 'u10', 'v10', 't2m', 'ssrd', 'strd', 'tsr', 'tp']



    # 학습/테스트 분할 (300일 학습, 100일 테스트)

    unique_dates = sorted(df['date'].unique())

    train_dates = set(unique_dates[:TRAIN_DAYS])

    test_dates = set(unique_dates[TRAIN_DAYS:TRAIN_DAYS + TEST_DAYS])



    train = df[df['date'].isin(train_dates)].copy()

    test = df[df['date'].isin(test_dates)].copy()



    n_train_days = len(train_dates)

    n_test_days = len(test_dates)

    print(f"학습: {n_train_days}일, 테스트: {n_test_days}일")

    print(f"테스트 기간: {test['TIMESTAMP'].min()} ~ {test['TIMESTAMP'].max()}")



    train_solar = to_daily_arrays(train)

    test_solar = to_daily_arrays(test)



    # --- AR ---

    print("\n=== AR 모델 학습 (300일) ===")

    ar_model = ARModel(n_lags=12)

    ar_model.fit(train_solar)



    print("\n=== AR 예측 실행 ===")

    ar_forecast = np.zeros((n_test_days, N_HOURS))

    for day in range(n_test_days):

        past = np.vstack([train_solar[-12:], ar_forecast[:day]])

        ar_forecast[day] = ar_model.predict_next_day(past)



    print(f"  AR nRMSE (전체):   {nrmse_overall(test_solar.flatten(), ar_forecast.flatten()):.2f}%")

    print(f"  AR nRMSE (day-avg): {nrmse_per_day_average(test_solar, ar_forecast):.2f}%")

    print(f"  AR nRMSE (hour-avg): {nrmse_per_hour_average(test_solar, ar_forecast):.2f}%")



    # --- MLR(12vars) ---

    print("\n=== MLR 모델 학습 (12개 변수, 300일) ===")

    mlr_model = MLRModel(weather_vars, use_stepwise=False)

    mlr_model.fit(train_solar, train, N_HOURS)



    mlr_forecast = np.zeros((n_test_days, N_HOURS))

    for day in range(n_test_days):

        date = sorted(test_dates)[day]

        test_day_df = test[test['date'] == date]

        mlr_forecast[day] = mlr_model.predict_next_day(test_day_df, N_HOURS)



    print(f"  MLR(12vars) nRMSE (전체):   {nrmse_overall(test_solar.flatten(), mlr_forecast.flatten()):.2f}%")

    print(f"  MLR(12vars) nRMSE (day-avg): {nrmse_per_day_average(test_solar, mlr_forecast):.2f}%")

    print(f"  MLR(12vars) nRMSE (hour-avg): {nrmse_per_hour_average(test_solar, mlr_forecast):.2f}%")



    # --- MLR(sw-3vars) ---

    print("\n=== MLR 모델 학습 (ssrd, tsr, hour, 300일) ===")

    mlr_sw = MLRModel(weather_vars, use_stepwise=True)

    mlr_sw.fit(train_solar, train, N_HOURS)



    mlr_sw_forecast = np.zeros((n_test_days, N_HOURS))

    for day in range(n_test_days):

        date = sorted(test_dates)[day]

        test_day_df = test[test['date'] == date]

        mlr_sw_forecast[day] = mlr_sw.predict_next_day(test_day_df, N_HOURS)



    print(f"  MLR(sw) nRMSE (전체):   {nrmse_overall(test_solar.flatten(), mlr_sw_forecast.flatten()):.2f}%")

    print(f"  MLR(sw) nRMSE (day-avg): {nrmse_per_day_average(test_solar, mlr_sw_forecast):.2f}%")

    print(f"  MLR(sw) nRMSE (hour-avg): {nrmse_per_hour_average(test_solar, mlr_sw_forecast):.2f}%")



    # --- 저장 ---

    result_rows = [

        {'model': 'AR',

         'nRMSE_overall': nrmse_overall(test_solar.flatten(), ar_forecast.flatten()),

         'nRMSE_day_avg': nrmse_per_day_average(test_solar, ar_forecast),

         'nRMSE_hour_avg': nrmse_per_hour_average(test_solar, ar_forecast),

         'train_days': n_train_days, 'test_days': n_test_days},

        {'model': 'MLR(12vars)',

         'nRMSE_overall': nrmse_overall(test_solar.flatten(), mlr_forecast.flatten()),

         'nRMSE_day_avg': nrmse_per_day_average(test_solar, mlr_forecast),

         'nRMSE_hour_avg': nrmse_per_hour_average(test_solar, mlr_forecast),

         'train_days': n_train_days, 'test_days': n_test_days},

        {'model': 'MLR(sw-3vars)',

         'nRMSE_overall': nrmse_overall(test_solar.flatten(), mlr_sw_forecast.flatten()),

         'nRMSE_day_avg': nrmse_per_day_average(test_solar, mlr_sw_forecast),

         'nRMSE_hour_avg': nrmse_per_hour_average(test_solar, mlr_sw_forecast),

         'train_days': n_train_days, 'test_days': n_test_days},

    ]



    outpath = os.path.join(OUTPUT_DIR, 'experiment_no_miso_300days.csv')

    pd.DataFrame(result_rows).to_csv(outpath, index=False)

    print(f"\n=== 결과 저장: {outpath} ===")



    # --- 논문과 비교 ---

    print(f"\n=== 논문 vs 실행 (penalty=50%) ===")

    print(f"{'Model':<16} {'논문':>8} {'전체':>8} {'day-avg':>10} {'hour-avg':>10}")

    print(f"{'AR':<16} {'34.76%':>8} {result_rows[0]['nRMSE_overall']:>7.2f}% {result_rows[0]['nRMSE_day_avg']:>9.2f}% {result_rows[0]['nRMSE_hour_avg']:>9.2f}%")

    print(f"{'MLR(12vars)':<16} {'21.76%':>8} {result_rows[1]['nRMSE_overall']:>7.2f}% {result_rows[1]['nRMSE_day_avg']:>9.2f}% {result_rows[1]['nRMSE_hour_avg']:>9.2f}%")

    print(f"{'MLR(sw)':<16} {'21.76%':>8} {result_rows[2]['nRMSE_overall']:>7.2f}% {result_rows[2]['nRMSE_day_avg']:>9.2f}% {result_rows[2]['nRMSE_hour_avg']:>9.2f}%")





if __name__ == '__main__':

    run_experiment()
    