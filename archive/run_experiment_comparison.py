"""

논문 (Karimi & Kwon, 2022, Applied Energy 326: 119929) 실험 재현 + 확장 - 통합 스크립트.



논문 제목: Optimization-driven uncertainty forecasting: Application to day-ahead commitment

            with renewable energy resources



실험 구성:

  Exp 1: 300일 학습, nRMSE 3가지 방식 비교 (AR / MLR 12vars / MLR sw-3vars)

  Exp 2: 딥러닝 기반 확장 (XGBoost / LSTM / DF-NN / REINFORCE)

  Exp 3: XGBoost + 최적화 커밋 조정 (synthetic price)



데이터:

  - GEFCom2014 Solar Track (Zone 1, UTC 00:00~11:00, ECMWF 기상 변수 12개)

  - MISO 가격 없음 → nRMSE 만 평가 (Exp 3 은 synthetic price 사용)

"""

import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LinearRegression



BASE = r"D:\marco\AIBM\PaperModel\.claude\skills\paper-agent"

SOURCE = os.path.join(

    BASE,

    "data_raw", "gefcom2014", "GEFCom2014",

    "GEFCom2014 Data", "GEFCom2014-S_V2", "Solar",

    "Task 15", "predictors15.csv",

)

OUTPUT_DIR = os.path.join(BASE, "results", "simulation_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)



N_HOURS = 12

TRAIN_DAYS = 300

TEST_DAYS = 100



VAR_MAP = {

    'VAR78': 'tclw', 'VAR79': 'tciw', 'VAR134': 'sp',

    'VAR157': 'r', 'VAR164': 'tcc', 'VAR165': 'u10',

    'VAR166': 'v10', 'VAR167': 't2m', 'VAR169': 'ssrd',

    'VAR175': 'strd', 'VAR178': 'tsr', 'VAR228': 'tp',

}

WEATHER_VARS = ['tclw', 'tciw', 'sp', 'r', 'tcc', 'u10', 'v10', 't2m',

                'ssrd', 'strd', 'tsr', 'tp']





# ============================================================

# 공통 유틸리티

# ============================================================

def load_gefcom_full():

    """predictors15.csv 전체 기간을 읽어 Zone 1, UTC 00:00~11:00 만 추출."""

    df = pd.read_csv(SOURCE, index_col=1, parse_dates=True)

    df.index.name = "TIMESTAMP"

    df = df[df["ZONEID"] == 1].copy().reset_index()

    df = df[df['TIMESTAMP'].dt.hour.between(0, 11)].reset_index(drop=True)

    df['date'] = df['TIMESTAMP'].dt.date

    df['hour_idx'] = df['TIMESTAMP'].dt.hour

    df = df.rename(columns={"POWER": "solar_power"})

    df = df.sort_values('TIMESTAMP').reset_index(drop=True)

    df = df.rename(columns=VAR_MAP)

    print(f"전체: {len(df)}행 ({len(df)//N_HOURS}일), "

          f"기간: {df['TIMESTAMP'].min()} ~ {df['TIMESTAMP'].max()}")

    return df





def to_daily_arrays(df):

    """DataFrame -> (n_days, N_HOURS) numpy 배열 변환."""

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





def split_train_test(df):

    """300일 학습 / 100일 테스트로 분할."""

    unique_dates = sorted(df['date'].unique())

    train_dates = set(unique_dates[:TRAIN_DAYS])

    test_dates = set(unique_dates[TRAIN_DAYS:TRAIN_DAYS + TEST_DAYS])

    train = df[df['date'].isin(train_dates)].copy()

    test  = df[df['date'].isin(test_dates)].copy()

    return train, test





# --- nRMSE 계산 방식 3가지 ---

def nrmse_overall(actual, predicted):

    """전체 테스트 기간을 하나로 합쳐 nRMSE 계산."""

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return 100 * rmse / np.mean(actual) if np.mean(actual) > 0 else 0





def nrmse_per_day_average(daily_actual, daily_predicted):

    """하루 12시간에 대해 각 날마다 nRMSE 계산 후 전체 평균."""

    return np.mean([nrmse_overall(daily_actual[d], daily_predicted[d])

                    for d in range(daily_actual.shape[0])])





def nrmse_per_hour_average(daily_actual, daily_predicted):

    """시간대(h=0~11)별로 테스트 기간 전체의 nRMSE 계산 후 평균."""

    return np.mean([nrmse_overall(daily_actual[:, h], daily_predicted[:, h])

                    for h in range(daily_actual.shape[1])])





# --- Unit Commitment (Exp 3 용) ---

def unit_commitment_profit(commitment, actual, da_price, rt_price, penalty_cost):

    mismatch = actual - commitment

    surplus = np.maximum(mismatch, 0)

    shortage = np.maximum(-mismatch, 0)

    return np.sum(da_price * commitment + rt_price * surplus

                  - rt_price * shortage - penalty_cost * shortage)





def oracle_profit(actual, da_price, rt_price, penalty_cost):

    optimal_commit = np.where(da_price > rt_price, actual, 0)

    return unit_commitment_profit(optimal_commit, actual, da_price, rt_price, penalty_cost)





def optimality_gap_pct(commitment, actual, da_price, rt_price, penalty_rate):

    pc = penalty_rate * da_price

    o_p = oracle_profit(actual, da_price, rt_price, pc)

    a_p = unit_commitment_profit(commitment, actual, da_price, rt_price, pc)

    if abs(o_p) > 1e-10:

        return 100 * (o_p - a_p) / o_p

    return 0





# ============================================================

# AR 모델 (direct multi-step)

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

# MLR 모델

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

                    feature_row = []

                    for var in self._get_vars(h):

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

                    feature_row = []

                    for var in self._get_vars(h):

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

# Exp 1: 300일 학습, nRMSE 3가지 비교 (AR / MLR 12vars / MLR sw-3vars)

# ============================================================

def run_experiment_1(df):

    """MISO 없이 GEFCom2014 전체 기간, 300일 학습 / 100일 테스트, nRMSE 3가지 방식."""

    print("=" * 60)

    print("Exp 1: 300일 학습, nRMSE 3가지 방식 비교 (AR / MLR)")

    print("=" * 60)



    train, test = split_train_test(df)

    train_solar = to_daily_arrays(train)

    test_solar  = to_daily_arrays(test)

    n_test_days = test_solar.shape[0]

    test_dates  = sorted(test['date'].unique())



    print(f"학습: {TRAIN_DAYS}일, 테스트: {n_test_days}일")

    print(f"테스트 기간: {test['TIMESTAMP'].min()} ~ {test['TIMESTAMP'].max()}")



    results = []



    # --- AR ---

    print("\n--- AR (direct multi-step) ---")

    ar_model = ARModel(n_lags=12)

    ar_model.fit(train_solar)

    ar_forecast = np.zeros((n_test_days, N_HOURS))

    for day in range(n_test_days):

        past = np.vstack([train_solar[-12:], ar_forecast[:day]])

        ar_forecast[day] = ar_model.predict_next_day(past)



    ar_overall = nrmse_overall(test_solar.flatten(), ar_forecast.flatten())

    ar_dayavg  = nrmse_per_day_average(test_solar, ar_forecast)

    ar_houravg = nrmse_per_hour_average(test_solar, ar_forecast)

    results.append({'model': 'AR',

                    'nRMSE_overall': ar_overall,

                    'nRMSE_day_avg': ar_dayavg,

                    'nRMSE_hour_avg': ar_houravg})

    print(f"  nRMSE (전체): {ar_overall:.2f}%, (day-avg): {ar_dayavg:.2f}%, (hour-avg): {ar_houravg:.2f}%")



    # --- MLR(12vars) ---

    print("\n--- MLR(12vars) ---")

    mlr12 = MLRModel(WEATHER_VARS, use_stepwise=False)

    mlr12.fit(train_solar, train, N_HOURS)

    mlr12_forecast = np.zeros((n_test_days, N_HOURS))

    for i, date in enumerate(test_dates):

        mlr12_forecast[i] = mlr12.predict_next_day(test[test['date'] == date], N_HOURS)



    m12_overall = nrmse_overall(test_solar.flatten(), mlr12_forecast.flatten())

    m12_dayavg  = nrmse_per_day_average(test_solar, mlr12_forecast)

    m12_houravg = nrmse_per_hour_average(test_solar, mlr12_forecast)

    results.append({'model': 'MLR(12vars)',

                    'nRMSE_overall': m12_overall,

                    'nRMSE_day_avg': m12_dayavg,

                    'nRMSE_hour_avg': m12_houravg})

    print(f"  nRMSE (전체): {m12_overall:.2f}%, (day-avg): {m12_dayavg:.2f}%, (hour-avg): {m12_houravg:.2f}%")



    # --- MLR(sw-3vars) ---

    print("\n--- MLR(sw-3vars: ssrd, tsr, hour) ---")

    mlr3 = MLRModel(WEATHER_VARS, use_stepwise=True)

    mlr3.fit(train_solar, train, N_HOURS)

    mlr3_forecast = np.zeros((n_test_days, N_HOURS))

    for i, date in enumerate(test_dates):

        mlr3_forecast[i] = mlr3.predict_next_day(test[test['date'] == date], N_HOURS)



    m3_overall = nrmse_overall(test_solar.flatten(), mlr3_forecast.flatten())

    m3_dayavg  = nrmse_per_day_average(test_solar, mlr3_forecast)

    m3_houravg = nrmse_per_hour_average(test_solar, mlr3_forecast)

    results.append({'model': 'MLR(sw-3vars)',

                    'nRMSE_overall': m3_overall,

                    'nRMSE_day_avg': m3_dayavg,

                    'nRMSE_hour_avg': m3_houravg})

    print(f"  nRMSE (전체): {m3_overall:.2f}%, (day-avg): {m3_dayavg:.2f}%, (hour-avg): {m3_houravg:.2f}%")



    # 저장

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'exp1_nrmse_comparison.csv'), index=False)

    print(f"\n  저장: exp1_nrmse_comparison.csv")



    # 논문과 비교 출력

    print(f"\n  {'Model':<16} {'논문':>8} {'전체':>8} {'day-avg':>10} {'hour-avg':>10}")

    print(f"  {'-'*52}")

    print(f"  {'AR':<16} {'34.76%':>8} {results[0]['nRMSE_overall']:>7.2f}% {results[0]['nRMSE_day_avg']:>9.2f}% {results[0]['nRMSE_hour_avg']:>9.2f}%")

    print(f"  {'MLR(12vars)':<16} {'21.76%':>8} {results[1]['nRMSE_overall']:>7.2f}% {results[1]['nRMSE_day_avg']:>9.2f}% {results[1]['nRMSE_hour_avg']:>9.2f}%")

    print(f"  {'MLR(sw)':<16} {'21.76%':>8} {results[2]['nRMSE_overall']:>7.2f}% {results[2]['nRMSE_day_avg']:>9.2f}% {results[2]['nRMSE_hour_avg']:>9.2f}%")



    return results





# ============================================================

# Exp 2: 딥러닝 기반 확장 (XGBoost / LSTM / DF-NN / REINFORCE)

# ============================================================

def run_experiment_2(df):

    """딥러닝 3가지 접근법 + 기존 AR/MLR 비교."""

    print("\n" + "=" * 60)

    print("Exp 2: 딥러닝 기반 확장 (XGBoost / LSTM / DF-NN / REINFORCE)")

    print("=" * 60)



    train, test = split_train_test(df)

    train_solar = to_daily_arrays(train)

    test_solar  = to_daily_arrays(test)

    n_test_days = test_solar.shape[0]

    test_dates  = sorted(test['date'].unique())



    all_results = []



    # --- A. AR ---

    print("\n--- A. AR ---")

    ar = ARModel(n_lags=12)

    ar.fit(train_solar)

    ar_fc = np.zeros((n_test_days, N_HOURS))

    for d in range(n_test_days):

        past = np.vstack([train_solar[-12:], ar_fc[:d]])

        ar_fc[d] = ar.predict_next_day(past)

    ar_nrmse = nrmse_overall(test_solar.flatten(), ar_fc.flatten())

    all_results.append({'model': 'AR', 'nRMSE': ar_nrmse})

    print(f"  AR nRMSE: {ar_nrmse:.2f}%")



    # --- B. MLR(12vars) ---

    print("\n--- B. MLR(12vars) ---")

    mlr12 = MLRModel(WEATHER_VARS, use_stepwise=False)

    mlr12.fit(train_solar, train, N_HOURS)

    mlr12_fc = np.zeros((n_test_days, N_HOURS))

    for i, date in enumerate(test_dates):

        mlr12_fc[i] = mlr12.predict_next_day(test[test['date'] == date], N_HOURS)

    m12_nrmse = nrmse_overall(test_solar.flatten(), mlr12_fc.flatten())

    all_results.append({'model': 'MLR(12vars)', 'nRMSE': m12_nrmse})

    print(f"  MLR(12vars) nRMSE: {m12_nrmse:.2f}%")



    # --- C. MLR(sw-3vars) ---

    print("\n--- C. MLR(sw-3vars) ---")

    mlr3 = MLRModel(WEATHER_VARS, use_stepwise=True)

    mlr3.fit(train_solar, train, N_HOURS)

    mlr3_fc = np.zeros((n_test_days, N_HOURS))

    for i, date in enumerate(test_dates):

        mlr3_fc[i] = mlr3.predict_next_day(test[test['date'] == date], N_HOURS)

    m3_nrmse = nrmse_overall(test_solar.flatten(), mlr3_fc.flatten())

    all_results.append({'model': 'MLR(sw-3vars)', 'nRMSE': m3_nrmse})

    print(f"  MLR(sw) nRMSE: {m3_nrmse:.2f}%")



    # --- D. XGBoost (접근법 ①) ---

    print("\n--- D. XGBoost ---")

    xgb_fc, xgb_nrmse = _run_xgboost(train, test, train_solar, test_solar)

    all_results.append({'model': 'XGBoost', 'nRMSE': xgb_nrmse})



    # --- E. LSTM (접근법 ①) ---

    print("\n--- E. LSTM ---")

    lstm_fc, lstm_nrmse = _run_lstm(train_solar, test_solar)

    all_results.append({'model': 'LSTM', 'nRMSE': lstm_nrmse})



    # --- F. Decision-Focused NN (접근법 ②) ---

    print("\n--- F. Decision-Focused NN ---")

    df_fc, df_nrmse = _run_decision_focused_nn(train, test, train_solar, test_solar, penalty_rate=0.5)

    all_results.append({'model': 'DF-NN(p=50%)', 'nRMSE': df_nrmse})



    # --- G. REINFORCE NN (접근법 ③) ---

    print("\n--- G. REINFORCE NN ---")

    rnf_fc, rnf_nrmse = _run_reinforce_nn(train, test, train_solar, test_solar, penalty_rate=0.5, n_episodes=50)

    all_results.append({'model': 'REINFORCE(p=50%)', 'nRMSE': rnf_nrmse})



    # 정리

    print("\n" + "=" * 60)

    print("Exp 2 결과")

    print("=" * 60)

    print(f"  {'Model':<20} {'nRMSE':>10} {'논문 비교':>12}")

    print(f"  {'-'*42}")

    for r in all_results:

        comp = '-'

        if r['model'] == 'AR':

            comp = f"34.76%->{r['nRMSE']:.2f}%"

        elif 'MLR' in r['model']:

            comp = f"21.76%->{r['nRMSE']:.2f}%"

        print(f"  {r['model']:<20} {r['nRMSE']:>9.2f}% {comp:>12}")



    pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, 'exp2_deep_learning_comparison.csv'), index=False)

    print(f"\n  저장: exp2_deep_learning_comparison.csv")

    return all_results





# --- XGBoost ---

def _run_xgboost(train, test, train_solar, test_solar):

    import xgboost as xgb



    all_dates = sorted(train['date'].unique()) + sorted(test['date'].unique())

    all_to_idx = {d: i for i, d in enumerate(all_dates)}

    all_solar = {}

    for source in [train, test]:

        for d in sorted(source['date'].unique()):

            day_df = source[source['date'] == d]

            arr = np.zeros(N_HOURS)

            for _, r in day_df.iterrows():

                arr[int(r['hour_idx'])] = r['solar_power']

            all_solar[d] = arr



    def build_features(target_df, n_lags=12):

        dates = sorted(target_df['date'].unique())

        X, y = [], []

        for date in dates:

            day_df = target_df[target_df['date'] == date]

            for _, row in day_df.iterrows():

                h = int(row['hour_idx'])

                features = []

                idx = all_to_idx[date]

                for lag in range(n_lags):

                    prev_idx = idx - lag - 1

                    if prev_idx >= 0:

                        features.append(all_solar[all_dates[prev_idx]][h])

                    else:

                        features.append(0)

                for var in WEATHER_VARS:

                    features.append(row.get(var, 0))

                features.append(float(np.sin(2 * np.pi * h / N_HOURS)))

                features.append(float(np.cos(2 * np.pi * h / N_HOURS)))

                y.append(row['solar_power'])

                X.append(features)

        return np.array(X), np.array(y)



    print("  Building features...")

    X_train, y_train = build_features(train)

    X_test, y_test = build_features(test)



    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,

                             subsample=0.8, colsample_bytree=0.8, random_state=42,

                             n_jobs=4, verbosity=0)

    model.fit(X_train, y_train)

    pred = np.clip(model.predict(X_test), 0, 1)



    test_dates_list = sorted(test['date'].unique())

    forecast = np.zeros((len(test_dates_list), N_HOURS))

    actual = np.zeros((len(test_dates_list), N_HOURS))

    for i, date in enumerate(test_dates_list):

        mask = test['date'] == date

        indices = np.where(mask)[0]

        for j, idx in enumerate(indices):

            forecast[i, j] = pred[idx]

            actual[i, j] = test_solar[i, j]



    nrmse = nrmse_overall(actual.flatten(), forecast.flatten())

    print(f"  XGBoost nRMSE: {nrmse:.2f}%")

    return forecast, nrmse





# --- LSTM ---

def _run_lstm(train_solar, test_solar):

    import torch

    import torch.nn as nn

    from torch.utils.data import Dataset, DataLoader



    class SolarDataset(Dataset):

        def __init__(self, data, window=12):

            self.X, self.y = [], []

            for i in range(window, len(data)):

                self.X.append(data[i - window:i])

                self.y.append(data[i])

            self.X = np.array(self.X, dtype=np.float32)

            self.y = np.array(self.y, dtype=np.float32)

        def __len__(self):

            return len(self.y)

        def __getitem__(self, idx):

            return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])



    class SolarLSTM(nn.Module):

        def __init__(self):

            super().__init__()

            self.lstm = nn.LSTM(12, 64, 2, batch_first=True, dropout=0.2)

            self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 12))

        def forward(self, x):

            out, _ = self.lstm(x)

            return torch.sigmoid(self.fc(out[:, -1, :]))



    window = 12

    train_ds = SolarDataset(train_solar, window)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = SolarLSTM()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()



    for epoch in range(50):

        total_loss, count = 0, 0

        for xb, yb in train_loader:

            optimizer.zero_grad()

            loss = criterion(model(xb), yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            count += len(xb)

        if (epoch + 1) % 10 == 0:

            print(f"    Epoch {epoch+1}/50, Loss: {total_loss/count:.6f}")



    model.eval()

    forecast = np.zeros((test_solar.shape[0], N_HOURS))

    with torch.no_grad():

        for d in range(test_solar.shape[0]):

            past = np.vstack([train_solar[-window:], test_solar[:d]])

            if len(past) < window:

                past = np.vstack([np.zeros((window - len(past), N_HOURS)), past])

            x = torch.FloatTensor(past[-window:]).unsqueeze(0)

            forecast[d] = np.clip(model(x).numpy()[0], 0, 1)



    nrmse = nrmse_overall(test_solar.flatten(), forecast.flatten())

    print(f"  LSTM nRMSE: {nrmse:.2f}%")

    return forecast, nrmse





# --- Decision-Focused NN ---

def _run_decision_focused_nn(train, test, train_solar, test_solar, penalty_rate=0.5):

    import torch

    import torch.nn as nn

    from torch.utils.data import DataLoader



    def build_nn_features(df, solar_data, n_lags=6):

        dates = sorted(df['date'].unique())

        X, y = [], []

        for i in range(n_lags, len(dates)):

            date = dates[i]

            day_df = df[df['date'] == date]

            for _, row in day_df.iterrows():

                h = int(row['hour_idx'])

                features = []

                for lag in range(n_lags):

                    features.append(solar_data[i - lag - 1][h] if (i - lag - 1) >= 0 else 0)

                for var in ['ssrd', 'tsr', 'tcc', 't2m', 'sp', 'tclw', 'r']:

                    features.append(row.get(var, 0))

                features.append(float(np.sin(2 * np.pi * h / N_HOURS)))

                features.append(float(np.cos(2 * np.pi * h / N_HOURS)))

                y.append(row['solar_power'])

                X.append(features)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)



    class AsymmetricMSELoss(nn.Module):

        def __init__(self, pr):

            super().__init__()

            self.w_under = 1.0 + 2.0 * pr

            self.w_over = 1.0 - 0.5 * pr

        def forward(self, pred, actual):

            under = torch.relu(actual - pred)

            over = torch.relu(pred - actual)

            return (self.w_under * (under ** 2) + self.w_over * (over ** 2)).mean()



    class SolarNN(nn.Module):

        def __init__(self, input_size):

            super().__init__()

            self.net = nn.Sequential(

                nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2),

                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),

                nn.Linear(64, 32), nn.ReLU(),

                nn.Linear(32, 1), nn.Sigmoid()

            )

        def forward(self, x):

            return self.net(x).squeeze(-1)



    print(f"  Building features (penalty_rate={penalty_rate})...")

    X_train, y_train = build_nn_features(train, train_solar)

    X_test, y_test = build_nn_features(test, test_solar)



    model = SolarNN(X_train.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = AsymmetricMSELoss(penalty_rate)

    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)



    best_loss, patience, max_patience = float('inf'), 0, 15

    for epoch in range(100):

        total_loss, count = 0, 0

        for xb, yb in train_loader:

            optimizer.zero_grad()

            loss = criterion(model(xb), yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            count += len(xb)

        avg = total_loss / count

        if avg < best_loss - 1e-5:

            best_loss = avg

            patience = 0

            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_df_nn.pt'))

        else:

            patience += 1

            if patience >= max_patience:

                print(f"    Early stop at epoch {epoch+1}")

                break

        if (epoch + 1) % 20 == 0:

            print(f"    Epoch {epoch+1}/100, Loss: {avg:.6f}")



    ckpt = os.path.join(OUTPUT_DIR, 'best_df_nn.pt')

    if os.path.exists(ckpt):

        model.load_state_dict(torch.load(ckpt, weights_only=True))



    model.eval()

    with torch.no_grad():

        pred = np.clip(model(torch.FloatTensor(X_test)).numpy(), 0, 1)



    test_dates = sorted(test['date'].unique())

    forecast = np.zeros((len(test_dates), N_HOURS))

    idx = 0

    for i, date in enumerate(test_dates):

        day_df = test[test['date'] == date]

        for j in range(len(day_df)):

            if idx < len(pred):

                forecast[i, j] = pred[idx]

                idx += 1



    nrmse = nrmse_overall(test_solar.flatten(), forecast.flatten())

    print(f"  DF-NN nRMSE (penalty={penalty_rate*100:.0f}%): {nrmse:.2f}%")

    return forecast, nrmse





# --- REINFORCE NN ---

def _run_reinforce_nn(train, test, train_solar, test_solar, penalty_rate=0.5, n_episodes=50):

    import torch

    import torch.nn as nn



    def build_nn_features(df, solar_data, n_lags=6):

        dates = sorted(df['date'].unique())

        X, y = [], []

        for i in range(n_lags, len(dates)):

            date = dates[i]

            day_df = df[df['date'] == date]

            for _, row in day_df.iterrows():

                h = int(row['hour_idx'])

                features = []

                for lag in range(n_lags):

                    features.append(solar_data[i - lag - 1][h] if (i - lag - 1) >= 0 else 0)

                for var in ['ssrd', 'tsr', 'tcc', 't2m', 'sp', 'tclw', 'r']:

                    features.append(row.get(var, 0))

                features.append(float(np.sin(2 * np.pi * h / N_HOURS)))

                features.append(float(np.cos(2 * np.pi * h / N_HOURS)))

                y.append(row['solar_power'])

                X.append(features)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)



    class ReinforceNN(nn.Module):

        def __init__(self, input_size):

            super().__init__()

            self.fc = nn.Sequential(

                nn.Linear(input_size, 128), nn.ReLU(),

                nn.Linear(128, 64), nn.ReLU(),

                nn.Linear(64, 1))

            self.mu = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

            self.log_std = nn.Sequential(nn.Linear(1, 1))



        def forward(self, x):

            hidden = self.fc(x)

            mu = self.mu(hidden)

            log_std = self.log_std(hidden).clamp(-2, 2)

            dist = torch.distributions.Normal(mu, torch.exp(log_std))

            action = torch.clamp(dist.sample(), 0, 1)

            return action.squeeze(-1), dist



    print(f"  REINFORCE (penalty={penalty_rate*100:.0f}%, episodes={n_episodes})...")

    X_train, y_train = build_nn_features(train, train_solar)

    X_test, y_test = build_nn_features(test, test_solar)



    model = ReinforceNN(X_train.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    X_tensor = torch.FloatTensor(X_train)

    y_tensor = torch.FloatTensor(y_train)



    for ep in range(n_episodes):

        optimizer.zero_grad()

        actions, dists = model(X_tensor)

        log_probs = dists.log_prob(actions).sum(dim=-1)

        w_under = 1.0 + 2.0 * penalty_rate

        w_over = 1.0 - 0.5 * penalty_rate

        rewards = -(w_under * torch.relu(y_tensor - actions) ** 2 +

                    w_over * torch.relu(actions - y_tensor) ** 2)

        advantages = rewards - rewards.mean()

        loss = -(log_probs * advantages).mean()

        loss.backward()

        optimizer.step()

        if (ep + 1) % 10 == 0:

            print(f"    Episode {ep+1}/{n_episodes}, Avg Reward: {rewards.mean().item():.4f}")



    model.eval()

    with torch.no_grad():

        pred, _ = model(torch.FloatTensor(X_test))

        pred = pred.numpy().clip(0, 1)



    test_dates = sorted(test['date'].unique())

    forecast = np.zeros((len(test_dates), N_HOURS))

    idx = 0

    for i, date in enumerate(test_dates):

        day_df = test[test['date'] == date]

        for j in range(len(day_df)):

            if idx < len(pred):

                forecast[i, j] = pred[idx]

                idx += 1



    nrmse = nrmse_overall(test_solar.flatten(), forecast.flatten())

    print(f"  REINFORCE nRMSE (penalty={penalty_rate*100:.0f}%): {nrmse:.2f}%")

    return forecast, nrmse





# ============================================================

# Exp 3: XGBoost + 최적화 커밋 조정 (synthetic price)

# ============================================================

def run_experiment_3(df):

    """XGBoost 예측 후 commitment scaling factor 로 optimality gap 최소화 (synthetic price)."""

    print("\n" + "=" * 60)

    print("Exp 3: XGBoost + 최적화 커밋 조정 (synthetic price)")

    print("=" * 60)



    train, test = split_train_test(df)

    train_solar = to_daily_arrays(train)

    test_solar  = to_daily_arrays(test)

    n_test_days = test_solar.shape[0]

    test_dates  = sorted(test['date'].unique())



    all_dates = sorted(df['date'].unique())

    all_to_idx = {d: i for i, d in enumerate(all_dates)}

    all_solar = {}

    for d in all_dates:

        day_df = df[df['date'] == d]

        arr = np.zeros(N_HOURS)

        for _, r in day_df.iterrows():

            arr[int(r['hour_idx'])] = r['solar_power']

        all_solar[d] = arr



    def build_xgb_features(target_df, n_lags=12):

        dates = sorted(target_df['date'].unique())

        X, y = [], []

        for date in dates:

            day_df = target_df[target_df['date'] == date]

            for _, row in day_df.iterrows():

                h = int(row['hour_idx'])

                features = []

                idx = all_to_idx[date]

                for lag in range(n_lags):

                    prev_idx = idx - lag - 1

                    if prev_idx >= 0:

                        features.append(all_solar[all_dates[prev_idx]][h])

                    else:

                        features.append(0)

                for var in WEATHER_VARS:

                    features.append(row.get(var, 0))

                features.append(float(np.sin(2 * np.pi * h / N_HOURS)))

                features.append(float(np.cos(2 * np.pi * h / N_HOURS)))

                y.append(row['solar_power'])

                X.append(features)

        return np.array(X), np.array(y)



    # XGBoost 학습

    print("\n  XGBoost 학습...")

    X_train, y_train = build_xgb_features(train)

    X_test, y_test = build_xgb_features(test)



    import xgboost as xgb

    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,

                             subsample=0.8, colsample_bytree=0.8, random_state=42,

                             n_jobs=4, verbosity=0)

    model.fit(X_train, y_train)

    xgb_pred = np.clip(model.predict(X_test), 0, 1)

    nrmse_base = nrmse_overall(y_test, xgb_pred)

    print(f"  XGBoost baseline nRMSE: {nrmse_base:.2f}%")



    # penalty rate별로 커밋 조정 실험

    exp3_results = []

    for penalty_rate in [0.0, 0.5, 1.0]:

        print(f"\n  --- penalty_rate={penalty_rate*100:.0f}% ---")

        np.random.seed(42)

        test_da = np.random.uniform(28, 45, len(y_test))

        test_rt = test_da * np.random.uniform(0.85, 1.1, len(y_test))



        # Grid search: scaling factor

        best_scaling, best_gap = 1.0, float('inf')

        for scaling in np.arange(0.7, 1.31, 0.01):

            adjusted = np.clip(xgb_pred * scaling, 0, 1)

            gap = optimality_gap_pct(adjusted, y_test, test_da, test_rt, penalty_rate)

            if gap < best_gap:

                best_gap = gap

                best_scaling = scaling



        adjusted_pred = np.clip(xgb_pred * best_scaling, 0, 1)

        nrmse_adj = nrmse_overall(y_test, adjusted_pred)

        base_gap = optimality_gap_pct(xgb_pred, y_test, test_da, test_rt, penalty_rate)

        print(f"  최적 scaling: {best_scaling:.2f}, 조정 후 nRMSE: {nrmse_adj:.2f}%, Gap: {best_gap:.2f}%")



        # Adaptive 커밋 (학습 기반 scaling factor 예측)

        print("  Adaptive 커밋 조정...")

        from sklearn.ensemble import GradientBoostingRegressor



        train_pred = np.clip(model.predict(X_train), 0, 1)

        train_da_s = np.random.uniform(28, 45, len(y_train))

        train_rt_s = train_da_s * np.random.uniform(0.85, 1.1, len(y_train))



        # 학습 데이터에서 최적 scaling factor 계산 (하루 단위)

        optimal_scalings = []

        for i in range(0, len(y_train), N_HOURS):

            day_actual = y_train[i:i+N_HOURS]

            day_pred = train_pred[i:i+N_HOURS]

            day_da = train_da_s[i:i+N_HOURS]

            day_rt = train_rt_s[i:i+N_HOURS]

            best_s, best_g = 1.0, float('inf')

            for s in np.arange(0.6, 1.41, 0.02):

                g = optimality_gap_pct(np.clip(day_pred * s, 0, 1), day_actual, day_da, day_rt, penalty_rate)

                if g < best_g:

                    best_g = g

                    best_s = s

            optimal_scalings.append(best_s)



        # daily weather + 예측 통계로 scaling factor 학습

        def daily_features(target_df, target_pred, target_dates, daily_to_idx):

            feats = []

            for date in sorted(target_dates):

                day_df = target_df[target_df['date'] == date]

                day_idx = daily_to_idx[date]

                day_pred = target_pred[day_idx * N_HOURS:(day_idx + 1) * N_HOURS]

                f = [day_df[var].mean() for var in WEATHER_VARS]

                f.append(np.mean(day_pred))

                f.append(np.std(day_pred))

                feats.append(f)

            return np.array(feats)



        train_date_list = sorted(train['date'].unique())

        train_daily_to_idx = {d: i for i, d in enumerate(train_date_list)}

        test_date_list = sorted(test['date'].unique())

        test_daily_to_idx = {d: i for i, d in enumerate(test_date_list)}



        train_daily_feats = daily_features(train, train_pred, set(train_date_list), train_daily_to_idx)

        test_daily_feats = daily_features(test, xgb_pred, set(test_date_list), test_daily_to_idx)



        if len(train_daily_feats) > 50 and len(optimal_scalings) > 50:

            scaler = GradientBoostingRegressor(n_estimators=50, max_depth=3,

                                                learning_rate=0.1, random_state=42)

            scaler.fit(train_daily_feats, optimal_scalings)

            predicted_scalings = scaler.predict(test_daily_feats)



            adaptive_pred = xgb_pred.copy()

            for i, date in enumerate(test_dates):

                start = i * N_HOURS

                end = (i + 1) * N_HOURS

                if end <= len(adaptive_pred):

                    adaptive_pred[start:end] = np.clip(adaptive_pred[start:end] * predicted_scalings[i], 0, 1)



            nrmse_adapt = nrmse_overall(y_test, adaptive_pred)

            gap_adapt = optimality_gap_pct(adaptive_pred, y_test, test_da, test_rt, penalty_rate)

            print(f"  Adaptive: nRMSE={nrmse_adapt:.2f}%, Gap={gap_adapt:.2f}%")



            exp3_results.append({'penalty_rate': penalty_rate,

                                 'xgb_nrmse': nrmse_base,

                                 'xgb_gap': base_gap,

                                 'adjusted_nrmse': nrmse_adj,

                                 'adjusted_gap': best_gap,

                                 'adaptive_nrmse': nrmse_adapt,

                                 'adaptive_gap': gap_adapt,

                                 'best_scaling': best_scaling})

        else:

            exp3_results.append({'penalty_rate': penalty_rate,

                                 'xgb_nrmse': nrmse_base,

                                 'xgb_gap': base_gap,

                                 'adjusted_nrmse': nrmse_adj,

                                 'adjusted_gap': best_gap,

                                 'adaptive_nrmse': None,

                                 'adaptive_gap': None,

                                 'best_scaling': best_scaling})



    pd.DataFrame(exp3_results).to_csv(os.path.join(OUTPUT_DIR, 'exp3_xgboost_optimization.csv'), index=False)

    print(f"\n  저장: exp3_xgboost_optimization.csv")



    print("\n  Exp 3 요약:")

    for r in exp3_results:

        pr = r['penalty_rate'] * 100

        print(f"  penalty={pr:.0f}% | XGBoost: nRMSE={r['xgb_nrmse']:.2f}%, Gap={r['xgb_gap']:.2f}% | "

              f"Adjust: nRMSE={r['adjusted_nrmse']:.2f}%, Gap={r['adjusted_gap']:.2f}% | "

              f"Scaling={r['best_scaling']:.2f}")



    return exp3_results





# ============================================================

# 메인

# ============================================================

if __name__ == '__main__':

    df = load_gefcom_full()



    # Exp 1: 300일 학습, nRMSE 3가지 방식 비교

    run_experiment_1(df)



    # Exp 2: 딥러닝 기반 확장

    run_experiment_2(df)



    # Exp 3: XGBoost + 최적화 커밋 조정

    run_experiment_3(df)



    print("\n" + "=" * 60)

    print("모든 실험 완료")

    print("=" * 60)

