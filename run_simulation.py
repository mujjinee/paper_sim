
"""

논문(Karimi & Kwon, 2022, Applied Energy 326: 119929) 실험 재현 스크립트.



논문 제목: Optimization-driven uncertainty forecasting: Application to day-ahead commitment with renewable energy resources



데이터:

  - Solar/Weather: GEFCom2014 Solar Track (호주 3개 태양광 발전소, ECMWF 기상 예보)

  - Price: MISO day-ahead / real-time LMP



시간대:

  GEFCom2014 timestamp는 UTC 기준. 호주 현지 시간은 UTC+11(여름 DST) / UTC+10(겨울).

  Solar 발전량이 있는 낮 시간은 UTC 00:00~11:00 범위.

  매일 정확히 12시간을 추출하여 사용.



학습/테스트:

  - 학습: 1~3월 (90일), 테스트: 4월 (30일)

  - 현재 MISO 가격 데이터가 2014년 1~4월까지만 있어서 이 기간으로 제한됨.

  - 논문은 300일 학습 / 100일 테스트를 사용함.



모델:

  1. AR (Auto-Regressive, 내생적): 과거 n_lags 일 같은 시간대 데이터를 선형회귀 입력

     - 논문 Section 4.3.1, Eq. (3): direct multi-step 방식

     - 하루 12시간에 대해 각각 별도의 모델을 학습 (각 시간대 독립)

  2. MLR (Multiple Linear Regression, 외생적): 기상 변수 12개를 입력

     - 논문 Section 4.3.2, Eq. (6)

     - 하루 12시간에 대해 각각 별도의 모델을 학습

  3. Proposed (Optimization-Driven Forecasting):

     - 논문 Section 4.4, Eq. (9): AR 모델 + unit commitment 최적화를 통합

     - W1 (optimality gap 가중치), W2 (forecasting error 가중치) 로 trade-off 조정



평가지표:

  - nRMSE: Normalized Root Mean Square Error, 예측 정확도

    논문 Eq. (11): RMSE / 평균 발전량 × 100

  - Optimality Gap: (최적 이익 - 실제 이익) / 최적 이익 × 100

    논문 Eq. (12): commitment 전략이 이상적인 최적 대비 얼마나 이익을 잃는지

"""

import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LinearRegression

from scipy.optimize import minimize

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


import os
#BASE = r"D:\marco\AIBM\PaperModel\.claude\skills\paper-agent"
BASE = r"D:\03_JiWon\JiWonProject"
#BASE1 = r"."

#BASE1= os.getcwd()
DATA_PATH = os.path.join(BASE, "data", "merged_for_simulation.csv")

OUTPUT_DIR = os.path.join(BASE, "results", "simulation_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)



# 하루 12시간 사용 (UTC 00:00~11:00 = 호주 현지 낮 시간)

N_HOURS = 12





# ============================================================

# 1. 데이터 로딩 및 전처리

# ============================================================

def load_data():

    """merged_for_simulation.csv 를 읽어 학습/테스트에 쓸 형태로 정리함.



    반환:

        DataFrame (timestamp, date, hour_idx, solar_norm, da_price, rt_price, 기상 변수 12개)

    """

    df = pd.read_csv(DATA_PATH)

    df.columns = [c.strip() for c in df.columns]

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=False)



    # UTC 00:00~11:00 만 사용 (호주 현지 낮 시간)

    df = df[df['timestamp'].dt.hour.between(0, 11)].reset_index(drop=True)



    df['date'] = df['timestamp'].dt.date

    df['hour_idx'] = df['timestamp'].dt.hour  # 0~11 (하루 내 시간대 인덱스)



    # solar_power는 이미 0~1 로 정규화되어 있음 (30MW로 나누지 않음)

    df['solar_norm'] = df['solar_power']



    # 결측치 있는 행 제거 (RT price 마지막 날 누락으로 merge 시 발생)

    df = df.dropna(subset=['solar_norm', 'da_price', 'rt_price'])



    return df





def split_train_test(df):

    """1~3 월 학습, 4 월 테스트로 나눔.



    반환:

        (train_df, test_df)

    """

    months = df['timestamp'].dt.month

    train = df[months.isin([1, 2, 3])].copy()

    test  = df[months == 4].copy()

    return train, test





def to_daily_arrays(df):

    """DataFrame 을 (n_days, N_HOURS) 형태의 numpy 배열로 변환함.



    매일 정확히 N_HOURS=12 시간이므로 2차원 배열로 만들 수 있음.

    각 행 = 하루, 각 열 = 시간대 (0~11)



    반환:

        solar_arr, da_arr, rt_arr: 각 (n_days, 12) 형태의 float64 배열

    """

    dates = sorted(df['date'].unique())

    n_days = len(dates)

    solar_arr = np.zeros((n_days, N_HOURS))

    da_arr = np.zeros((n_days, N_HOURS))

    rt_arr = np.zeros((n_days, N_HOURS))



    for i, date in enumerate(dates):

        day_df = df[df['date'] == date]

        for _, row in day_df.iterrows():

            h = int(row['hour_idx'])

            if 0 <= h < N_HOURS:

                solar_arr[i, h] = row['solar_norm']

                da_arr[i, h] = row['da_price']

                rt_arr[i, h] = row['rt_price']



    return solar_arr, da_arr, rt_arr





# ============================================================

# 2. Unit Commitment 계산 (논문 Section 4.2, Eq. 1a-1g)

# ============================================================

def unit_commitment_profit(commitment, actual, da_price, rt_price, penalty_cost):

    """Day-ahead commitment 이 주어졌을 때 이익을 계산함.



    논문 Eq. (1a) 목적함수:

      max Σ { DP_t · x_t + RP_t · y⁺_t - RP_t · y⁻_t - PC_t · y⁻_t }



    매개변수:

        commitment: day-ahead commitment x_t (예측치를 그대로 커밋)

        actual: 실제 발전량 S_t

        da_price: day-ahead 가격 DP_t

        rt_price: real-time 가격 RP_t

        penalty_cost: shortage penalty PC_t (= penalty_rate × da_price)



    반환:

        전체 시간대에 대한 총 이익 (float)

    """

    mismatch = actual - commitment  # y_t = S_t - x_t

    surplus  = np.maximum(mismatch, 0)   # y⁺_t: 실제 > 커밋 (남은 전력)

    shortage = np.maximum(-mismatch, 0)  # y⁻_t: 커밋 > 실제 (부족한 전력)

    # profit = DA에서 파는 이익 + RT에서 surplus 보상 - RT에서 shortage 조정 - shortage 패널티

    profit = np.sum(da_price * commitment + rt_price * surplus - rt_price * shortage - penalty_cost * shortage)

    return profit





def oracle_profit(actual, da_price, rt_price, penalty_cost):

    """지혜신적 최적 이익 (사후에 알았을 때的最佳 커밋).



    논문의 optimality gap 계산에 분모로 사용.

    DA 가격이 RT 가격보다 높으면 실제 발전량 전체를 DA에 커밋하고,

    그렇지 않으면 커밋하지 않는 것이 최적.



    반환:

        최적 이익 (float)

    """

    optimal_commit = np.where(da_price > rt_price, actual, 0)

    return unit_commitment_profit(optimal_commit, actual, da_price, rt_price, penalty_cost)





def nrmse_overall(actual, predicted):

    """전체 테스트 기간을 하나로 합쳐 nRMSE 계산.



    nRMSE = 100 × RMSE / 평균 발전량 (논문 Eq. 11)



    반환:

        nRMSE (%)

    """

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return 100 * rmse / np.mean(actual) if np.mean(actual) > 0 else 0





def nrmse_per_day_average(daily_actual, daily_predicted):

    """하루 12시간에 대해 각 날마다 nRMSE 계산 후 전체 평균.



    논문이 nRMSE 를 하루별 계산 후 평균한 것인지, 전체를 하나로 합쳐 계산한

    것인지 명시하지 않음. 두 방법을 모두 보고함.



    매개변수:

        daily_actual: (n_test_days, 12) 실제 발전량 배열

        daily_predicted: (n_test_days, 12) 예측 발전량 배열



    반환:

        각 날 nRMSE 의 평균 (%)

    """

    nrmse_list = []

    for day in range(daily_actual.shape[0]):

        day_nrmse = nrmse_overall(daily_actual[day], daily_predicted[day])

        nrmse_list.append(day_nrmse)

    return np.mean(nrmse_list)





def optimality_gap_pct(commitment, actual, da_price, rt_price, penalty_rate):

    """Optimality gap 계산 (논문 Eq. 12).



    G = 100 × (최적 이익 - 실제 이익) / 최대 가능한 이익



    반환:

        optimality gap (%)

    """

    pc = penalty_rate * da_price  # penalty cost = penalty_rate × DA 가격

    o_p = oracle_profit(actual, da_price, rt_price, pc)

    a_p = unit_commitment_profit(commitment, actual, da_price, rt_price, pc)

    if abs(o_p) > 1e-10:

        return 100 * (o_p - a_p) / o_p

    return 0





# ============================================================

# 3. AR 모델 - direct multi-step (논문 Section 4.3.1, Eq. 3)

# ============================================================

class ARModel:

    """Auto-Regressive 예측 모델 (내생적, 기상 변수 없음).



    하루 12시간 (h=0~11) 에 대해 각각 별도의 LinearRegression 모델을 학습.

    각 시간대 h 에 대해 과거 n_lags 일의 같은 시간대 값을 입력으로 사용.



    논문 Eq. (3): Ŝ_t = α + Σ β_h · S_{t-h}

    여기서 t 는 같은 시간대의 이전观测値.

    """

    def __init__(self, n_lags=12):

        """n_lags: 과거 며칠 데이터를 사용할지 (논문에서 12시간 사용)"""

        self.n_lags = n_lags

        self.models = {}  # hour_idx → LinearRegression 모델



    def fit(self, daily_solar):

        """학습 데이터에서 12개 시간대별 모델을 학습.



        매개변수:

            daily_solar: (n_days, 12) 배열, 매일의 solar 발전량

        """

        n_days = daily_solar.shape[0]

        n_hours = daily_solar.shape[1]

        for h in range(n_hours):

            X, y = [], []

            for day in range(self.n_lags, n_days):

                # 과거 n_lags 일의 같은 시간 h 의 값을 특징으로 사용

                features = [daily_solar[day - i - 1, h] for i in range(self.n_lags)]

                X.append(features)

                y.append(daily_solar[day, h])

            X, y = np.array(X), np.array(y)

            # 학습 데이터가 5개 이상일 때만 모델 학습

            if len(X) > 5:

                model = LinearRegression()

                model.fit(X, y)

                self.models[h] = model



    def predict_next_day(self, past_daily):

        """과거 데이터를 기반으로 내일 하루 12시간을 예측.



        매개변수:

            past_daily: (n_lags, 12) 배열, 가장 최근 n_lags 일의 발전량



        반환:

            (12,) 배열, 0~1 로 클립된 예측값

        """

        n_hours = past_daily.shape[1]

        forecast = np.zeros(n_hours)

        for h in range(n_hours):

            if h in self.models:

                # 과거 n_lags 일의 같은 시간 h 값을 특징으로 사용

                features = [past_daily[self.n_lags - i - 1, h] for i in range(self.n_lags)]

                pred = self.models[h].predict([features])[0]

            else:

                # 모델이 학습되지 않았으면 과거 평균 사용

                pred = np.mean(past_daily[:, h])

            # 발전량은 0~1 범위이므로 클립

            forecast[h] = np.clip(pred, 0, 1)

        return forecast





# ============================================================

# 4. MLR 모델 (논문 Section 4.3.2, Eq. 6)

# ============================================================

class MLRModel:

    """Multiple Linear Regression 예측 모델 (외생적, 기상 변수 사용).



    하루 12시간에 대해 각각 별도의 LinearRegression 모델을 학습.

    각 시간대 h 에 대해 기상 변수를 입력으로 사용.



    논문 Eq. (6): Ŝ_t = α + Σ β_k · V_{k,t}

    여기서 V_k 는 k 번째 독립 변수 (기상 변수).



    논문에서는 backward stepwise 로 SSRD, TSR, Hour 3개만 최종 선택.

    use_stepwise=True 로 설정 시 논문과 동일하게 3개 변수만 사용.

    """

    def __init__(self, weather_vars, use_stepwise=False):

        """weather_vars: 기상 변수 컬럼명 리스트

           use_stepwise: True 면 논문과 동일하게 ssrd, tsr, hour 3개만 사용"""

        self.weather_vars = weather_vars

        self.use_stepwise = use_stepwise

        self.models = {}  # hour_idx → LinearRegression 모델



    def _get_vars(self, hour_idx):

        """사용할 변수 컬럼명 반환.



        use_stepwise=True 면 논문과 동일하게 ssrd, tsr, hour 3개만 사용.

        그렇지 않으면 12개 전 변수 사용.

        """

        if self.use_stepwise:

            return ['ssrd', 'tsr', 'hour']

        return self.weather_vars



    def fit(self, daily_solar, train_df, n_hours):

        """학습 데이터에서 12개 시간대별 MLR 모델을 학습.



        매개변수:

            daily_solar: (n_days, 12) 배열

            train_df: 학습 DataFrame (기상 변수 포함)

            n_hours: 사용 시간대 수 (12)

        """

        dates = sorted(train_df['date'].unique())

        n_days = len(dates)

        for h in range(n_hours):

            X, y = [], []

            for day in range(n_days):

                date = dates[day]

                row = train_df[(train_df['date'] == date) & (train_df['hour_idx'] == h)]

                if len(row):

                    vars = self._get_vars(h)

                    feature_row = []

                    for var in vars:

                        if var == 'hour':

                            feature_row.append(h)  # 시간대 인덱스 (0~11)

                        else:

                            feature_row.append(row.iloc[0][var])

                    X.append(feature_row)

                    y.append(daily_solar[day, h])

            X, y = np.array(X), np.array(y)

            if len(X) > 5:

                model = LinearRegression()

                model.fit(X, y)

                self.models[h] = model



    def predict_next_day(self, test_day_df, n_hours):

        """테스트 날의 기상 변수로 내일 하루 12시간을 예측.



        매개변수:

            test_day_df: 테스트 날의 DataFrame (기상 변수 포함)

            n_hours: 사용 시간대 수 (12)



        반환:

            (12,) 배열, 0~1 로 클립된 예측값

        """

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

# 5. 최적화 기반 예측 - 제안 방식 (논문 Section 4.4, Eq. 9)

# ============================================================

def optimization_driven_ar(train_solar_daily, train_da_daily, train_rt_daily,

                           penalty_rate=0.5, W1=1.0, W2=1.0, n_lags=12):

    """제안 모델 (Optimization-Driven Forecasting, endogenous/AR 버전).



    기존 AR 모델과 달리 회귀 계수 학습 시 예측 오차만 최소화하는 것이 아니라,

    optimality gap (commitment 이익 손실) 도 동시에 최소화하는 계수를 찾음.



    논문 Eq. (9) 목적함수:

      min W1 · Σ [optimality gap] + W2 · Σ [forecasting error]



    W1/W2 비율이 클수록 optimality gap 최소화 쪽으로,

    W1=0, W2=1 이면 순수 AR 과 동일.



    매개변수:

        train_solar_daily: (n_train_days, 12) 학습 태양광 배열

        train_da_daily: (n_train_days, 12) 학습 DA 가격 배열

        train_rt_daily: (n_train_days, 12) 학습 RT 가격 배열

        penalty_rate: shortage penalty rate (0.0~1.0)

        W1: optimality gap 가중치

        W2: forecasting error 가중치

        n_lags: 과거 사용 일 수



    반환:

        (forecast, info) - 첫 테스트 날에 대한 예측값 (12,) 과 학습된 계수 정보

    """

    n_days = train_solar_daily.shape[0]

    n_hours = train_solar_daily.shape[1]

    if n_days < 3:

        return np.zeros(n_hours), None



    def objective(params):

        """통합 목적함수: optimality gap + forecasting error.



        params: [alpha, beta_1, beta_2, ..., beta_n_lags]

        """

        alpha = params[0]

        betas = params[1:1 + n_lags]

        total_gap = 0

        total_error = 0



        for day in range(1, n_days):

            # 과거 n_lags 일의 같은 시간대 평균

            prev_avg = np.mean(train_solar_daily[max(0, day - n_lags):day, :], axis=0)

            forecast = np.clip(alpha + np.sum(betas) * prev_avg, 0, 1)

            actual = train_solar_daily[day]

            da_p, rt_p = train_da_daily[day], train_rt_daily[day]

            pc = penalty_rate * da_p



            o_p = oracle_profit(actual, da_p, rt_p, pc)

            a_p = unit_commitment_profit(forecast, actual, da_p, rt_p, pc)



            if abs(o_p) > 1e-10:

                total_gap += (o_p - a_p) / o_p

            total_error += np.sum(np.abs(forecast - actual))



        return W1 * total_gap + W2 * total_error



    # 초기값 설정

    x0 = np.zeros(1 + n_lags)

    x0[0] = 0.5  # alpha 초기값

    x0[1:] = 1.0 / n_lags  # beta 초기값 (평균 분포)



    result = minimize(objective, x0, method='L-BFGS-B',

                      options={'maxiter': 500, 'ftol': 1e-8})



    # 테스트 첫날 예측

    last_avg = np.mean(train_solar_daily[-n_lags:], axis=0)

    alpha_opt = result.x[0]

    betas_sum = np.sum(result.x[1:1 + n_lags])

    test_forecast = np.clip(alpha_opt + betas_sum * last_avg, 0, 1)



    return test_forecast, {'alpha': alpha_opt, 'betas_sum': betas_sum}





# ============================================================

# 6. 실행 (메인)

# ============================================================

def run_simulation():

    """AR, MLR, 제안 방식 세 모델을 학습/예측/평가하고 결과를 CSV 로 저장."""

    print("데이터 로딩...")

    df = load_data()

    train, test = split_train_test(df)



    print(f"학습: {len(train)}행 ({len(train)//N_HOURS}일), 테스트: {len(test)}행 ({len(test)//N_HOURS}일)")



    # DataFrame → (n_days, 12) 배열 변환

    train_solar_daily, train_da_daily, train_rt_daily = to_daily_arrays(train)

    test_solar_daily, test_da_daily, test_rt_daily = to_daily_arrays(test)



    n_train_days = train_solar_daily.shape[0]

    n_test_days = test_solar_daily.shape[0]



    # 논문과 동일한 파라미터 스윕

    penalty_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    W_ratios = [(1, 20), (1, 10), (1, 5), (1, 2), (1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (1, 0)]



    results_ar       = {}

    results_mlr      = {}

    results_mlr_sw   = {}  # stepwise 변수 선택 MLR

    results_proposed = {}



    # 전체 테스트 기간을 1차원으로 편평하게 (nRMSE/gap 계산용)

    test_solar_all = test_solar_daily.flatten()

    test_da_all = test_da_daily.flatten()

    test_rt_all = test_rt_daily.flatten()



    # --- AR 기준선 (논문 Table 3) ---

    print("\n=== AR 모델 학습 (direct multi-step) ===")

    ar_model = ARModel(n_lags=12)

    ar_model.fit(train_solar_daily)



    print("\n=== AR 예측 실행 ===")

    ar_forecast_all = np.zeros((n_test_days, N_HOURS))

    for day in range(n_test_days):

        # 학습 데이터 마지막 12일 + 테스트 데이터 이전 일 사용

        past = np.vstack([train_solar_daily[-12:], test_solar_daily[:day]])

        ar_forecast_all[day] = ar_model.predict_next_day(past)



    for penalty_rate in penalty_rates:

        # nRMSE 두 가지 방식 모두 보고

        ar_nrmse_all = nrmse_overall(test_solar_all, ar_forecast_all.flatten())

        ar_nrmse_avg = nrmse_per_day_average(test_solar_daily, ar_forecast_all)

        ar_gap = optimality_gap_pct(ar_forecast_all.flatten(), test_solar_all, test_da_all, test_rt_all, penalty_rate)

        results_ar[penalty_rate] = {

            'nRMSE_overall': ar_nrmse_all, 'nRMSE_per_day_avg': ar_nrmse_avg,

            'optimality_gap': ar_gap

        }

        print(f"  Penalty {penalty_rate*100:.0f}%: AR nRMSE={ar_nrmse_all:.2f}% (day-avg={ar_nrmse_avg:.2f}%), Gap={ar_gap:.2f}%")



    # --- MLR 기준선 (12개 전 변수 사용) ---

    weather_vars = ['tclw', 'tciw', 'sp', 'r', 'tcc', 'u10', 'v10', 't2m', 'ssrd', 'strd', 'tsr', 'tp']

    print("\n=== MLR 모델 학습 (12개 변수) ===")

    mlr_model = MLRModel(weather_vars, use_stepwise=False)

    mlr_model.fit(train_solar_daily, train, N_HOURS)



    mlr_forecast_all = np.zeros((n_test_days, N_HOURS))

    test_dates = sorted(test['date'].unique())

    for day in range(n_test_days):

        date = test_dates[day]

        test_day_df = test[test['date'] == date]

        mlr_forecast_all[day] = mlr_model.predict_next_day(test_day_df, N_HOURS)



    for penalty_rate in penalty_rates:

        mlr_nrmse_all = nrmse_overall(test_solar_all, mlr_forecast_all.flatten())

        mlr_nrmse_avg = nrmse_per_day_average(test_solar_daily, mlr_forecast_all)

        mlr_gap = optimality_gap_pct(mlr_forecast_all.flatten(), test_solar_all, test_da_all, test_rt_all, penalty_rate)

        results_mlr[penalty_rate] = {

            'nRMSE_overall': mlr_nrmse_all, 'nRMSE_per_day_avg': mlr_nrmse_avg,

            'optimality_gap': mlr_gap

        }

        print(f"  Penalty {penalty_rate*100:.0f}%: MLR(12vars) nRMSE={mlr_nrmse_all:.2f}% (day-avg={mlr_nrmse_avg:.2f}%), Gap={mlr_gap:.2f}%")



    # --- MLR 기준선 (backward stepwise: ssrd, tsr, hour 3개만 사용, 논문과 동일) ---

    print("\n=== MLR 모델 학습 (ssrd, tsr, hour 3개만) ===")

    mlr_sw_model = MLRModel(weather_vars, use_stepwise=True)

    mlr_sw_model.fit(train_solar_daily, train, N_HOURS)



    mlr_sw_forecast_all = np.zeros((n_test_days, N_HOURS))

    for day in range(n_test_days):

        date = test_dates[day]

        test_day_df = test[test['date'] == date]

        mlr_sw_forecast_all[day] = mlr_sw_model.predict_next_day(test_day_df, N_HOURS)



    for penalty_rate in penalty_rates:

        mlr_sw_nrmse_all = nrmse_overall(test_solar_all, mlr_sw_forecast_all.flatten())

        mlr_sw_nrmse_avg = nrmse_per_day_average(test_solar_daily, mlr_sw_forecast_all)

        mlr_sw_gap = optimality_gap_pct(mlr_sw_forecast_all.flatten(), test_solar_all, test_da_all, test_rt_all, penalty_rate)

        results_mlr_sw[penalty_rate] = {

            'nRMSE_overall': mlr_sw_nrmse_all, 'nRMSE_per_day_avg': mlr_sw_nrmse_avg,

            'optimality_gap': mlr_sw_gap

        }

        print(f"  Penalty {penalty_rate*100:.0f}%: MLR(sw) nRMSE={mlr_sw_nrmse_all:.2f}% (day-avg={mlr_sw_nrmse_avg:.2f}%), Gap={mlr_sw_gap:.2f}%")



    # --- 제안 방식 (Optimization-Driven AR, 논문 Eq. 9) ---

    print("\n=== 제안 방식 (최적화 기반 AR) 실행 ===")

    for W1, W2 in W_ratios:

        for penalty_rate in penalty_rates:

            forecast, info = optimization_driven_ar(

                train_solar_daily, train_da_daily, train_rt_daily,

                penalty_rate=penalty_rate, W1=W1, W2=W2, n_lags=12

            )

            if forecast is not None and len(forecast) > 0:

                # 현재 테스트 첫날에 대한 예측만 평가

                prop_nrmse = nrmse_overall(test_solar_daily[0], forecast)

                prop_gap = optimality_gap_pct(forecast, test_solar_daily[0],

                                              test_da_daily[0], test_rt_daily[0], penalty_rate)

                results_proposed[(penalty_rate, W1, W2)] = {'nRMSE': prop_nrmse, 'optimality_gap': prop_gap}



    # --- 결과 CSV 저장 ---

    ar_rows = []

    for pr in penalty_rates:

        if pr in results_ar:

            ar_rows.append({'model': 'AR', 'penalty_rate': pr,

                            'nRMSE_overall': results_ar[pr]['nRMSE_overall'],

                            'nRMSE_per_day_avg': results_ar[pr]['nRMSE_per_day_avg'],

                            'optimality_gap': results_ar[pr]['optimality_gap']})



    mlr_rows = []

    for pr in penalty_rates:

        if pr in results_mlr:

            mlr_rows.append({'model': 'MLR(12vars)', 'penalty_rate': pr,

                             'nRMSE_overall': results_mlr[pr]['nRMSE_overall'],

                             'nRMSE_per_day_avg': results_mlr[pr]['nRMSE_per_day_avg'],

                             'optimality_gap': results_mlr[pr]['optimality_gap']})



    mlr_sw_rows = []

    for pr in penalty_rates:

        if pr in results_mlr_sw:

            mlr_sw_rows.append({'model': 'MLR(sw-3vars)', 'penalty_rate': pr,

                                'nRMSE_overall': results_mlr_sw[pr]['nRMSE_overall'],

                                'nRMSE_per_day_avg': results_mlr_sw[pr]['nRMSE_per_day_avg'],

                                'optimality_gap': results_mlr_sw[pr]['optimality_gap']})



    prop_rows = []

    for (pr, w1, w2), val in sorted(results_proposed.items()):

        prop_rows.append({'model': f'Proposed(W1={w1},W2={w2})', 'penalty_rate': pr,

                          'nRMSE': val['nRMSE'], 'optimality_gap': val['optimality_gap']})



    pd.DataFrame(ar_rows).to_csv(os.path.join(OUTPUT_DIR, 'ar_results.csv'), index=False)

    pd.DataFrame(mlr_rows).to_csv(os.path.join(OUTPUT_DIR, 'mlr_results.csv'), index=False)

    pd.DataFrame(mlr_sw_rows).to_csv(os.path.join(OUTPUT_DIR, 'mlr_sw_results.csv'), index=False)

    pd.DataFrame(prop_rows).to_csv(os.path.join(OUTPUT_DIR, 'proposed_results.csv'), index=False)



    print(f"\n=== 결과 저장 완료: {OUTPUT_DIR} ===")

    print(f"\n=== 논문 기대값 vs 실행 결과 (penalty=50%) ===")

    pr = 0.5

    print(f"  AR:     기대 nRMSE=34.76%, Gap=15.04%")

    print(f"           실행 (전체) nRMSE={results_ar[pr]['nRMSE_overall']:.2f}%, Gap={results_ar[pr]['optimality_gap']:.2f}%")

    print(f"           실행 (day-avg) nRMSE={results_ar[pr]['nRMSE_per_day_avg']:.2f}%")

    print(f"  MLR(12vars):    기대 nRMSE=21.76%, Gap=12.59%")

    print(f"           실행 (전체) nRMSE={results_mlr[pr]['nRMSE_overall']:.2f}%, Gap={results_mlr[pr]['optimality_gap']:.2f}%")

    print(f"           실행 (day-avg) nRMSE={results_mlr[pr]['nRMSE_per_day_avg']:.2f}%")

    print(f"  MLR(sw-3vars):  기대 nRMSE=21.76%, Gap=12.59%")

    print(f"           실행 (전체) nRMSE={results_mlr_sw[pr]['nRMSE_overall']:.2f}%, Gap={results_mlr_sw[pr]['optimality_gap']:.2f}%")

    print(f"           실행 (day-avg) nRMSE={results_mlr_sw[pr]['nRMSE_per_day_avg']:.2f}%")



    key = (0.5, 1, 20)

    if key in results_proposed:

        print(f"  제안:   기대 nRMSE=34.89%, Gap=13.91%")

        print(f"           실행 nRMSE={results_proposed[key]['nRMSE']:.2f}%, Gap={results_proposed[key]['optimality_gap']:.2f}%")





if __name__ == '__main__':

    run_simulation()
    
    