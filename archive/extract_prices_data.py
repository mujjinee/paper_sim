
"""

MISO 가격 데이터 (Day-Ahead + Real-Time)를 data_raw Excel -> CSV로 뽑아냄.



출력:

  data/prices/day_ahead_prices.csv

  data/prices/real_time_prices.csv



Solar + Weather는 extract_solar_weather.py 에서 처리함.



Usage:

    python extract_prices_data.py

"""

import os

import glob

import pandas as pd



#BASE = r"D:\marco\AIBM\PaperModel\.claude\skills\paper-agent"
BASE = r"D:\03_JiWon\JiWonProject"

RAW  = os.path.join(BASE, "data_raw")

OUT  = os.path.join(BASE, "data")





def _find_miso_system_row(df):

    """엑셀에서 'MISO System'이 있는 행 번호를 찾음."""

    for r in range(len(df)):

        val = str(df.iloc[r, 1]).strip()

        if val == "MISO System":

            return r

    return None





def _extract_miso_prices(xls_dir_pattern, column_name):

    """MISO Excel 파일에서 MISO System 가격 추출 (공통 로직)."""

    outpath = os.path.join(OUT, "prices", f"{column_name.lower()}_prices.csv")

    records = []



    xls_dirs = sorted(glob.glob(os.path.join(RAW, "miso", "raw", xls_dir_pattern)))

    for xls_dir in xls_dirs:

        if not os.path.isdir(xls_dir):

            continue

        for xls in sorted(glob.glob(os.path.join(xls_dir, "*.xls"))):

            df = pd.read_excel(xls, header=None)

            header = str(df.iloc[1, 0])

            market_date_str = header.split(":")[-1].strip()

            market_date = pd.to_datetime(market_date_str, format="%m/%d/%Y")



            sys_row = _find_miso_system_row(df)

            if sys_row is None:

                continue

            first_price_row = sys_row + 1



            for h in range(24):

                row = first_price_row + h

                if row < len(df):

                    price = df.iloc[row, 1]

                    if isinstance(price, (int, float)):

                        ts = market_date + pd.Timedelta(hours=h)

                        records.append({"TIMESTAMP": ts, column_name: float(price)})



    rdf = pd.DataFrame(records).sort_values("TIMESTAMP").reset_index(drop=True)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    rdf.to_csv(outpath, index=False)

    print(f"[{column_name}] {len(rdf)} rows saved -> {outpath}")

    return rdf





# ============================================================

# 1. MISO Day-Ahead 가격

# ============================================================

def extract_miso_da():

    return _extract_miso_prices("201*_da_pr_xls", "DA_LMP")





# ============================================================

# 2. MISO Real-Time 가격

# ============================================================

def extract_miso_rt():

    return _extract_miso_prices("201*_rt_pr_xls", "RT_LMP")





if __name__ == "__main__":

    print("=== MISO Day-Ahead ===")

    extract_miso_da()

    print("\n=== MISO Real-Time ===")

    extract_miso_rt()

