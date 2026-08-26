"""Read-only audit for the data extraction and merge pipeline.

It verifies byte-for-value equality between the chosen GEFCom source slice and
the generated solar/weather CSVs, re-extracts MISO System prices in memory,
and verifies that the merged CSV is exactly the result of the current merge
rules.  It does not modify project data.
"""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


BASE = Path(__file__).resolve().parent
GEFCOM = BASE / "data_raw" / "gefcom2014" / "GEFCom2014" / "GEFCom2014 Data" / "GEFCom2014-S_V2" / "Solar" / "Task 15" / "predictors15.csv"
RAW_MISO = BASE / "data_raw" / "miso" / "raw"
START = pd.Timestamp("2014-01-01")
END = pd.Timestamp("2014-04-30 23:00:00")
WEATHER = ["VAR78", "VAR79", "VAR134", "VAR157", "VAR164", "VAR165", "VAR166", "VAR167", "VAR169", "VAR175", "VAR178", "VAR228"]


def miso_system_prices(pattern: str, column: str) -> pd.DataFrame:
    rows = []
    for directory in sorted(RAW_MISO.glob(pattern)):
        for xls in sorted(directory.glob("*.xls")):
            sheet = pd.read_excel(xls, header=None)
            market_date = pd.to_datetime(str(sheet.iloc[1, 0]).split(":")[-1].strip(), format="%m/%d/%Y")
            matches = sheet.index[sheet.iloc[:, 1].astype(str).str.strip().eq("MISO System")]
            if len(matches) != 1:
                raise ValueError(f"Expected one MISO System row in {xls}, found {len(matches)}")
            first = matches[0] + 1
            for offset in range(24):
                price = sheet.iloc[first + offset, 1]
                if not isinstance(price, (int, float)):
                    raise TypeError(f"Non-numeric MISO System price in {xls}, offset {offset}: {price!r}")
                rows.append({"TIMESTAMP": market_date + pd.Timedelta(hours=offset), column: float(price)})
    return pd.DataFrame(rows).sort_values("TIMESTAMP", ignore_index=True)


def main() -> None:
    source = pd.read_csv(GEFCOM)
    source["TIMESTAMP"] = pd.to_datetime(source["TIMESTAMP"])
    source = source.loc[(source["ZONEID"].eq(1)) & source["TIMESTAMP"].between(START, END)].copy()

    solar_expected = source[["TIMESTAMP", "POWER"]].rename(columns={"POWER": "solar_power"}).reset_index(drop=True)
    weather_expected = source[["TIMESTAMP", *WEATHER]].reset_index(drop=True)
    solar_actual = pd.read_csv(BASE / "data" / "solar" / "solar-energy-generation.csv", parse_dates=["TIMESTAMP"])
    weather_actual = pd.read_csv(BASE / "data" / "weather" / "weather_data.csv", parse_dates=["TIMESTAMP"])
    assert_frame_equal(solar_actual, solar_expected, check_dtype=False, check_exact=True)
    assert_frame_equal(weather_actual, weather_expected, check_dtype=False, check_exact=True)

    da_expected = miso_system_prices("201*_da_pr_xls", "DA_LMP")
    rt_expected = miso_system_prices("201*_rt_pr_xls", "RT_LMP")
    da_actual = pd.read_csv(BASE / "data" / "prices" / "da_lmp_prices.csv", parse_dates=["TIMESTAMP"])
    rt_actual = pd.read_csv(BASE / "data" / "prices" / "rt_lmp_prices.csv", parse_dates=["TIMESTAMP"])
    # CSV decimal serialization can change the final binary floating-point bit.
    assert_frame_equal(da_actual, da_expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=0)
    assert_frame_equal(rt_actual, rt_expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=0)

    names = {"VAR78": "tclw", "VAR79": "tciw", "VAR134": "sp", "VAR157": "r", "VAR164": "tcc", "VAR165": "u10", "VAR166": "v10", "VAR167": "t2m", "VAR169": "ssrd", "VAR175": "strd", "VAR178": "tsr", "VAR228": "tp"}
    expected = solar_actual.rename(columns={"TIMESTAMP": "timestamp"})
    da_for_merge = da_actual.rename(columns={"TIMESTAMP": "timestamp", "DA_LMP": "da_price"})
    rt_for_merge = rt_actual.rename(columns={"TIMESTAMP": "timestamp", "RT_LMP": "rt_price"})
    expected = expected.merge(da_for_merge, on="timestamp", how="left").merge(rt_for_merge, on="timestamp", how="left")
    expected = expected.merge(weather_actual.rename(columns={"TIMESTAMP": "timestamp", **names}), on="timestamp", how="left")
    expected["rt_price"] = expected["rt_price"].ffill()
    expected = expected.sort_values("timestamp", ignore_index=True)
    actual = pd.read_csv(BASE / "data" / "merged_for_simulation.csv", parse_dates=["timestamp"])
    assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=0)

    missing_rt_before_fill = expected.loc[expected["timestamp"].gt(rt_actual["TIMESTAMP"].max()), "timestamp"]
    print("PASS: generated solar CSV equals Task 15 / Zone 1 / 2014-01..04 source slice.")
    print("PASS: generated weather CSV equals the same source slice's 12 VAR columns.")
    print(f"PASS: DA ({len(da_actual)}) and RT ({len(rt_actual)}) CSVs equal re-extraction from available MISO xls files.")
    print("PASS: merged_for_simulation.csv equals the current merge script's rules.")
    print(f"RT missing before ffill: {len(missing_rt_before_fill)} rows, {missing_rt_before_fill.min()} to {missing_rt_before_fill.max()}.")


if __name__ == "__main__":
    main()
