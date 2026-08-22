# Project memory

This file keeps durable, project-specific context for future work. Update it when a decision, convention, or verified result matters; do not store secrets or transient command output.

## Objective

- Reproduce and extend the study *Optimization-driven Uncertainty Forecasting* for solar-power day-ahead market commitments.
- Compare a forecasting-first baseline (AR/MLR) with an optimization-driven forecasting approach.

## Reference material

- `references/01_요약.txt`: paper summary and reported benchmark outcomes.
- `references/02_시뮬레이션가이드.txt`: modelling, metrics, and implementation outline.
- `references/03_데이터다운로드.txt`: data sources and intended directory layout.
- `references/04_시뮬코드.txt`: simulation code reference.
- `references/05_전처리스크립트.txt`: preprocessing reference.
- `references/06_시뮬레이션 코드 사용 가이드.txt`: simulation usage guide.
- `references/APEN_논문.pdf`: source paper.

## Intended data layout

```text
data/
  solar/solar-energy-generation.csv
  prices/day_ahead_prices.csv
  prices/real_time_prices.csv
  weather/weather_data.csv
```

## Core conventions from the references

- Solar data: use daylight observations (09:00–20:00) and normalize output by 30 MW to a 0–1 scale.
- Split: January–March for training; April for testing.
- Inputs: GEFCom2014 solar generation, MISO day-ahead/real-time prices, and optional ECMWF/ERA5 weather features.
- Evaluate both forecast accuracy (nRMSE) and operational performance (optimality gap).
- Test penalty-cost rates from 0% to 100% and multiple relative weights for operational loss versus forecast error.

## Verified source-data requirements (paper Section 5)

- Solar and weather must come together from the **GEFCom2014 Solar** track, not from separately downloaded ERA5 reanalysis. It contains normalized hourly solar generation (0–1) for three undisclosed Australian solar plants and 12 hourly day-ahead ECMWF weather forecasts.
- The GEFCom2014 Solar period spans 2012-04-01 through 2014-07-01 UTC. Its weather forecasts are issued at midnight UTC for the next 24 hours.
- The 12 weather variables are: `tclw`, `tciw`, `sp`, `r`, `tcc`, `10u`, `10v`, `2t`, `ssrd`, `strd`, `tsr`, and `tp`. The paper's selected MLR variables are `SSRD`, `TSR`, and hour.
- Do **not** divide the GEFCom solar target by 30: it is already normalized. The paper treats 30 MW only as an assumed capacity for profit calculations.
- MISO day-ahead and real-time prices are required separately. The paper does not disclose its exact MISO node or date range; it declares that data are available on request. Exact numerical replication therefore requires the authors' processed/merged data or their node and dates.

## Local GEFCom2014 archive check (2026-08-21)

- Archive extraction at `data/1-s2.0-S0169207016000133-mmc1/GEFCom2014 Data/` is the expected GEFCom2014 supplementary package.
- Relevant solar files are under `GEFCom2014-S_V2/Solar/`. `Task 15/train15.csv` contains 56,952 historical solar records (2012-04-01 01:00 through 2014-06-01 00:00) for zones 1–3. `predictors15.csv` contains the 12 ECMWF variables and the full released solar target values through 2014-07-01; `Solution to Task 15/Solution to Task 15.csv` supplies the 2,160 June-2014 outcomes.
- The bundled `GEFCom2014-P_V2/Price/` track is not the required MISO price input: it has a single `Zonal Price`, no real-time price, and a 2011–2013 period. Do not merge it with the Solar data for paper replication.

## Local MISO price data (2026-08-21)

- Downloaded publicly available MISO monthly pricing archives for January–April 2014 to `data/miso/raw/`: `201401`–`201404` × `da_pr_xls.zip` and `rt_pr_xls.zip`.
- The day-ahead archives contain one daily Excel report per day. Verified `20140101_da_pr.xls` has hourly LMPs for MISO System plus Illinois, Michigan, Minnesota, Indiana, Arkansas, Louisiana, and Texas hubs.
- These are valid public MISO day-ahead/real-time price candidates, but they are not yet proven identical to the paper's undisclosed node/period selection.

## Decisions log

| Date | Decision / verified fact | Reason |
| --- | --- | --- |
| 2026-08-21 | Created this project memory file. | Preserve key context derived from `references/`. |
| 2026-08-21 | The documented MISO daily URL pattern for `20140101_da_expost_lmp.csv` returns HTTP 404. | Use the MISO historical-report archive or obtain the original price extracts; do not assume current daily URLs retain 2014 files. |
| 2026-08-21 | Corrected the data plan from the source paper. | `references/03_데이터다운로드.txt`, its ERA5 notebook, and the preprocessing script do not fully match the paper's GEFCom2014 Solar source and normalization. |

## Open items

- Confirm the actual downloaded datasets, columns, time zone, and timestamp alignment before implementing the pipeline.
- Validate reference formulas and code against the PDF before treating benchmark metrics as reproduced.
