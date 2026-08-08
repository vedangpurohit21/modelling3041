# Climate & Emissions Modelling (MATH3041)

A single-file Python model that fits historical population, GDP, and carbon-intensity
trends to simple closed-form curves, chains them together through the **Kaya identity**
to reproduce historical CO₂ emissions and global temperature anomaly, and then projects
three future emissions/temperature scenarios under different technology-improvement and
active carbon-removal assumptions.

## How it works

The model builds up world-level projections in stages, each fitted independently to
historical data and then composed:

1. **Population `P(t)`** — a logistic growth curve fit to historical population, with a
   second logistic term subtracted to model the growth slowdown/decline expected later
   this century (population is capped at `WORLD_MAX_POP = 12` billion).
2. **GDP per capita `Y(t)`** — an exponential fit (linear regression on `ln(GDP - 1000)`
   vs. year).
3. **Technology term `D⁻¹(t)`** — an exponential fit to the product of *energy intensity*
   (energy/GDP) and *carbon intensity* (CO₂/energy) from the Kaya identity dataset. This
   captures the combined rate at which the economy is decarbonising per unit of output.
4. **Annual emissions `E(t) = k · P(t) · Y(t) · D⁻¹(t)`** — the Kaya identity, with a
   constant `k` calibrated against historical annual CO₂ emissions.
5. **Temperature anomaly `T(t)`** — modelled as a scaled running integral (cumulative
   sum) of annual emissions since 1850, with the scale constant `k_temp` calibrated
   against the observed temperature anomaly record.

Each stage prints its fitted parameters and goodness-of-fit (% error mean/stdev, or
correlation `r`) to stdout, and (optionally) plots actual vs. modelled curves.

### Scenarios

From `MAGIC_YEAR = 2027` onward, three scenarios apply stricter multipliers to the GDP
growth rate and technology (decarbonisation) rate, paired with different active carbon
removal trajectories, to project temperature outcomes to 2110:

| Scenario | GDP growth multiplier | Decarbonisation rate multiplier | Carbon removal |
|----------|:---:|:---:|---|
| 1 | 0.7× | 1.5× | Ramps linearly to a 5 Gt/yr cap |
| 2 | 0.6× | 2.0× | Logistic proportion of that year's emissions |
| 3 | 0.5× | 2.5× | Ramps linearly to 30 Gt in 20 years, then matches emissions after 2095 (net zero) |

## Data

All data is in `data/`, sourced from [Our World in Data](https://ourworldindata.org/)
(OWID) datasets, except population which is fetched at runtime from the `janzika/MATH3041`
GitHub repository. Model years span 1750–2110 depending on the stage (each dataset starts
at a different year; see `*_YEAR_INIT` constants in the script).

| File | Contents |
|---|---|
| `data/global-average-gdp-per-capita.csv` | World GDP per capita by year (from 1950) |
| `data/kaya-identity-co2.csv` | Per-country/world Kaya identity components: CO₂ emissions, energy intensity, GDP per capita, population, carbon intensity (from 1965) |
| `data/temperature-anomaly.csv` | Global surface temperature anomaly by year, with bounds (from 1850) |
| `data/annual-co2-emissions-per-country.csv` | Annual CO₂ emissions by country (from 1750) |

The script currently only processes the `World` entity (`OWID_WRL`); the `brackets` list
in `model_ravens.py` has other regions/income groups available but commented out.

## Requirements

```
pip install -r requirements.txt
```

Requires internet access on first run, since population data is fetched from a remote URL.

## Usage

```
python3 model_ravens.py [-h | --help] [population | gdp | technology | emissions | temperature | scenario1 | scenario2 | scenario3]
```

- No arguments: runs everything and shows every plot.
- One or more keywords: only shows plots for those stages (fit diagnostics are always
  printed to stdout regardless).

Example:

```
python3 model_ravens.py scenario1 scenario3
```

## Notes

- `getcontext().prec = 100` and `Decimal` are used for the emissions/temperature
  calibration constants (`k`, `k_temp`) to avoid floating-point drift when accumulating
  over ~350 years of emissions.
- The emissions calibration constant `k` is currently fixed at a previously found value
  (`kd = 1298384`) rather than being re-searched on each run (see the `break` in the
  emissions section of the script).
- The temperature calibration loop searches for `k_temp` by decrementing until the model
  mean error crosses zero.
