import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt
from decimal import Decimal, getcontext

# output control
output_all = False
output_flags = {
    'population': False,
    'gdp': False,
    'technology': False,
    'emissions': False,
    'temperature': False,
    'scenario1': False,
    'scenario2': False,
    'scenario3': False,
}
if len(sys.argv) > 1:
    if '--help' in sys.argv or '-h' in sys.argv:
        print(f"Usage: ./{sys.argv[0]} [-h or --help] [population | gdp | technology | emissions | temperature | scenario1 | scenario2 | scenario3]")
        print("\t-h / --help outputs this menu")
        print()
        print("\tadding population, gdp, technology, etc. to the arguments will give relevant graphs for those")
        print(f"\tE.g. running \"./{sys.argv[0]} scenario1 technology scenario2\" will give the graphs for technology, and scenarios 1 and 2")
        print()
        print("\talternatively, giving no arguments will give all the graphs")
        sys.exit(0)

    for arg in sys.argv[1:]:
        if arg in output_flags:
            output_flags[arg] = True
        else:
            print("Couldn't recognise argument", arg)
            print("Use one of these: population | gdp | technology | emissions | temperature | scenario1 | scenario2 | scenario3")
            sys.exit(1)
else:
    output_all = True


getcontext().prec = 100

url_pop = 'https://raw.githubusercontent.com/janzika/MATH3041/refs/heads/main/data/population/population.csv'
uri_gdp = 'data/global-average-gdp-per-capita.csv'
uri_kaya = 'data/kaya-identity-co2.csv'
uri_temp = 'data/temperature-anomaly.csv'
uri_co2 = 'data/annual-co2-emissions-per-country.csv'
POP_data = pd.read_csv(url_pop)

GDPPC_data = pd.read_csv(uri_gdp).infer_objects()
num_cols = GDPPC_data.select_dtypes(include='number').columns
GDPPC_data[num_cols] = GDPPC_data[num_cols].interpolate()

KAYA_data = pd.read_csv(uri_kaya).infer_objects()
num_cols = KAYA_data.select_dtypes(include='number').columns
KAYA_data[num_cols] = KAYA_data[num_cols].interpolate()

TEMP_data = pd.read_csv(uri_temp)
CO2_data = pd.read_csv(uri_co2)

brackets = [
    ('World', 'OWID_WRL', 'tab:green'),
    # ('Europe', 'OWID_EUR', 'tab:cyan'),
    # ('India', 'IND', 'tab:green'),
    # ('Low-income countries', 'OWID_LIC', 'tab:red'),
    # ('Lower-middle-income countries', 'OWID_LMC', 'tab:orange'),
    # ('Upper-middle-income countries', 'OWID_UMC', 'tab:blue'),
    # ('High-income countries', 'OWID_HIC', 'tab:purple')
]

TEMP_YEAR_INIT = 1850
YEAR_INIT = 1950
CO2_YEAR_INIT = 1750
KAYA_YEAR_INIT = 1965
YEAR_END = 2110
MODEL_PRE_YEARS = 20
WORLD_MAX_POP = 12_000_000_000
SENSITIVITY = 1 # removal sensitivity: set to 1 for perfect removal (100%), or a fraction to test sensitivity if targets are off by X%

# Actual population, GDP per capita -> model P, Y
pop_years = []
pop_levels = []
gdppc_years = []
gdppc_nums = []
kaya_years = []
temp_years = []
co2_years = []
energy_int_levels = []
carbon_int_levels = []
carbon_emi_levels = []
temp_levels = []
co2_levels = []
for _, code, _ in brackets:
    pop_points = np.where((POP_data['Code'] == code) & (POP_data['Year'] >= YEAR_INIT))[0]
    pop_years.append(POP_data['Year'][pop_points[0]:pop_points[-1] + 1])
    pop_levels.append(POP_data['all years'][pop_points[0]:pop_points[-1] + 1])

    gdppc_points = np.where((GDPPC_data['Code'] == code) & (GDPPC_data['Year'] >= YEAR_INIT))[0]
    gdppc_years.append(GDPPC_data['Year'][gdppc_points[0]:gdppc_points[-1] + 1])
    gdppc_nums.append(GDPPC_data['GDP per capita'][gdppc_points[0]:gdppc_points[-1] + 1])

    kaya_points = np.where((KAYA_data['Code'] == code) & (KAYA_data['Year'] >= KAYA_YEAR_INIT))[0]
    kaya_years.append(KAYA_data['Year'][kaya_points[0]:kaya_points[-1] + 1])
    energy_int_levels.append(KAYA_data['Energy intensity (Energy / GDP)'][kaya_points[0]:kaya_points[-1] + 1])
    carbon_int_levels.append(KAYA_data['Carbon intensity (CO₂ / energy)'][kaya_points[0]:kaya_points[-1] + 1])
    carbon_emi_levels.append(KAYA_data['CO₂ emissions'][kaya_points[0]:kaya_points[-1] + 1])

    temp_points = np.where((TEMP_data['Code'] == code) & (TEMP_data['Year'] >= TEMP_YEAR_INIT))[0]
    temp_years = TEMP_data['Year'][temp_points[0]:temp_points[-1] + 1]
    temp_levels = TEMP_data['Average'][temp_points[0]:temp_points[-1] + 1]

    co2_points = np.where((CO2_data['Code'] == code) & (CO2_data['Year'] >= CO2_YEAR_INIT))[0]
    co2_years = CO2_data['Year'][co2_points[0]:co2_points[-1] + 1]
    co2_levels = CO2_data['Annual CO₂ emissions'][co2_points[0]:co2_points[-1] + 1]

# Actual population growth -> model P'
pop_changes = []
pop_change_years = []
for entity_pops, entity_pop_years in zip(pop_levels, pop_years):
    pop_changes.append(entity_pops.diff().iloc[1:])
    pop_change_years.append(entity_pop_years.iloc[1:])

def logistic(init, final, r, t, t0):
    return final / (1 + (final / init - 1) * exp(-r * (t - t0)))



"""""""""""""""""""""""""""""""""""""""

               Graphing              

"""""""""""""""""""""""""""""""""""""""

def population(_tp):
    return logistic(p_init, WORLD_MAX_POP, r_pop, _tp, YEAR_INIT) - logistic(250_000_000, 2_500_000_000, 0.04, _tp, YEAR_INIT + 100)

print('*** Population (P)')
plt.title('Population over Time')
plt.xlabel('Year')
plt.ylabel('Population (Billions)')
plt.grid()
model_logis = []
for i, (years, levels) in enumerate(zip(pop_years, pop_levels)):
    plt.plot(years, levels / 1_000_000_000, label=brackets[i][0] + ' Actual', marker='.', markerfacecolor='None', linestyle='None', color=brackets[i][2])

    p_init = levels.iloc[0]
    p_final = levels.iloc[-1]
    # print(p_init, p_final, end=": ")

    # Find the median
    total_diff = p_final - p_init
    median_target = total_diff / 2 + p_init
    previous_value = p_init
    current_value = p_init
    median_year = 0
    for level in levels.to_list():
        previous_value = current_value
        current_value = level
        if current_value >= median_target:
            break
        median_year += 1
    ## Whichever year/value is closer is the median
    if current_value == median_target or abs(current_value - median_target) < abs(previous_value - median_target):
        median = current_value
    else:
        median = previous_value
        median_year -= 1

    r = -1/(median_year) * (log(WORLD_MAX_POP / median - 1) - log(WORLD_MAX_POP / p_init - 1))
    print(median, median_year, r)
    r_pop = r

    modelling_years = range(YEAR_INIT - MODEL_PRE_YEARS, YEAR_END)
    # model_pops = [logistic(p_init, WORLD_MAX_POP, r, y, YEAR_INIT) for y in modelling_years]
    model_pops = [population(y) for y in modelling_years]
    print(f'y = {WORLD_MAX_POP}/(1 + ({WORLD_MAX_POP}/{p_init} - 1)*e^{-r}(t - {YEAR_INIT}))')

    plt.plot(modelling_years, [p / 1_000_000_000 for p in model_pops], linestyle='--', color=brackets[i][2], label='Model')

    # residuals = [level - model_pop for level, model_pop in zip(levels, model_pops[MODEL_PRE_YEARS:])]
    # ssr = sum(r ** 2 for r in residuals)  # Is this called SSR? Square Sum of Residuals?
    # total_sum_squared = sum(l ** 2 for l in levels)
    # print(brackets[i][0], 'model:', '1 - SSR/SST =', 1 - ssr / total_sum_squared)

    error = [model_pop/level - 1 for level, model_pop in zip(levels, model_pops[MODEL_PRE_YEARS:])]
    mean = sum(error) / len(error)

    error_mean_diffs = [e - mean for e in error]
    variance = sum(e * e for e in error_mean_diffs) / len(error)
    print(f'{brackets[i][0]} model: {years.iloc[0]}–{years.iloc[-1]}')
    print('\t% error mean  =', mean * 100)
    print('\t% error stdev =', sqrt(variance) * 100)


plt.legend()
if not (output_flags['population'] or output_all):
    plt.clf()
else:
    plt.show()







print('*** GDP per Capita (Y)')
plt.subplot(1, 2, 1)
plt.title('GDP per Capita – Logarithmic')
plt.xlabel('Year')
plt.ylabel('ln(GDP - 1000)')
plt.grid()
for i, (years, nums) in enumerate(zip(gdppc_years, gdppc_nums)):
    nums = [n - 1000 for n in nums]
    log_gdppc = [log(n) for n in nums]

    y_mean = sum(log_gdppc) / len(log_gdppc)
    x_mean = sum(years) / len(years)

    x_diffs = [x - x_mean for x in years]
    y_diffs = [y - y_mean for y in log_gdppc]
    top_term = sum(x_diff * y_diff for x_diff, y_diff in zip(x_diffs, y_diffs))
    x_stdev = sqrt(sum([xd * xd for xd in x_diffs]))
    y_stdev = sqrt(sum([yd * yd for yd in y_diffs]))
    r = top_term / (x_stdev * y_stdev)

    plt.plot(years, log_gdppc, label=brackets[i][0] + f' Actual (r = {r:.2})', marker='.', markerfacecolor='None', linestyle='None', color=brackets[i][2])

    xy_all = [x * y for x, y in zip(years, log_gdppc)]
    xy_mean = sum(xy_all) / len(xy_all)
    x2_all = [x * x for x in years]
    x2_mean = sum(x2_all) / len(x2_all)

    m = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean ** 2)
    b = y_mean - m * x_mean

    modelling_years = range(YEAR_INIT - MODEL_PRE_YEARS, YEAR_END)
    plt.plot(modelling_years, [m * y + b for y in modelling_years], label='Model', linestyle='--', color=brackets[i][2])

plt.legend()

plt.subplot(1, 2, 2)
plt.title('GDP per Capita')
plt.xlabel('Year')
plt.ylabel('GDP per Capita (international-$, USD)')
plt.grid()
for i, (years, nums) in enumerate(zip(gdppc_years, gdppc_nums)):
    nums = [n for n in nums]
    log_gdppc = [log(n - 1000) for n in nums]
    plt.plot(years, nums, label=brackets[i][0], marker='.', markerfacecolor='None', linestyle='None', color=brackets[i][2])

    y_mean = sum(log_gdppc) / len(log_gdppc)
    x_mean = sum(years) / len(years)

    xy_all = [x * y for x, y in zip(years, log_gdppc)]
    xy_mean = sum(xy_all) / len(xy_all)
    x2_all = [x * x for x in years]
    x2_mean = sum(x2_all) / len(x2_all)

    m = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean ** 2)
    b = y_mean - m * x_mean

    m_gdppc = m
    b_gdppc = b

    # print(m, b)

    modelling_years = range(YEAR_INIT - MODEL_PRE_YEARS, YEAR_END)
    model_gdppc = [exp(m * y + b) + 1000 for y in modelling_years]
    print(f'y = 1000 + {exp(b)}*e^{m}t')

    # for j in range(1800, 1991, 10):
    #     print(j, exp(m * j + b) + 1000)

    # residuals = [level - logistic for level, logistic in zip(nums, model_gdppc[MODEL_PRE_YEARS:])]
    # ssr = sum(r ** 2 for r in residuals)  # Is this called SSR? Square Sum of Residuals?
    # total_sum_squared = sum(n ** 2 for n in nums)
    # print(brackets[i][0], 'model:', '1 - SSR/SST =', 1 - ssr / total_sum_squared)

    error = [model_gdppc/num - 1 for num, model_gdppc in zip(nums, model_gdppc[MODEL_PRE_YEARS:])]
    mean = sum(error) / len(error)

    error_mean_diffs = [e - mean for e in error]
    variance = sum(e * e for e in error_mean_diffs) / len(error)
    print(f'{brackets[i][0]} model: {years.iloc[0]}–{years.iloc[-1]}')
    print('\t% error mean  =', mean * 100)
    print('\t% error stdev =', sqrt(variance) * 100)

    # ln(y) = -bt + ln(C) -> y = exp(-bt + ln(C)) = C exp(-bt)
    plt.plot(modelling_years, model_gdppc, linestyle='--', color=brackets[i][2])

if not (output_flags['gdp'] or output_all):
    plt.clf()
else:
    plt.show()
def gdppc(_ty):
    return exp(m_gdppc * _ty + b_gdppc) + 1000






print('*** Carbon intensity × energy intensity (∝ D)')
plt.subplot(1, 2, 1)
plt.title('Carbon intensity × energy intensity – Logarithmic')
plt.xlabel('Year')
plt.ylabel('ln(The Product)')
plt.grid()
for i, (years, energy_ints, carbon_ints) in enumerate(zip(kaya_years, energy_int_levels, carbon_int_levels)):
    product_intensities = [e * c for e, c in zip(energy_ints, carbon_ints)]
    log_prod_ints = [log(p) for p in product_intensities]

    y_mean = sum(log_prod_ints) / len(log_prod_ints)
    x_mean = sum(years) / len(years)

    x_diffs = [x - x_mean for x in years]
    y_diffs = [y - y_mean for y in log_prod_ints]
    top_term = sum(x_diff * y_diff for x_diff, y_diff in zip(x_diffs, y_diffs))
    x_stdev = sqrt(sum([xd * xd for xd in x_diffs]))
    y_stdev = sqrt(sum([yd * yd for yd in y_diffs]))
    r = top_term / (x_stdev * y_stdev)

    plt.plot(years, log_prod_ints, label=brackets[i][0] + f' Actual (r = {r:.2})', marker='.', markerfacecolor='None', linestyle='None', color=brackets[i][2])

    xy_all = [x * y for x, y in zip(years, log_prod_ints)]
    xy_mean = sum(xy_all) / len(xy_all)
    x2_all = [x * x for x in years]
    x2_mean = sum(x2_all) / len(x2_all)

    m = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean ** 2)
    b = y_mean - m * x_mean

    modelling_years = range(KAYA_YEAR_INIT - MODEL_PRE_YEARS, YEAR_END)
    plt.plot(modelling_years, [m * y + b for y in modelling_years], label='Model', linestyle='--', color=brackets[i][2])

plt.legend()

plt.subplot(1, 2, 2)
plt.title('Carbon intensity × energy intensity')
plt.xlabel('Year')
plt.ylabel('The product (GtCO2 / GDP)')
plt.grid()
for i, (years, energy_ints, carbon_ints) in enumerate(zip(kaya_years, energy_int_levels, carbon_int_levels)):
    product_intensities = [e * c for e, c in zip(energy_ints, carbon_ints)]
    log_prod_ints = [log(p) for p in product_intensities]
    plt.plot(years, product_intensities, label=brackets[i][0], marker='.', markerfacecolor='None', linestyle='None', color=brackets[i][2])

    y_mean = sum(log_prod_ints) / len(log_prod_ints)
    x_mean = sum(years) / len(years)

    xy_all = [x * y for x, y in zip(years, log_prod_ints)]
    xy_mean = sum(xy_all) / len(xy_all)
    x2_all = [x * x for x in years]
    x2_mean = sum(x2_all) / len(x2_all)

    m = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean ** 2)
    b = y_mean - m * x_mean

    m_dinv = m
    b_dinv = b

    # print(m, b)

    modelling_years = range(KAYA_YEAR_INIT - MODEL_PRE_YEARS, YEAR_END)
    model_prod_ints = [exp(m * y + b) for y in modelling_years]
    print(f'y = {exp(b)}*e^{m}t')

    # residuals = [level - prod_int for level, prod_int in zip(product_intensities, model_prod_ints[MODEL_PRE_YEARS:])]
    # ssr = sum(r ** 2 for r in residuals)  # Is this called SSR? Square Sum of Residuals?
    # total_sum_squared = sum(p ** 2 for p in product_intensities)
    # print(brackets[i][0], 'model:', '1 - SSR/SST =', 1 - ssr / total_sum_squared)

    error = [model_prod_int/prod_int - 1 for prod_int, model_prod_int in zip(product_intensities, model_prod_ints[MODEL_PRE_YEARS:])]
    mean = sum(error) / len(error)

    error_mean_diffs = [e - mean for e in error]
    variance = sum(e * e for e in error_mean_diffs) / len(error)
    print(f'{brackets[i][0]} model: {years.iloc[0]}–{years.iloc[-1]}')
    print('\t% error mean  =', mean * 100)
    print('\t% error stdev =', sqrt(variance) * 100)

    # ln(y) = -bt + ln(C) -> y = exp(-bt + ln(C)) = C exp(-bt)
    plt.plot(modelling_years, model_prod_ints, linestyle='--', color=brackets[i][2])
if not (output_flags['technology'] or output_all):
    plt.clf()
else:
    plt.show()
def d_inv(_td):
    """
    Called 'D inverse' because technological inefficiency was
    previously considered the inverse of technological efficiency.
    """
    return exp(m_dinv * _td + b_dinv)






print('*** Annual Emissions (E = kPYD^{-1})')
plt.title('Annual CO₂ Emissions')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions (10 Gt)')
plt.grid()
model_years = list(range(KAYA_YEAR_INIT - MODEL_PRE_YEARS, YEAR_END))

plt.plot(kaya_years[0], carbon_emi_levels[0] / 10_000_000_000, label='World Actual', marker='.', markerfacecolor='None', linestyle='None', color='tab:green')
# plt.plot(co2_years, co2_levels / 10_000_000_000, label='World Actual', marker='.', markerfacecolor='None', linestyle='None', color='tab:green')

# Linear scan for best value for k
## Optimise for lowest absolute mean
kd = 1_300_000
mean = 1
model_emis = []
while True:
    model_emis = [population(y) * gdppc(y) * d_inv(y) / kd for y in model_years]

    error = [model_emi/actual_emi - 1 for actual_emi, model_emi in zip(carbon_emi_levels[0], model_emis[MODEL_PRE_YEARS:])]
    # error = [model_emi/actual_emi - 1 for actual_emi, model_emi in zip(co2_levels, model_emis[MODEL_PRE_YEARS:])]
    mean = sum(error) / len(error)

    if mean < 0:
        break

    kd += 1

k = Decimal(1) / kd
plt.plot(model_years, [e / 10_000_000_000 for e in model_emis], label=f'Model (k = 1 / {kd})', linestyle='--', color='tab:green')
k_carbon = k

error_mean_diffs = [e - mean for e in error]
variance = sum(e * e for e in error_mean_diffs) / len(error)

print(f'k = {k:.15f}')
print(f'World model: {kaya_years[0].iloc[0]}–{kaya_years[0].iloc[-1]}')
# print(f'World model: {co2_years.iloc[0]}–{co2_years.iloc[-1]}')
print('\t% error mean  =', mean * 100)
print('\t% error stdev =', sqrt(variance) * 100)

plt.legend()

if not (output_flags['emissions'] or output_all):
    plt.clf()
else:
    plt.show()

def carbon_emission(_te):
    return population(_te) * gdppc(_te) * d_inv(_te) * float(k_carbon)







print(r'*** Temperature (T = k * \int_1850^t C(s)ds)')
plt.title('Temperature Anomaly over Time')
plt.xlabel('Year')
plt.ylabel('Temperature difference since 1850 (°C)')
plt.grid()
model_years = list(range(TEMP_YEAR_INIT, YEAR_END))

actual_temps = [t - temp_levels.iloc[0] for t in temp_levels]
plt.plot(temp_years, actual_temps, label='World Actual', marker='.', markerfacecolor='None', linestyle='None', color='tab:green')

k_temp = Decimal(1) / Decimal(1.5 * 10 ** 12)   # temp

model_temps = [0.]
kd = 1.5 * 10 ** 12
mean = -1
while True:
    model_temps = [0.]
    for y in model_years[1:]:
        model_temps.append(model_temps[-1] + carbon_emission(y))

    model_temps = [float(Decimal(t)) / kd for t in model_temps]
    error = [model - actual for actual, model in zip(actual_temps, model_temps)]
    mean = sum(error) / len(error)

    if mean > 0:
        break
    
    kd -= 10 ** 8

actual_model_temps = model_temps[::1]
k_temp = Decimal(1) / Decimal(kd)
plt.plot(model_years, model_temps, label=f'Model (k = 1 / {round(kd)})', linestyle='--', color='tab:green')

variance = sum(e * e for e in error) / len(error)

print(f'k = {k_temp:.15f}')
print(f'World model: {temp_years.iloc[0]}–{temp_years.iloc[-1]}')
print('\terror mean  =', mean)
print('\terror stdev =', sqrt(variance))

plt.legend()
if not (output_flags['temperature'] or output_all):
    plt.clf()
else:
    plt.show()






"""""""""""""""""""""""""""""""""""""""

          Modelling Scenarios              

"""""""""""""""""""""""""""""""""""""""
MAGIC_YEAR = 2027

def gdppc_alt(_ty, scenario_num=1):
    if _ty < MAGIC_YEAR:
        return gdppc(_ty)
    multiplier = 1
    if scenario_num == 1:
        multiplier = 0.7
    elif scenario_num == 2:
        multiplier = 0.6
    elif scenario_num == 3:
        multiplier = 0.5

    m_gdppc_alt = multiplier * m_gdppc
    return exp(m_gdppc * MAGIC_YEAR + b_gdppc) * exp(m_gdppc_alt * (_ty - MAGIC_YEAR)) + 1000

def d_inv_alt(_td, scenario_num=1):
    if _td < MAGIC_YEAR:
        return d_inv(_td)
    multiplier = 1
    if scenario_num == 1:
        multiplier = 1.5
    elif scenario_num == 2:
        multiplier = 2
    elif scenario_num == 3:
        multiplier = 2.5
    m_dinv_alt = multiplier * m_dinv
    return exp(m_dinv * MAGIC_YEAR + b_dinv) * exp(m_dinv_alt * (_td - MAGIC_YEAR))


modelling_years = list(range(2015, YEAR_END))
extended_modelling_years = list(range(2015, YEAR_END + 200))
# plt.plot(modelling_years, [gdppc_alt(y) for y in modelling_years], marker='.', linestyle='None')
# plt.show()

# plt.plot(modelling_years, [d_inv_alt(y) for y in modelling_years], marker='.', linestyle='None')
# plt.show()

def scenario_emissions(year, scenario_num=1):
    return population(year) * gdppc_alt(year, scenario_num) * d_inv_alt(year, scenario_num) * float(k_carbon)


# Scenario 1
def scenario_1_emissions(year):
    return scenario_emissions(year, 1)

def removal_scenario_1_proportion(year):
    if year < MAGIC_YEAR:
        return 0
    return 2/(1 + exp(-0.007 * (year - MAGIC_YEAR))) - 1

def removal_scenario_1(year):
    if year < MAGIC_YEAR:
        return 0
    return min(SENSITIVITY * 5 * 10**9, 1/50 * 3 * 10**9 * (year - MAGIC_YEAR))
    # return scenario_1_emissions(year) * removal_scenario_1_proportion(year)

plt.subplot(1, 2, 1)
plt.title('Scenario 1 – Active Carbon Removal')
plt.xlabel('Year')
plt.ylabel('Carbon Removal (Gtonnes)')
plt.plot(modelling_years, [removal_scenario_1(y) / 1_000_000_000 for y in modelling_years], marker='.', linestyle='None')
plt.grid()
# if not (output_flags['scenario1'] or output_all):
#     plt.clf()
# else:
#     plt.show()

model_temps = [0.]
for y in modelling_years[1:]:
    model_temps.append(model_temps[-1] + (scenario_1_emissions(y) - removal_scenario_1(y)))
model_temps = [float(k_temp) * m for m in model_temps]
model_temps = [m + actual_model_temps[modelling_years[0] - TEMP_YEAR_INIT] for m in model_temps]

plt.subplot(1, 2, 2)
plt.title('Scenario 1 – Trajectory')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (°C)')
plt.grid()
plt.plot(modelling_years, model_temps, marker='.', linestyle='None')
if not (output_flags['scenario1'] or output_all):
    plt.clf()
else:
    plt.show()


model_temps = [0.]
for y in extended_modelling_years[1:]:
    model_temps.append(model_temps[-1] + (scenario_1_emissions(y) - removal_scenario_1(y)))
model_temps = [float(k_temp) * m for m in model_temps]
model_temps = [m + actual_model_temps[extended_modelling_years[0] - TEMP_YEAR_INIT] for m in model_temps]

plt.title('Scenario 1 – Trajectory (Long term)')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (°C)')
plt.grid()
plt.plot(extended_modelling_years, model_temps, marker='.', linestyle='None')
if not (output_flags['scenario1'] or output_all):
    plt.clf()
else:
    plt.show()







# Scenario 2
def scenario_2_emissions(year):
    return scenario_emissions(year, 2)

def removal_scenario_2_proportion(year):
    if year < MAGIC_YEAR:
        return 0
    return 2/(1 + exp(-0.07 * (year - MAGIC_YEAR))) - 1

def removal_scenario_2(year):
    return SENSITIVITY * scenario_2_emissions(year) * removal_scenario_2_proportion(year)

plt.subplot(1, 2, 1)
plt.title('Scenario 2 – Active Carbon Removal')
plt.xlabel('Year')
plt.ylabel('Carbon Removal (Gtonnes)')
plt.plot(modelling_years, [removal_scenario_2(y) / 1_000_000_000 for y in modelling_years], marker='.', linestyle='None')
plt.grid()
# if not (output_flags['scenario2'] or output_all):
#     plt.clf()
# else:
#     plt.show()

model_temps = [0.]
for y in modelling_years[1:]:
    model_temps.append(model_temps[-1] + (scenario_2_emissions(y) - removal_scenario_2(y)))
model_temps = [float(k_temp) * m for m in model_temps]
model_temps = [m + actual_model_temps[modelling_years[0] - TEMP_YEAR_INIT] for m in model_temps]

plt.subplot(1, 2, 2)
plt.title('Scenario 2 – Trajectory')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (°C)')
plt.grid()
plt.plot(modelling_years, model_temps, marker='.', linestyle='None')
if not (output_flags['scenario2'] or output_all):
    plt.clf()
else:
    plt.show()





# Scenario 3
def scenario_3_emissions(year):
    return scenario_emissions(year, 3)

def removal_scenario_3(year):
    if year < MAGIC_YEAR:
        return 0
    if year > 2104:
        return scenario_emissions(year, 3)
    # Linear growth of CO2 removal year on year after MAGIC_YEAR
    # Aim for 26 Gtonnes in 20 years, and continue to grow linearly
    return SENSITIVITY * 1/20 * 26 * 10**9 * (year - MAGIC_YEAR)


plt.subplot(1, 2, 1)
plt.title('Scenario 3 – Active Carbon Removal')
plt.xlabel('Year')
plt.ylabel('Carbon Removal (Gtonnes)')
plt.plot(modelling_years, [removal_scenario_3(y) / 1_000_000_000 for y in modelling_years], marker='.', linestyle='None')
plt.grid()
# if not (output_flags['scenario3'] or output_all):
#     plt.clf()
# else:
#     plt.show()


model_temps = [0.]
for y in modelling_years[1:]:
    model_temps.append(model_temps[-1] + (scenario_3_emissions(y) - removal_scenario_3(y)))
model_temps = [float(k_temp) * m for m in model_temps]
model_temps = [m + actual_model_temps[modelling_years[0] - TEMP_YEAR_INIT] for m in model_temps]

plt.subplot(1, 2, 2)
plt.title('Scenario 3 – Trajectory')
plt.xlabel('Year')
plt.ylabel('Temperature anomaly (°C)')
plt.grid()
plt.plot(modelling_years, model_temps, marker='.', linestyle='None')
if not (output_flags['scenario3'] or output_all):
    plt.clf()
else:
    plt.show()

# Scenario 3 sea level projections
# Rahmstorf's equation: dH/dt = a * (T - T_0) => (T - T_0) is the surface temperature anomaly already obtained.
# a = 3.4 mm / year obtained from the original paper.
sea_levels = []
last_state = 0
for m in model_temps:
    new_state = 3.4 * m + last_state
    sea_levels.append(new_state)
    last_state = new_state
print("Scenario 3 final sea level:", sea_levels[-1], "mm.")


plt.title('Scenario 3 – Sea levels')
plt.xlabel('Year')
plt.ylabel('Sea levels since 2015 (mm)')
plt.grid()
plt.plot(modelling_years, sea_levels, marker='.', linestyle='None')
if not (output_flags['scenario3'] or output_all):
    plt.clf()
else:
    plt.show()
