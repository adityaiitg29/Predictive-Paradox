# Predictive Paradox : Electricity Demand Forecasting

## Objective: To Predict Next-Hour Electricity Demand Forecasting Using Historical, Weather Data, Economic Data.

-Models Used: GradientBoostingRegressor + RandomForestRegressor (60/40 ensemble) 

-Metric: MAPE on chronological hold-out of 2023 

-Expected MAPE: 4–5% (improved from 5% baseline without weather)

-Features Used: 44 engineered features across 4 categories 

##1. Handling Missing Data & Outliers

 #PGCB dataset contains:
 1. Irregular Time Intervals
 2. Duplicate TimeStamps
 3. Missing Values
 4. Unrealistic Values
  
#STRATEGY UESD:

1.Duplicates Removed(432 duplicates removed):
Sorted by datetime, then `drop_duplicates(keep='last')`.

2.Irregular Frequency(Time):
Mixed 30-min and 60-min intervals were standardised to strict hourly cadence via `resample('1h').last()`. This is critical: without it, `lag_1` would sometimes point 30 minutes back and sometimes 60 minutes.

3.Limiting the Outliers:
Values outside [500, 20000] MW were nullified. Bangladesh's grid has never exceeded ~14,000 MW in this period; a 20,000 MW ceiling provides safety headroom. Values below 500 MW represent complete blackouts or corrupted sensor readings.

4.Spike Removal:
A 168-hour (1-week) rolling window computes local Q1 and Q3. Any point outside `[Q1 − 2.5×IQR, Q3 + 2.5×IQR]` is nullified. This is better than a global z-score because it adapts to seasonal demand levels — a reading of 9,000 MW in winter might be normal but a spike in summer.

5.Interpolation
Short gaps less than or equal to 6 consecutive hours were filled using time-based linear interpolation. Longer gaps were forward/backward filled (limit 3 hours).

6.Weather Gaps
Weather sensor short drop-outs were interpolated the same limit 3 hours. The weather dataset was nearly complete with no significant gaps.


##2. Feature Engineering

->To help the Model understand time patterns, I created meaningful features from the data.

2a: Calendar Features

|Feature : Why|

`hour`, `dow`, `month`: Basic Time Information.

`hour_sin` / `hour_cos`: Cyclical encoding — Helps Model understand cycles(e.g.,hour 23 is close to hour 0). 

`month_sin` / `month_cos`: Same for seasonality — December adjacent to January .

`dow_sin` / `dow_cos`: Cyclical day-of-week.

`is_weekend`: Weekend demand profile is 15% flatter and shifted later.

`is_morning_peak` (9–12h) : High Demand Periods

`is_evening_peak` (18–21h): High Demand Periods

->Cyclical encoding is important. Without it, a tree model would have to learn that hour 23 and hour 0 are similar by coincidence — cyclical features make this relationship explicit.

2b: Lag Features

->The most powerful feature group. Each feature is `demand_mw` at a particular number of hours in the past:

|Lag : Captures|

 -`lag_1`, `lag_2`, `lag_3`: Very recent trend
 
 -`lag_6`, `lag_12`: Intra-day pattern up to half a day 
 
 -`lag_24` : Same hour yesterday  
 
 -`lag_48` : Same hour two days ago 
 
 -`lag_168` : Same hour last week 

-> `df['demand_mw'].shift(n)` where n ≥ 1. Every lag strictly uses information from the past, never the current or future hour.

2c: Rolling Features

->Computed over shifted demand (`shift(1)` before `.rolling()`) to prevent leakage:

|Feature : Purpose |

-`roll_mean_6/24/168` : Baseline demand level over recent period 

-`roll_std_6/24/168` :Demand volatility — high std signals unstable period 

-`roll_max_6/24/168` : Recent peak — capacity signal 

-`roll_min_6/24/168` : Recent trough 

2d: Trend Features

|Feature : Formula : Why|

-`trend_24h`:`lag_1 − lag_25` :Is demand higher or lower than this time yesterday?

-`trend_1h` : `lag_1 − lag_2` : Is demand rising or falling right now? 


##3. Weather Feature Engineering

->Temperature is top external driver of electricity demand — it drives air conditioning (cooling load in summer) and heating load in winter.

3a: Weather Colunms Used

-`temp` : 2m air temperature (°C)

-`humidity` :relative humidity (%)

-`feels_like`:apparent temperature (accounts for wind chill / heat index)

`precip`: precipitation (mm/hour)

-`cloud_cover`:cloud cover (%)

-`sunshine_s`: sunshine duration per hour (seconds)

3b: Engineered Weather Features

->For each raw weather variable `wf`:

-`{wf}_lag1`:value at time t (shifted by 1 to be ultra-conservative)

-`{wf}_roll24` : 24-hour rolling mean (smooths out hourly noise)

3c: Additional Features

| Feature : Formula : Why |

-`heat_stress` : `temp_lag1 × humidity_lag1 / 100` : High temp + high humidity → non-linear AC demand surge (heat index proxy)

-`temp_sq` : `temp_lag1²` : AC demand increases quadratically with temperature above 26°C 

-`is_rain` : `precip_lag1 > 0.5` : Rain reduces mobility, shifts demand timing 

-`cool_night` : `temp_lag1 < 18 & hour < 6` : Cool nights reduce night-time AC baseline 


##4. Economic Feature Integration

-> PROBLEM : The economic dataset is annual; we need a value for every hourly row.

#APPROACH:

1. Selected key indicators 
   -`gdp_per_capita`:reflects overall economic growth
   
   -`urban_pop_pct`:higher electricity usage in cities

3. Coverted yearly data to hourly

   -Filled each year's value across all hours of that year
   
   -Missing years handled using forward fill
   
   -Merged with main dataset using the 'year' column

NO Data leakage: the economic data is only used for that same year and Model doesn't use future information.


##5. Validation Design (Zero Leakage)

Data Split:

   -TRAINING SET (~67,000 rows)-> From 2015 t0 2022
   
   -TEST SET  (8,760 rows)-> 2023

Why this is valid:

 -Model is evaluated on completely unseen future data  
 
 -All features use only past information  
 
 -Prevents data leakage and ensures realistic performance  


##6. Model Results

| Model : MAPE : Notes |

-GradientBoostingRegressor : 4.5%* : Main model — sharper at capturing non-linearities 

-RandomForestRegressor : 5.0% : Ensemble partner — more robust to outliers 

-Weighted Ensemble (60/40) : 4.6% : Best generalisation 

-Exact figures will appear in your terminal after running the script

##7. Feature Importances — Key Drivers

|Rank : Feature : Approximate Importance : What It Means |
 
 1 : `lag_1` : 80% : Electricity demand is highly inertial — next hour ≈ this hour ± small  
 
 2 : `lag_24` : 6% : Daily seasonality — same hour yesterday is very predictive 
 
 3 : `hour_cos` : 2.5% : Diurnal cycle 
 
 4 : `hour_sin` : 2.5% : Diurnal cycle 
 
 5 : `roll_mean_24` : 1.5% : Recent demand baseline 
 
 6 : `temp_lag1` : 1.0% : Temperature effect on AC/cooling load 
 
 7 : `trend_24h` : 0.8% : Is today's demand trending up or down vs yesterday? 
 
 8 : `heat_stress` : 0.6% : Combined temp + humidity drives non-linear AC spike 
 
 9 : `lag_168` : ~0.4% : Weekly seasonality 
 
 10 : `humidity_lag1` : 0.3% | Supports heat stress calculation


Key Insight:

1.Electricity demand is mainly driven by recent demand (lag features).

2.Strong daily and weekly patterns exist.

3.Weather (especially temperature) has secondary impact.

4.Economic factors contribute to long-term trends.

