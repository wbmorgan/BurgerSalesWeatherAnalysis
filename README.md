# Weather impact on Sales and Forecasting with PostgreSQL and Python 

## Project Overview
In a personal endeavour to learn about time series analysis and forecasting, and using Python on real-life data I decided to do a project on Five Guys Sales data at my local branch. 

The managers were struggling to predict sales based on 7-day lag, and due to this were making constant losses due to over/undershooting in inventory orders, so created this project to see if there was a way to predict this more accurately.

After talking to staff and management, I thought the main factors of how busy the store got were as follows:

- Weather Patterns (Temperature, Precipitation, Wind, etc)
- Popular local events (Football, Rugby, Concerts)
- Public Holidays

This analysis mostly asses the influence of weather on Sales.

## Data Sources
I created a PostgreSQL database to store and access the data. I used Excel to store and digitalise two years' worth of analogue sales data and connected to weather API's (WorldWeatherOnline, MetOffice Datapoint) to get local historical weather data. 

I used the sales data and analysed it with various weather conditions and aimed to use this, along with the effect of Local events and holidays to analyse the effect and eventually use it the predict future Sales.
## Methodology
-  Upload data from Excel sheets to PostgreSQL and import API to the database using Python. 
- Data extraction from PostgreSQL
- Data cleaning and processing
- Sales analysis with reference to weather
- Forecasting using RandomForest Regression using main factors in feature engineering 

## Results
Discuss the key results of your analysis. You can summarize the findings from the graphs and analysis here.

I found that weather had some influence on sales but only on certain metrics.

## Correlation Table

| Metric            | Correlation with 'total_day' |
|-------------------|-----------------------------:|
| prcp (Precipitation)     | 0.1588                   |
| wpgt (Wind Gust)         | -0.0613                  |
| wspd (Wind Speed)        | -0.0427                  |
| tavg (Average Temperature)| -0.0646                  |

This was using Met Office Daily data, but I wanted to get more accurate data in a shorter time frame than 24h so I used an API to check weather conditions with both Day and Night Sales.

I drew a map to use weather conditions by weather severity (1 = Clear, 9 = Storm/Thunderstorm) and looked at the weighted averages (to offset the effect of count)

## Weighted Statistics by Severity

| Severity | Weighted Mean Open | Weighted Median Open | Weighted Median Close | Count |
|----------|-------------------:|---------------------:|-----------------------:|------:|
| 1.0      | 1460.09            | 1251.66              | 3179.23                | 77.0  |
| 3.0      | 1529.19            | 1369.32              | 2945.25                | 50.0  |
| 4.0      | 1624.99            | 1464.21              | 3525.00                | 54.0  |
| 5.0      | 1346.69            | 1077.13              | 2876.53                | 8.0   |
| 6.0      | 1553.33            | 1432.43              | 3934.91                | 8.0   |
| 7.0      | 1868.16            | 1696.01              | 2827.08                | 4.0   |

We can see that slightly higher sales are shown in the upper bounds of severity.

## Correlation and ANOVA Results

| Comparison                          | Correlation / F-statistic | p-value   |
|-------------------------------------|--------------------------:|----------:|
| Weather Severity vs Day Sales       | 0.09                      | 0.18903   |
| Weather Severity vs Night Sales     | 0.04                      | 0.56608   |
| ANOVA (Day Sales)                   | F-statistic = 0.80         | 0.55011   |
| ANOVA (Night Sales)                 | F-statistic = 0.70         | 0.62600   |

The correlations are weak from direct results, and the ANOVA results show little significant impact of weather severity on sales, however from the weighted averages it shows some relationship, which is contradictory.

## Visualizations
Explain the key graphs and visualizations (you can add screenshots of your Power BI visualizations here as well).
