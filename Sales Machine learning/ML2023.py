import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import numpy as np

# Connect to the database
con = psycopg2.connect(
    host="Wills-Computer.local",
    database="fgsales",
    user="postgres",
    password="postgres"
)

# Execute query to select daily total sales for 2023 only
query = """
    SELECT 
        date,
        total_day,
        EXTRACT(YEAR FROM date) AS year
    FROM sales_23
"""

# Use pandas to execute query and store it in DataFrame
df = pd.read_sql(query, con)

# Close the connection
con.close()

# Event data dictionary: keys are event dates, values are a tuple of (event name, attendance)
event_data = {
    '2023-01-13': ('Hearts Game', 1),
    '2023-01-18': ('Hearts Game', 1),
    '2023-02-01': ('Hearts Game', 0),
    '2023-02-04': ('Hearts Game', 1),
    '2023-02-11': ('6N', 1),
    '2023-03-04': ('Hearts Game', 1),
    '2023-03-11': ('Hearts Game', 0),
    '2023-03-12': ('6N', 0),
    '2023-03-18': ('6N', 1),
    '2023-04-08': ('Hearts Game', 0),
    '2023-04-22': ('Hearts Game', 1),
    '2023-05-07': ('Hearts Game', 0),
    '2023-05-20': ('Hearts Game', 1),
    '2023-05-26': ('Concert', None),
    '2023-05-27': ('Hearts Game + Concert', -1),
    '2023-07-30': ('Hearts Game + Concert', 0),
    '2023-07-29': ('Rugby', 1),
    '2023-08-05': ('Rugby', 1),
    '2023-08-13': ('Hearts Game', -1),
    '2023-08-17': ('Hearts Game', 1),
    '2023-08-20': ('Hearts Game', 1),
    '2023-08-24': ('Hearts Game', 0),
    '2023-08-26': ('Rugby', 1),
    '2023-09-03': ('Hearts Game', 0),
    '2023-09-16': ('Hearts Game', 1),
    '2023-10-07': ('Hearts Game', -1),
    '2023-10-22': ('Hearts Game', 0),
    '2023-11-01': ('Hearts Game', 1),
    '2023-11-05': ('Hearts Game', 1),
    '2023-11-25': ('Hearts Game', 1),
    '2023-12-06': ('Hearts Game', 0),
    '2023-12-23': ('Hearts Game', 1),
    '2023-12-30': ('Hearts Game', -1)
}

# Convert dictionary keys (event dates) to datetime for easier comparison
event_data = {pd.to_datetime(k): v for k, v in event_data.items()}

# List of public holidays
holidays = [
    '2023-01-01',  # New Year's Day
    '2023-04-14',  # Good Friday
    '2023-04-17',  # Easter Monday
    '2023-05-01',  # Early May Bank Holiday
    '2023-05-29',  # Spring Bank Holiday
    '2023-08-28',  # Summer Bank Holiday
    '2023-12-25',  # Christmas Day
    '2023-12-26'   # Boxing Day
]

# Convert string dates to datetime
holidays = pd.to_datetime(holidays)





# Ensure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])  

# Add event day: 1 if it's an event, 0 otherwise
df['event_day'] = df['date'].apply(lambda x: 1 if x in event_data else 0)

# Add a new column for day of the year
df['day_of_year'] = df['date'].dt.dayofyear


# Feature Engineering: 

# only using lag_1 and lag_7 as others werent significant
df['total_day_lag_7'] = df['total_day'].shift(7)
df['total_day_lag_3'] = df['total_day'].shift(3)

# Define lag features to only include lag 1 and lag 7
#lag_features = df[['total_day_lag_1']]
lag_features = df[['total_day_lag_7', 'total_day_lag_3']]

event_features = df[['event_day']]

# Add day of the week
df['day_of_week'] = df['date'].dt.dayofweek

# Add month of the year
df['month'] = df['date'].dt.month

# Create a boolean feature for whether the date is near a holiday
df['is_near_holiday'] = df['date'].apply(lambda x: 1 if (x - pd.DateOffset(days=1) in holidays) or 
                                                       (x + pd.DateOffset(days=1) in holidays) else 0)


target = df['total_day']



# Drop NaN values due to lagging (these will be at the start of the dataset)
df.dropna(inplace=True)

# Combine these dummies with your existing features
# Combine the new features with your existing features
combined_features = pd.concat([lag_features, event_features,  
                               df[['is_near_holiday' ]]], axis=1)


# Drop rows where there are NaNs in either features or target
combined_features = combined_features.replace([np.inf, -np.inf], np.nan)
combined_features.dropna(inplace=True)
target = target.loc[combined_features.index]  # Align target with cleaned features



# Train the model on both event and lag features
model_combined = RandomForestRegressor(n_estimators=100, min_samples_split=80, random_state=1)
model_combined.fit(combined_features, target)

# Predict using the trained model
predictions_combined = model_combined.predict(combined_features)

# Evaluate the model with Mean Absolute Error (MAE)
mae_combined = mean_absolute_error(target, predictions_combined)
print(f'Mean Absolute Error (2023) with Combined Features: {mae_combined:.2f}')

# Plot actual vs predicted sales for 2023
plt.figure(figsize=(10, 6))
plt.plot(target.values, label='Actual Sales (2023)', alpha=0.7)
plt.plot(predictions_combined, label='Predicted Sales (2023)', alpha=0.7)
plt.title('Actual vs Predicted Sales for 2023 (With Combined Features)')
plt.xlabel('Days')
plt.ylabel('Sales (£)')
plt.legend()
plt.show()

# Define a rolling window size (e.g., 7-day rolling average)
window_size = 3

# Calculate rolling averages for actual and predicted sales
target_rolling = target.rolling(window=window_size).mean()
predictions_combined_rolling = pd.Series(predictions_combined).rolling(window=window_size).mean()

# Plot actual vs predicted sales with rolling averages for better readability
plt.figure(figsize=(10, 6))
plt.plot(target_rolling, label=f'Actual Sales (2023) - {window_size}-Day Rolling Avg', alpha=0.7)
plt.plot(predictions_combined_rolling, label=f'Predicted Sales (2023) - {window_size}-Day Rolling Avg', alpha=0.7)
plt.title(f'Actual vs Predicted Sales for 2023 with {window_size}-Day Rolling Average')
plt.xlabel('Days')
plt.ylabel('Sales (£)')
plt.legend()
plt.show()


# Get feature importances
importances = model_combined.feature_importances_

# Create a DataFrame for better visualization
features = combined_features.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:\n", importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Random Forest Model')
plt.show()

# Group by month to see the average sales
monthly_sales = df.groupby(df['month'])['total_day'].mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.title('Average Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Average Sales (£)')
plt.xticks(monthly_sales.index)
plt.grid()
plt.show()

