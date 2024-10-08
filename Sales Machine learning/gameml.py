import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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

# Event data dictionary
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

# Ensure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Add event day: 1 if it's an event, 0 otherwise
df['event_day'] = df['date'].apply(lambda x: 1 if x in event_data else 0)

# Add game result with string values: 'Win', 'Loss', 'Draw', or 'No Game'
df['game_result'] = df['date'].apply(
    lambda x: 'Win' if event_data.get(x, [None, None])[1] == 1 
              else ('Loss' if event_data.get(x, [None, None])[1] == 0 
              else ('Draw' if event_data.get(x, [None, None])[1] == -1 
              else 'No Game'))
)

# Map game results to numerical values for modeling purposes
# 'Win' -> 1, 'Loss' -> 0, 'Draw' -> -1, 'No Game' -> NaN (or 0 if you prefer)
result_mapping = {'Win': 1, 'Loss': 0, 'Draw': -1, 'No Game': None}

df['game_result_numeric'] = df['game_result'].map(result_mapping)

# Fill 'No Game' (None) with a neutral value if needed (e.g., 0)
df['game_result_numeric'].fillna(0, inplace=True)  # or keep NaN if you prefer

# Check the dataframe to ensure correct mapping
print(df[['date', 'event_day', 'game_result', 'game_result_numeric']].head())


# Add a new column for day of the year
df['day_of_year'] = df['date'].dt.dayofyear

# Feature Engineering: Create lagged features for 2023 data
for lag in range(1, 8):  # Create 7 lagged features for the past week
    df[f'total_day_lag_{lag}'] = df['total_day'].shift(lag)

# Drop NaN values due to lagging
df.dropna(inplace=True)

# Select only the last lag feature and new event features
lag_features = df[['total_day_lag_7']]
event_features = df[['event_day', 'game_result_numeric']]
target = df['total_day']

# Train the model with only total_day_lag_7
model_lag = RandomForestRegressor(n_estimators=100, min_samples_split=100, random_state=1)
model_lag.fit(lag_features, target)

# Train the model with event features
model_event = RandomForestRegressor(n_estimators=100, min_samples_split=100, random_state=1)
model_event.fit(event_features, target)

# Get feature importances for lag features
importances_lag = model_lag.feature_importances_
importance_lag_df = pd.DataFrame({'Feature': lag_features.columns, 'Importance': importances_lag})
importance_lag_df.sort_values(by='Importance', ascending=False, inplace=True)


# Get feature importances for event features
importances_event = model_event.feature_importances_
importance_event_df = pd.DataFrame({'Feature': event_features.columns, 'Importance': importances_event})
importance_event_df.sort_values(by='Importance', ascending=False, inplace=True)

# Print the feature importances for event features
print("\nEvent Features Importance:")
print(importance_event_df)

# Evaluate the model with Mean Absolute Error (MAE) on both feature sets
predictions_lag = model_lag.predict(lag_features)
predictions_event = model_event.predict(event_features)

mae_lag = mean_absolute_error(target, predictions_lag)
mae_event = mean_absolute_error(target, predictions_event)

print(f'Mean Absolute Error (Lag Features): {mae_lag:.2f}')
print(f'Mean Absolute Error (Event Features): {mae_event:.2f}')

# Plot actual vs predicted sales for 2023 using lag features
plt.figure(figsize=(10, 6))
plt.plot(target.values, label='Actual Sales (2023)', alpha=0.7)
plt.plot(predictions_lag, label='Predicted Sales with Lag Features', alpha=0.7)
plt.title('Actual vs Predicted Sales for Q1 2023 (Using total_day_lag_7)')
plt.xlabel('Days')
plt.ylabel('Sales (£)')
plt.legend()
plt.show()

# Plot actual vs predicted sales for 2023 using event features
plt.figure(figsize=(10, 6))
plt.plot(target.values, label='Actual Sales (2023)', alpha=0.7)
plt.plot(predictions_event, label='Predicted Sales with Event Features', alpha=0.7)
plt.title('Actual vs Predicted Sales for Q1 2023 (Event Features)')
plt.xlabel('Days')
plt.ylabel('Sales (£)')
plt.legend()
plt.show()

# Filter the data for wins and losses
wins_df = df[df['game_result'] == 'Win']
losses_df = df[df['game_result'] == 'Loss']

# Calculate average sales for wins and losses
average_sales_win = wins_df['total_day'].mean()
average_sales_loss = losses_df['total_day'].mean()

# Print the results
print(f"Average Sales on Win Days: £{average_sales_win:.2f}")
print(f"Average Sales on Loss Days: £{average_sales_loss:.2f}")

# Optional: Plot the comparison
import matplotlib.pyplot as plt

# Data to plot
labels = ['Win', 'Loss']
average_sales = [average_sales_win, average_sales_loss]

# Create bar chart
plt.bar(labels, average_sales, color=['green', 'red'])
plt.title('Average Sales: Win vs Loss')
plt.ylabel('Sales (£)')
plt.show()


