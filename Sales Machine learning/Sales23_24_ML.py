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

# Execute query to select daily total sales for both years
query = """
    SELECT 
        date,
        total_day,
        EXTRACT(YEAR FROM date) AS year
    FROM (
        SELECT 
            date, 
            total_day 
        FROM 
            sales_23 
        WHERE 
            date >= '2023-01-01' AND date < '2023-04-01'  -- Only first three months of 2023
        UNION ALL
        SELECT 
            date, 
            total_day 
        FROM 
            Fgsales2024_q1 
        WHERE 
            date >= '2024-01-01' AND date < '2024-04-01'  -- Only first three months of 2024
    ) AS combined_sales
    ORDER BY 
        date;
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
    '2023-12-30': ('Hearts Game', -1),


    # Add more events as necessary
}

# Convert dictionary keys (event dates) to datetime for easier comparison
event_data = {pd.to_datetime(k): v for k, v in event_data.items()}


# Assuming df is your sales DataFrame with a 'date' column
df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' column is in datetime format

# Add event day: 1 if it's an event, 0 otherwise
df['event_day'] = df['date'].apply(lambda x: 1 if x in event_data else 0)

# Add event name: 'No Event' if no event is found on the date
df['event_name'] = df['date'].apply(lambda x: event_data[x][0] if x in event_data else 'No Event')

# Add game result: Handle missing dates gracefully with .get()
df['game_result'] = df['date'].apply(lambda x: 1 if event_data.get(x, [None, None])[1] == 1 
                                     else (-1 if event_data.get(x, [None, None])[1] == -1 
                                     else 0))


# You can also calculate proximity to the event (e.g., days until or after the event)
df['days_until_event'] = df['date'].apply(lambda x: (min([abs((event_date - x).days) for event_date in event_data.keys()]) 
                                                      if any(event_data) else None))

# Check the DataFrame to see how event data has been added
print(df.head())

# Change 'date' to datetime object
df['date'] = pd.to_datetime(df['date'])

# Add a new column for day of the year
df['day_of_year'] = df['date'].dt.dayofyear

# Separate data for 2023 and 2024
df_2023 = df[df['year'] == 2023].copy()
df_2024 = df[df['year'] == 2024].copy()

# Feature Engineering: Create lagged features for both 2023 and 2024 data
for lag in range(1, 8):  # Create 7 lagged features for the past week
    df_2023[f'total_day_lag_{lag}'] = df_2023['total_day'].shift(lag)
    df_2024[f'total_day_lag_{lag}'] = df_2024['total_day'].shift(lag)

# Drop NaN values due to lagging (these will be at the start of the dataset)
df_2023.dropna(inplace=True)
df_2024.dropna(inplace=True)

# Define features (lagged sales) and target (current day's sales)
features_2023 = df_2023[[f'total_day_lag_{lag}' for lag in range(1, 8)] + ['event_day', 'game_result']]
target_2023 = df_2023['total_day']

features_2024 = df_2024[[f'total_day_lag_{lag}' for lag in range(1, 8)] + ['event_day', 'game_result']]
target_2024 = df_2024['total_day']

features_2024.dropna(inplace=True)
target_2024 = target_2024.loc[features_2024.index]  # Ensure target matches features


print(features_2024.isnull().sum())

print("Features 2024 shape:", features_2024.shape)
print("Target 2024 shape:", target_2024.shape)

# Display some sample data
print("Features 2024 sample:\n", features_2024.head())
print("Target 2024 sample:\n", target_2024.head())

# Train the Random Forest Regressor on 2023 data
model = RandomForestRegressor(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(features_2023, target_2023)

# Predict sales for 2024 data
predictions_2024 = model.predict(features_2024)

importances = model.feature_importances_
feature_names = features_2024.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

print(importance_df)

# Evaluate the model with Mean Absolute Error (MAE) on 2024 data
mae = mean_absolute_error(target_2024, predictions_2024)
print(f'Mean Absolute Error (2024): {mae:.2f}')

# Plot actual vs predicted sales for 2024
plt.figure(figsize=(10, 6))
plt.plot(target_2024.values, label='Actual Sales (2024)', alpha=0.7)
plt.plot(predictions_2024, label='Predicted Sales (2024)', alpha=0.7)
plt.title('Actual vs Predicted Sales for Q1 2024')
plt.xlabel('Days')
plt.ylabel('Sales (£)')
plt.legend()
plt.show()
