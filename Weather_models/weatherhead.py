import pandas as pd
import numpy as np
import plotly.graph_objects as go
import psycopg2
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Database connection parameters for PostgreSQL
db_params = {
    "host": "localhost",
    "database": "fgsales",
    "user": "postgres",
    "password": "postgres"
}

# Define the severity scale for weather descriptions
severity_map = {
    "Clear": 1,
    "Partly cloudy": 2,
    "Cloudy": 3,
    "Patchy rain possible": 4,
    "Light rain": 5,
    "Moderate rain": 6,
    "Heavy rain": 7,
    "Moderate rain at times": 6, 
    "Light drizzle": 5,
    "Moderate or heavy rain shower": 8,
    "Thundery outbreaks possible": 9,
    "Light snow": 5,
    "Patchy heavy snow": 5,
    "Moderate or heavy sleet": 6,
    "Fog": 2,
    "Blizzard": 9,
    "Overcast": 3,
    "Mist": 2,
    "Patchy snow possible": 4,
    "Blowing snow": 4,
}

# Define a mapping from holiday names to dates
holiday_mapping = {
    "New Year's Day": '2024-01-01',
    'Good Friday': '2024-03-29',
    'Easter Monday': '2024-04-01',
    "Valentine's Day": '2024-02-14',
    "Mother's Day (UK)": '2024-03-10',
    "Father's Day (UK)": '2024-06-16',
    'Early May Bank Holiday': '2024-05-06',
    'Spring Bank Holiday': '2024-05-27',
    'Summer Bank Holiday': '2024-08-26',
    'Boxing Day': '2024-12-26',
}




try:
    # Connect to PostgreSQL and fetch the data
    con = psycopg2.connect(**db_params)
    cur = con.cursor()
    print("Connected to the PostgreSQL database successfully")

    # Execute SELECT query to retrieve information
    select_query = """
    SELECT 
        e.date, e.weather_desc, s.total_day, s.localevents, s.cineworld, s.holidays
    FROM 
        weather_data_6hr e
    JOIN 
        total_sales_23_24 s ON e.date = s.date;
    """
    cur.execute(select_query)

    # Fetch all rows from query
    rows = cur.fetchall()

    # Convert rows to pandas DataFrame
    df = pd.DataFrame(rows, columns=['date', 'weather_desc', 'total_day', 'localevents', 'cineworld', 'holidays'])

    # Close the database connection
    cur.close()
    con.close()
    print("PostgreSQL connection closed.")

except Exception as e:
    print(f"Error: {e}")

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Clean 'total_day' for numeric conversion
df['total_day'] = pd.to_numeric(df['total_day'].astype(str).str.replace('£', '').str.replace(',', '').str.strip(), errors='coerce')


# Map severity to weather descriptions
df['weather_severity'] = df['weather_desc'].map(severity_map)

# Fill NaN values with a default value if any descriptions were unmapped
df['weather_severity'].fillna(0, inplace=True)  # or use an appropriate default severity



# Create binary columns for events, holidays, and cinema
df['has_event'] = df['localevents'].notna().astype(int)
df['has_cineworld'] = df['cineworld'].notna().astype(int)
df['has_holiday'] = df['holidays'].notna().astype(int)



# Create lagged sales feature
df['total_day_lag_7'] = df['total_day'].shift(7)

# Fill NaN values with the mean of the total_day


print("DataFrame after creating lagged features:")
print(df.isna().sum())
print(df.shape)  # Check the number of rows and columns

# Create features for first day of the month, day of week, and month
df['first_day_of_month'] = df['date'].dt.is_month_start.astype(int)
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Create payday feature (last Friday of the month)
df['payday'] = ((df['date'].dt.day == df['date'].dt.days_in_month) & 
                 (df['date'].dt.dayofweek == 4)).astype(int)

# Define the number of days before a holiday that you consider to be part of the holiday effect window
holiday_window = 3  # 3 days before and after the holiday

# Ensure 'holidays' is in datetime format
df['holiday_date'] = df['holidays'].map(holiday_mapping)
df['holiday_date'] = pd.to_datetime(df['holiday_date'], errors='coerce')


# Create a new column 'days_to_holiday' that calculates the difference in days between the current date and the nearest holiday
df['days_to_holiday'] = (df['holiday_date'] - df['date']).dt.days

# Create a binary column for whether the current date is within the holiday window (before or after)
df['near_holiday'] = ((df['days_to_holiday'] >= -holiday_window) & (df['days_to_holiday'] <= holiday_window)).astype(int)

# Fill NaN values in columns rather than dropping all rows
df['total_day'].fillna(df['total_day'].mean(), inplace=True)





# Drop rows with NaN values (due to lagging or incomplete rows)
df = df.dropna()

# Define feature columns
feature_columns = ['weather_severity', 'has_event', 'has_cineworld', 'has_holiday', 
                   'total_day_lag_7' , 'first_day_of_month', 
                   'day_of_week', 'month', 'payday', 'near_holiday']

# Add the 'date' column to the features DataFrame for tracking predictions
X = df[feature_columns + ['date']]
y = df['total_day']


# Check for empty datasets
if len(X) == 0 or len(y) == 0:
    print("The dataset is empty, check data fetching and preprocessing steps.")


# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train the XGBoost model with the best-found parameters
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=1.0,
    gamma=0,
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=2,
    n_estimators=200,
    subsample=1.0
)

model.fit(X_train[feature_columns], y_train)

# Make predictions
y_pred = model.predict(X_test[feature_columns])

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

# Visualize the predictions
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_test['date'], y=y_test, mode='lines', name='Actual Sales'))
fig.add_trace(go.Scatter(x=X_test['date'], y=y_pred, mode='lines', name='Predicted Sales'))
fig.update_layout(title='Actual vs Predicted Sales', xaxis_title='Date', yaxis_title='Sales')
fig.show()
