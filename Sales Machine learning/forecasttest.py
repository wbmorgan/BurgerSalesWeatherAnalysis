import pandas as pd
import numpy as np
import psycopg2
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import datetime

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
    "Partly Cloudy": 2,
    "Cloudy": 3,
    "Patchy rain possible": 4,
    "Light rain": 5,
    "Moderate rain at times": 6,
    "Moderate rain": 7,
    "Heavy rain": 8,
    "Storm": 9,
    "Thunderstorm": 9
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

# Create binary columns for events, holidays, and cinema
df['has_event'] = df['localevents'].notna().astype(int)
df['has_cineworld'] = df['cineworld'].notna().astype(int)
df['has_holiday'] = df['holidays'].notna().astype(int)

# Create lagged sales features
for lag in [1, 2, 3, 5, 7]:
    df[f'lagged_sales_{lag}'] = df['total_day'].shift(lag)

# Create features for first day of the month, day of week, and month
df['first_day_of_month'] = df['date'].dt.is_month_start.astype(int)
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Create payday feature (last Friday of the month)
df['payday'] = ((df['date'].dt.day == df['date'].dt.days_in_month) & 
                (df['date'].dt.dayofweek == 4)).astype(int)

# Create a binary column for the 3 days after payday
df['three_days_after_payday'] = (
    (df['payday'].shift(1).fillna(0) == 1) | 
    (df['payday'].shift(2).fillna(0) == 1) | 
    (df['payday'].shift(3).fillna(0) == 1)
).astype(int)

# Drop rows with NaN values (due to lagging)
df = df.dropna()

holiday_window = 7  # 3 days before and after the holiday

# Ensure 'holidays' is in datetime format
df['holiday_date'] = df['holidays'].map(holiday_mapping)
df['holiday_date'] = pd.to_datetime(df['holiday_date'], errors='coerce')

# Fill missing values in the holidays column with a placeholder
df['holidays'] = df['holidays'].fillna('No Holiday')

# Create a new column 'days_to_holiday' that calculates the difference in days between the current date and the nearest holiday
df['days_to_holiday'] = (df['holiday_date'] - df['date']).dt.days.fillna(np.inf)  # Use np.inf for no holiday

# Create a binary column for whether the current date is within the holiday window (before or after)
df['near_holiday'] = ((df['days_to_holiday'] >= -holiday_window) & (df['days_to_holiday'] <= holiday_window)).astype(int)


# Fill NaN values in columns rather than dropping all rows
df['total_day'].fillna(df['total_day'].mean(), inplace=True)

# Get today's date and prepare today's features
today = pd.to_datetime(datetime.datetime.now().date())

# Prepare today's features
today_features = pd.DataFrame({
    'date': [today],
    'weather_severity': [1],  # Replace with actual weather severity from your data
    'has_event': [0],          # Replace with actual event data from your data
    'has_cineworld': [0],      # Replace with actual Cineworld data from your data
    'has_holiday': [0],        # Replace with actual holiday data from your data
    'lagged_sales_1': [df['total_day'].iloc[-1]],  # Last known sales value
    'lagged_sales_3': [df['total_day'].iloc[-3]],  # Sales from 3 days ago
    'lagged_sales_5': [df['total_day'].iloc[-5]],  # Sales from 5 days ago
    'lagged_sales_7': [df['total_day'].iloc[-7]],  # Sales from 7 days ago
    'first_day_of_month': [1 if today.day == 1 else 0],
    'day_of_week': [today.dayofweek],
    'month': [today.month],
    'payday': [1 if today.day == today.days_in_month and today.weekday() == 4 else 0],
    'three_days_after_payday': [0],  # Replace with logic to check if today is 3 days after payday
    'near_holiday': [0]  # Replace with logic to check if near a holiday
})

# Ensure all feature columns are defined before the prediction
feature_columns = ['weather_severity', 'has_event', 'has_cineworld', 'has_holiday', 
                   'lagged_sales_1', 'lagged_sales_3', 'lagged_sales_5', 'lagged_sales_7', 
                   'first_day_of_month', 'day_of_week', 'month', 'payday', 
                   'three_days_after_payday', 'near_holiday']



# Add the 'date' column to the features DataFrame
X = df[feature_columns + ['date']]  # Include 'date' in features for later use
y = df['total_day']

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)  # Important: No shuffling for time series

# Train the XGBoost model with best-found parameters
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=1.0,
    gamma=0,
    learning_rate=0.01,
    max_depth=3,
    min_child_weight=1,
    n_estimators=1000,
    subsample=0.8,
    tree_method='auto',
    n_jobs=4,
)

model.fit(X_train[feature_columns], y_train)

# Evaluate the model
y_pred = model.predict(X_test[feature_columns])

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# Make a prediction for today
if not today_features.empty:
    expected_sales = model.predict(today_features[feature_columns])
    print(f'Expected sales for today ({today.date()}): £{expected_sales[0]:.2f}')
else:
    print("Error: Today's features are not available.")

