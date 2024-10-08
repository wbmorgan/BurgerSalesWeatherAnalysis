import psycopg2
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

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

try:
    # Connect to PostgreSQL and fetch the data
    con = psycopg2.connect(**db_params)
    cur = con.cursor()
    print("Connected to the PostgreSQL database successfully")

    # Execute SELECT query to retrieve information
    select_query = """
    SELECT 
        e.date, e.weather_desc, s.sales_open 
    FROM 
        weather_data_6hr e
    JOIN 
        sales_23 s ON e.date = s.date;
    """
    cur.execute(select_query)

    # Fetch all rows from query
    rows = cur.fetchall()

    # Convert rows to pandas DataFrame
    df = pd.DataFrame(rows, columns=['date', 'weather_desc', 'sales_open'])

    # Convert 'date' to datetime object for easier filtering and time extraction
    df['date'] = pd.to_datetime(df['date'])

    # Assign severity to weather descriptions based on the severity_map
    df['severity'] = df['weather_desc'].map(severity_map)

    # Convert sales_open to numeric and ensure it's numeric
    df['sales_open'] = pd.to_numeric(df['sales_open'], errors='coerce') 

    # Drop rows with NaN values in either severity or sales_open
    df.dropna(subset=['severity', 'sales_open'], inplace=True)

    # Prepare the dataset for machine learning
    # Use the date features, severity, and sales_open as target
    df['day_of_year'] = df['date'].dt.dayofyear  # Adding a feature for day of the year
    features = df[['day_of_year', 'severity']]
    target = df['sales_open']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')

    # Correlation between severity and sales (for reference)
    correlation, p_value = stats.pearsonr(df['severity'].dropna(), df['sales_open'].dropna())
    print(f'Correlation between weather severity and sales: {correlation:.2f} (p-value: {p_value:.5f})')

except psycopg2.Error as e:
    print(f'Error: {e}')

finally:
    # Ensure cursor and connection are closed properly
    if 'cur' in locals():
        cur.close()
    if 'con' in locals():
        con.close()
    print('PostgreSQL connection closed.')
