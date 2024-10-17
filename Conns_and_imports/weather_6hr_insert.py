import time
import requests
import json
from datetime import datetime, timedelta
import psycopg2

# Define your API key and base URL
api_key = '70b92de967654084863154611242409'
base_url = 'https://api.worldweatheronline.com/premium/v1/past-weather.ashx'

import requests
import json
import psycopg2
from datetime import datetime, timedelta
import time

# Define the location
location = 'Edinburgh, United Kingdom'

# Database connection details
db_config = {
    'dbname': 'fgsales',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
}

# Function to make API requests and collect data for each day
def fetch_weather_data_for_day(location, date):
    params = {
        'key': api_key,
        'q': location,
        'date': date,
        'tp': 6,  # 6-hour intervals
        'format': 'json'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        print(f"Data for {date} fetched successfully.")
        return response.json()
    elif response.status_code == 429:
        print(f"Rate limit hit. Status code: 429. Waiting for 60 seconds.")
        time.sleep(60)  # Wait for 60 seconds before retrying
        return fetch_weather_data_for_day(location, date)
    else:
        print(f"Failed to fetch data for {date}. Status code: {response.status_code}")
        return None

# Function to insert weather data into PostgreSQL
def insert_weather_data(cursor, day, location):
    date = day['date']
    for hourly in day['hourly']:
        time = int(hourly['time']) // 100  # Convert 'time' from 'hmm' to 'h'
        date_time = f"{date} {time}:00:00"
        
        # Check if the record already exists
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM weather_data_6hr WHERE date = %s AND location = %s)",
            (date_time, location)
        )
        exists = cursor.fetchone()[0]
        
        if not exists:
            cursor.execute(
                """
                INSERT INTO weather_data_6hr (date, tempC, tempF, humidity, windspeedMiles, windspeedKmph, weather_desc, location, raw_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    date_time,
                    int(hourly['tempC']),
                    int(hourly['tempF']),
                    int(hourly['humidity']),
                    int(hourly['windspeedMiles']),
                    int(hourly['windspeedKmph']),
                    hourly['weatherDesc'][0]['value'],
                    location,
                    json.dumps(hourly)  # Store the raw JSON data
                )
            )
            print(f"Inserted data for {date_time}.")
        else:
            print(f"Data for {date_time} already exists, skipping insertion.")


# Establish a connection to the PostgreSQL database
try:
    connection = psycopg2.connect(**db_config)
    connection.autocommit = True
    cursor = connection.cursor()
    print("Connected to the database successfully.")
    
    # Create the weather_data_6hr table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data_6hr (
            id SERIAL PRIMARY KEY,
            date TIMESTAMP,
            tempC INT,
            tempF INT,
            humidity INT,
            windspeedMiles INT,
            windspeedKmph INT,
            weather_desc TEXT,
            location VARCHAR(255),
            raw_data JSONB
        );
    """)
    print("Table weather_data_6hr created or already exists.")
    
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

# Collect data from 2023 until today
start_date = datetime(2024, 1, 1)
end_date = datetime.now()

current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    day_data = fetch_weather_data_for_day(location, date_str)
    
    if day_data and 'data' in day_data and 'weather' in day_data['data']:
        for day in day_data['data']['weather']:
            try:
                insert_weather_data(cursor, day, location)
                print(f"Data for {day['date']} inserted successfully.")
            except Exception as e:
                print(f"Failed to insert data for {day['date']}: {e}")
    
    # Add a delay of 1 second to avoid hitting rate limits
    time.sleep(1)
    current_date += timedelta(days=1)

# Close the database connection
cursor.close()
connection.close()
print("Database connection closed.") 


