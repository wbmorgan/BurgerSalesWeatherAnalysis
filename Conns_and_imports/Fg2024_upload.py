import psycopg2
import pandas as pd

# Database connection params for PostgreSQL
db_params = {
    "host": "localhost",
    "database": "fgsales",
    "user": "postgres",
    "password": "postgres"
}

# Load the CSV file using pandas
csv_file_path = '/Users/willb-m/Desktop/FIve Guys Sales Data/Five_guys_sales_q1_2024.csv'
df = pd.read_csv(csv_file_path)

# Replace 'NaN' or any string 'NaN' with actual None values
df = df.replace('NaN', None)

# Convert 'date' column to datetime to ensure it matches PostgreSQL DATE type
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert invalid dates to NaT

# Remove rows where the 'date' is missing or invalid (NaT)
df = df.dropna(subset=['date'])

# Ensure numeric columns are properly formatted
df['sales_open'] = pd.to_numeric(df['sales_open'], errors='coerce')
df['sales_close'] = pd.to_numeric(df['sales_close'], errors='coerce')
df['total_day'] = pd.to_numeric(df['total_day'], errors='coerce')

# Remove rows with NaN values in numeric columns
df = df.dropna(subset=['sales_open', 'sales_close', 'total_day'])

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    print("Connected to the PostgreSQL database successfully.")

    # Create the table (if it doesn't exist)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Fgsales2024_q1 (
        date DATE,                -- Date column
        day_of_week TEXT,         -- Day of the week
        sales_open NUMERIC(10, 2),  -- Opening Sales as currency (up to 10 digits, 2 decimals)
        sales_close NUMERIC(10, 2), -- Closing Sales as currency (up to 10 digits, 2 decimals)
        total_day NUMERIC(10, 2)    -- Total day sales as currency (up to 10 digits, 2 decimals)
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    print("Table created successfully.")

    # Insert data from DataFrame into PostgreSQL table
    for index, row in df.iterrows():
        insert_query = """
        INSERT INTO Fgsales2024_q1 (date, day_of_week, sales_open, sales_close, total_day)
        VALUES (%s, %s, %s, %s, %s);
        """
        cur.execute(insert_query, (
            row['date'],             # Date column (now handled as a valid DATE)
            row['day_of_week'],      # Day of the week
            row['sales_open'],       # Opening Sales
            row['sales_close'],      # Closing Sales
            row['total_day'],        # Total Day Sales
        ))
        conn.commit()  # Commit data after each insert

except psycopg2.Error as e:
    print(f"Error: {e}")

finally:
    if conn:
        cur.close()
        conn.close()
        print("PostgreSQL connection closed.")

