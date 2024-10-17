import pandas as pd
import psycopg2

db_params = {
    "host": "localhost",
    "database": "fgsales",
    "user": "postgres",
    "password": "postgres"
}

# Load the CSV file using pandas
csv_file_path = '/Users/willb-m/Desktop/Five Guys Sales Data/fiveguys_sales_23_24.csv'
df = pd.read_csv(csv_file_path)

# Print original column names
print("Original Columns in DataFrame:", df.columns.tolist())

# Strip whitespace from column names
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert invalid dates to NaT

# Print cleaned column names to confirm
print("Cleaned Columns in DataFrame:", df.columns.tolist())

# Check if 'total_day' exists
if 'total_day' not in df.columns:
    print("Column 'total_day' not found in DataFrame.")
else:
    # Clean the 'total_day' column
    df['total_day'] = df['total_day'].str.replace('£', '', regex=False)  # Remove '£'
    df['total_day'] = df['total_day'].str.replace(',', '', regex=False)   # Remove commas
    df['total_day'] = df['total_day'].str.strip()  # Remove any leading/trailing spaces

    # Convert the cleaned 'total_day' to numeric type
    df['total_day'] = pd.to_numeric(df['total_day'], errors='coerce')  # Convert to numeric, replace errors with NaN

    # Remove rows with NaN values in numeric columns
    df = df.dropna(subset=['total_day'])

# Connect to PostgreSQL and execute your insert code
try:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    print("Connected to the PostgreSQL database successfully.")

    # Create the table (if it doesn't exist)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS total_sales_23_24 (
        date DATE,
        day_of_wk TEXT,
        total_day NUMERIC(10, 2),
        cineworld TEXT,
        localevents TEXT,
        holidays TEXT
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    print("Table created successfully.")

    # Insert data from DataFrame into PostgreSQL table
    for index, row in df.iterrows():
        insert_query = """
        INSERT INTO total_sales_23_24 (date, day_of_wk, total_day, cineworld, localevents, holidays)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        cur.execute(insert_query, (
            row['date'],
            row['day_of_wk'],
            row['total_day'],        # This should now be a numeric value
            row['cineworld'],
            row['localevents'],
            row['holidays']
        ))
        conn.commit()  # Commit data after each insert

except psycopg2.Error as e:
    print(f"Error: {e}")

finally:
    if conn:
        cur.close()
        conn.close()
        print("PostgreSQL connection closed.")
