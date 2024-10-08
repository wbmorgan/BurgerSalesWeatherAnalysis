import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the database
con = psycopg2.connect(
    host="Wills-Computer.local",
    database="fgsales",
    user="postgres",
    password="postgres"
)

# Execute query to select average total sales by day of week for both years
query = """
    SELECT 
        EXTRACT(DOW FROM date) AS day_of_week,
        AVG(total_day) AS avg_total_sales,
        EXTRACT(YEAR FROM date) AS year
    FROM (
        SELECT 
            date, 
            total_day 
        FROM 
            sales_23 
        WHERE 
            date >= '2023-01-01' AND date < '2024-04-01'  -- Only first three months of 2023
        UNION ALL
        SELECT 
            date, 
            total_day 
        FROM 
            Fgsales2024_q1 
        WHERE 
            date >= '2024-01-01' AND date < '2024-04-01'  -- Only first three months of 2024
    ) AS combined_sales
    GROUP BY 
        day_of_week, year
    ORDER BY 
        year, day_of_week;
"""

# Use pandas to execute query and store it in DataFrame
df = pd.read_sql(query, con)

# Close the connection
con.close()

# Map day of week to week day names
day_map = {
    0: 'Sunday',
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday'
}
df['day_of_week'] = df['day_of_week'].map(day_map)

# Pivot the DataFrame to have years as columns for easier plotting
df_pivot = df.pivot(index='day_of_week', columns='year', values='avg_total_sales').fillna(0)

# Print first few rows to check 
print(df_pivot)

# Plot data
plt.figure(figsize=(10, 6))

# Bar width
bar_width = 0.35

# Set positions of bar on X axis
r1 = range(len(df_pivot))
r2 = [x + bar_width for x in r1]

# Create bars for each year
plt.bar(r1, df_pivot[2023], color='skyblue', width=bar_width, edgecolor='grey', label='2023')
plt.bar(r2, df_pivot[2024], color='lightgreen', width=bar_width, edgecolor='grey', label='2024')

# Add labels
plt.title('Average Total Sales by Day of the Week (Q1 2023 vs Q1 2024)')
plt.xlabel('Day of the Week')
plt.ylabel('Average Total Sales')
plt.xticks([r + bar_width / 2 for r in range(len(df_pivot))], df_pivot.index, rotation=45)  # Set x-tick labels to day of week
plt.legend()
plt.grid(axis='y')  # Add horizontal grid lines

plt.tight_layout()
plt.show()
