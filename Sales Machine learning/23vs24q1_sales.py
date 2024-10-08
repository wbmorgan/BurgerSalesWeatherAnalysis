import psycopg2
import pandas as pd
import plotly.graph_objects as go

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

# Change 'date' to datetime object for better plotting
df['date'] = pd.to_datetime(df['date'])

# Add a new column for day of the year
df['day_of_year'] = df['date'].dt.dayofyear

# Separate sales data for each year
sales_2023 = df[df['year'] == 2023][['day_of_year', 'total_day']].set_index('day_of_year')
sales_2024 = df[df['year'] == 2024][['day_of_year', 'total_day']].set_index('day_of_year')

# Align the two DataFrames by day_of_year
aligned_data = sales_2023.join(sales_2024, lsuffix='_2023', rsuffix='_2024')

# Calculate the correlation
correlation = aligned_data.corr().iloc[0, 1]  # Correlation between sales_2023 and sales_2024
print(f'Correlation between daily sales in 2023 and 2024: {correlation:.2f}')

# Create interactive plotly figure
fig = go.Figure()

# Add sales line plot for 2023
fig.add_trace(go.Scatter(
    x=aligned_data.index,
    y=aligned_data['total_day_2023'],
    mode='lines+markers',
    name='Sales 2023',
    marker=dict(color='skyblue', size=6),  # Adjust marker size
    line=dict(width=2),  # Adjust line width
    hovertemplate='Day of Year: %{x}<br>Sales 2023: %{y:.2f}<extra></extra>',
))

# Add sales line plot for 2024
fig.add_trace(go.Scatter(
    x=aligned_data.index,
    y=aligned_data['total_day_2024'],
    mode='lines+markers',
    name='Sales 2024',
    marker=dict(color='lightgreen', size=6),  # Adjust marker size
    line=dict(width=2),  # Adjust line width
    hovertemplate='Day of Year: %{x}<br>Sales 2024: %{y:.2f}<extra></extra>',
))

# Update layout for single x-axis
fig.update_layout(
    title='Daily Total Sales Comparison: Q1 2023 vs Q1 2024',
    xaxis_title='Day of Year (1-90)',
    yaxis_title='Total Day Sales (£)',
    hovermode='x unified',
    template='plotly_white',
    xaxis=dict(
        tickvals=list(range(1, 91)),  # Set ticks for days 1 to 90
        ticktext=[str(i) for i in range(1, 91)],  # Show days 1-90 as labels
    ),
    yaxis=dict(
        title='Sales (£)',
        gridcolor='lightgray',  # Light gray grid lines
        showgrid=True,  # Show grid
    ),
    legend=dict(
        title='Sales Year',
        orientation='h',  # Horizontal orientation
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

# Show the plot
fig.show()
