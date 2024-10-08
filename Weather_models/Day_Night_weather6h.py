import psycopg2
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# Database connection params for PostgreSQL
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

# Define colors for each severity level
color_map = {
    1: 'lightgreen',
    2: 'yellowgreen',
    3: 'yellow',
    4: 'lightblue',
    5: 'blue',
    6: 'orange',
    7: 'orange',
    8: 'red',
    9: 'darkred'
}

try:
    # Connect to PostgreSQL and fetch the data
    con = psycopg2.connect(**db_params)
    cur = con.cursor()
    print("Connected to the PostgreSQL database successfully")

    # Execute SELECT query to retrieve information
    select_query = """
    SELECT 
        e.date, e.weather_desc, s.sales_open, s.sales_close
    FROM 
        weather_data_6hr e
    JOIN 
        sales_23 s ON e.date = s.date;
    """
    cur.execute(select_query)

    # Fetch all rows from query
    rows = cur.fetchall()

    # Convert rows to pandas DataFrame
    df = pd.DataFrame(rows, columns=['date', 'weather_desc', 'sales_open', 'sales_close'])

    # Convert 'date' to datetime object for easier filtering and time extraction
    df['date'] = pd.to_datetime(df['date'])

    # Assign severity to weather descriptions based on the severity_map
    df['severity'] = df['weather_desc'].map(severity_map)

    # Convert Sales to numeric and drop rows with NaN values
    df['sales_open'] = pd.to_numeric(df['sales_open'], errors='coerce')
    df['sales_close'] = pd.to_numeric(df['sales_close'], errors='coerce')  

    df.dropna(subset=['severity', 'sales_open', 'sales_close'], inplace=True)

    # Calculate frequency of each severity level
    severity_counts = df['severity'].value_counts(normalize=True)

    # Assign weights inversely proportional to the frequency of each severity level
    df['weight'] = df['severity'].map(lambda x: 1 / severity_counts[x])

    # Group data by severity and weighted mean and median sales
    severity_sales_summary = df.groupby('severity').apply(
        lambda x: pd.Series({
            'weighted_mean_open': (x['sales_open'] * x['weight']).sum() / x['weight'].sum(),
            'weighted_median_open': x['sales_open'].median(),
            'weighted_mean_close': (x['sales_close'] * x['weight']).sum() / x['weight'].sum(),
            'weighted_median_close': x['sales_close'].median(),    # Median does not need to be weighted
            'count': x.shape[0]
        })
    )
    print(severity_sales_summary)


    # Create box plots for each severity level
    fig = go.Figure()

    for severity, color in color_map.items():
        severity_data = df[df['severity'] == severity]
        fig.add_trace(go.Box(
            y=severity_data['sales_open'],
            name=f'Severity {severity}',
            marker=dict(color=color),
            boxmean='sd',
            boxpoints='all',
            jitter=0.5,
            pointpos=-1.8
        ))

    # Highlighting severity 7
    fig.add_annotation(
        x=6.5,
        y=severity_sales_summary.loc[7, 'weighted_mean_open'] + 50,
        text="Highest Sales",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color='black', size=12)
    )

    # Customize layout with legend/key
    fig.update_layout(
        title='Day Sales Distribution by Weather Severity',
        xaxis_title='Weather Severity (1 = Clear, 9 = Storm/Thunderstorm)',
        yaxis_title='Sales (£)',
        boxmode='group',
        showlegend=True,
        width=1200,
        height=600,
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref='paper',
                yref='paper',
                text='Severity Key:<br>1: Clear (light green)<br>2: Partly Cloudy (yellow-green)<br>3: Cloudy (yellow)<br>4: Patchy rain possible (light blue)<br>5: Light rain (blue)<br>6: Moderate rain at times (orange)<br>7: Moderate rain (orange)<br>8: Heavy rain (red)<br>9: Storm/Thunderstorm (dark red)',
                showarrow=False,
                font=dict(size=12),
                align='center'
            )
        ]
    )

    fig.show()

    fig = go.Figure()

    for severity, color in color_map.items():
        severity_data = df[df['severity'] == severity]
        fig.add_trace(go.Box(
            y=severity_data['sales_close'],
            name=f'Severity {severity}',
            marker=dict(color=color),
            boxmean='sd',  # Show mean and standard deviation
            boxpoints='all',  # Show all data points
            jitter=0.5,  # Add jitter for better visibility
            pointpos=-1.8  # Position of points relative to the box
        ))

    # Highlighting severity 7
    fig.add_annotation(
        x=6.5,  # position for annotation
        y=severity_sales_summary.loc[7, 'weighted_mean_close'] + 50,  # position above the mean of severity 7
        text="Highest Sales",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color='black', size=12)
    )

    # Customize layout with legend/key
    fig.update_layout(
        title='Night Sales Distribution by Weather Severity',
        xaxis_title='Weather Severity (1 = Clear, 9 = Storm/Thunderstorm)',
        yaxis_title='Sales (£)',
        boxmode='group',
        showlegend=True,
        width=1200,
        height=600,
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref='paper', 
                yref='paper',
                text='Severity Key:<br>1: Clear (light green)<br>2: Partly Cloudy (yellow-green)<br>3: Cloudy (yellow)<br>4: Patchy rain possible (light blue)<br>5: Light rain (blue)<br>6: Moderate rain at times (orange)<br>7: Moderate rain (orange)<br>8: Heavy rain (red)<br>9: Storm/Thunderstorm (dark red)',
                showarrow=False,
                font=dict(size=12),
                align='center'
            )
        ]
    )

    fig.show()

    # Correlation between severity and sales
    correlation, p_value = stats.pearsonr(df['severity'].dropna(), df['sales_open'].dropna())
    print(f'Correlation between weather severity and Day sales: {correlation:.2f} (p-value: {p_value:.5f})')
    correlation, p_value = stats.pearsonr(df['severity'].dropna(), df['sales_close'].dropna())
    print(f'Correlation between weather severity and Night sales: {correlation:.2f} (p-value: {p_value:.5f})')

    # ANOVA test to check if sales are significantly different across severity levels
    anova_result = stats.f_oneway(*[df['sales_open'][df['severity'] == sev].dropna() for sev in df['severity'].unique()])
    print(f'ANOVA result for day: F-statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.5f}')
    anova_result = stats.f_oneway(*[df['sales_close'][df['severity'] == sev].dropna() for sev in df['severity'].unique()])
    print(f'ANOVA result for night: F-statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.5f}')

except psycopg2.Error as e:
    print(f'Error: {e}')

finally:
    # Ensure cursor and connection are closed properly
    if 'cur' in locals():
        cur.close()
    if 'con' in locals():
        con.close()
    print('PostgreSQL connection closed')
