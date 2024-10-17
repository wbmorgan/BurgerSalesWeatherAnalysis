import psycopg2
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import numpy as np


# Database connection params for PostgreSQL
db_params = {
    "host": "localhost",
    "database": "fgsales",
    "user": "postgres",
    "password": "postgres"
}

# Create output directory if it doesn't exist
output_directory = "/Users/willb-m/Desktop/output_files/"
os.makedirs(output_directory, exist_ok=True)

# Connect to PostgreSQL
try:
    con = psycopg2.connect(**db_params)
    cur = con.cursor()
    print("Connected to the PostgreSQL database successfully")

    # Execute SELECT query to retrieve information on daily precipitation, wind gust and wind speed and avg temp
    select_query = """
    SELECT 
        e.date, e.prcp, e.wpgt, e.wspd, e.tavg,  s.total_day 
    FROM 
        edinburgh_forecasts_2023 e
    JOIN 
        sales_23 s ON e.date = s.date;
    """
    cur.execute(select_query)

    # Fetch all rows from query
    rows = cur.fetchall()

    # Convert rows to pandas DataFrame
    df = pd.DataFrame(rows, columns=['date', 'prcp', 'wpgt', 'wspd', 'tavg', 'total_day'])
    print(df)

    # Change 'date' to datetime object for better plotting 
    df['date'] = pd.to_datetime(df['date'])

    # Rolling window to smooth data (over 7-day window)
    df['prcp_smooth'] = df['prcp'].rolling(window=7).mean()
    df['wpgt_smooth'] = df['wpgt'].rolling(window=7).mean()
    df['wspd_smooth'] = df['wspd'].rolling(window=7).mean()
    df['tavg_smooth'] = df['tavg'].rolling(window=7).mean()
    df['total_day_smooth'] = df["total_day"].rolling(window=7).mean()

    # Print first few rows to check 
    print(df.head())

  
    # Convert columns to numeric and set non numeric values to NaN
    df['prcp'] = pd.to_numeric(df['prcp'], errors='coerce')
    df['total_day'] = pd.to_numeric(df['total_day'], errors='coerce')
    df['wpgt'] = pd.to_numeric(df['wpgt'], errors = 'coerce')
    df['wspd'] = pd.to_numeric(df['wspd'], errors = 'coerce')
    df['tavg'] = pd.to_numeric(df['tavg'], errors = 'coerce')

      # Check for NaN values in columns
    print(df[['prcp','wpgt', 'wspd', 'tavg', 'total_day']].isna().sum())


    # Directly calculate the correlation 
    corrprcp = df['total_day'].corr(df['prcp'])
    corrwpgt = df['total_day'].corr(df['wpgt'])
    corrwspd = df['total_day'].corr(df['wspd'])
    corrtavg = df['total_day'].corr(df['tavg'])

    
    print("Correlation between 'total_day' and 'prcp':", corrprcp)
    print("Correlation between 'total_day' and 'wpgt':", corrwpgt)
    print("Correlation between 'total_day' and 'wspd':", corrwspd)
    print("Correlation between 'total_day' and 'tavg':", corrtavg)

    

    # Plot data for precipitation
    
    
    fig = go.Figure()

    # Add precipitation line plot with the primary y-axis
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['prcp_smooth'],
        mode='lines+markers',
        name='Precipitation (Smoothed)',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Avg Precipitatione: %{y:.2f}<extra></extra>',
        yaxis='y1'  # Primary y-axis
    ))

    # Add sales line plot with the secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_day_smooth'],
        mode='lines+markers',
        name='Sales (Smoothed)',
        marker=dict(color='red'),
        line=dict(color='red'),
        hovertemplate='Date: %{x}<br>Sales: %{y:.2f}<extra></extra>',
        yaxis='y2'  # Secondary y-axis
    ))

    # Customize layout with dual y-axis
    fig.update_layout(
        title='Precipitation and Daily Sales (Smoothed)',
        xaxis_title='Date',
        yaxis=dict(title='Precipitation', side='left'),  # Left y-axis for temperature
        yaxis2=dict(title='Sales (£)', side='right', overlaying='y'),  # Right y-axis for sales
        hovermode='x',
        template='plotly_white'        
    )
    fig.write_html(os.path.join(output_directory, 'precipitation_sales_plot.png'))

    fig.show()

    #Plot for avg Wind gust

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['wpgt_smooth'],
        mode='lines+markers',
        name='Wind Gust (Smoothed)',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Wind Gusts: %{y:.2f}<extra></extra>',
        yaxis='y1'  # Primary y-axis
    )) 

    # Add sales line plot with the secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_day_smooth'],
        mode='lines+markers',
        name='Sales (Smoothed)',
        marker=dict(color='red'),
        line=dict(color='red'),
        hovertemplate='Date: %{x}<br>Sales: %{y:.2f}<extra></extra>',
        yaxis='y2'  # Secondary y-axis
    ))

    # Customize layout with dual y-axis
    fig.update_layout(
        title='Wind Gusts and Daily Sales (Smoothed)',
        xaxis_title='Date',
        yaxis=dict(title='Wind Gusts', side='left'),  # Left y-axis for temperature
        yaxis2=dict(title='Sales (£)', side='right', overlaying='y'),  # Right y-axis for sales
        hovermode='x',
        template='plotly_white'
    )
    fig.write_html(os.path.join(output_directory, 'gust_sales_plot.png'))
    fig.show()

    #Plot for Wind Speed
    fig = go.Figure()

    # Add a line plot with the primary y-axis
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['wspd_smooth'],
        mode='lines+markers',
        name='Wind Speed (Smoothed)',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Wind Speed: %{y:.2f}<extra></extra>',
        yaxis='y1'  # Primary y-axis
    ))

    # Add sales line plot with the secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_day_smooth'],
        mode='lines+markers',
        name='Sales (Smoothed)',
        marker=dict(color='red'),
        line=dict(color='red'),
        hovertemplate='Date: %{x}<br>Sales: %{y:.2f}<extra></extra>',
        yaxis='y2'  # Secondary y-axis
    ))

    # Customize layout with dual y-axis
    fig.update_layout(
        title='Wind Speed and Daily Sales (Smoothed)',
        xaxis_title='Date',
        yaxis=dict(title='Wind Speed', side='left'),  # Left y-axis for temperature
        yaxis2=dict(title='Sales (£)', side='right', overlaying='y'),  # Right y-axis for sales
        hovermode='x',
        template='plotly_white'
    )
    fig.write_html(os.path.join(output_directory, 'wspd_sales_plot.png'))
    fig.show()

    #Plot for avg temperature
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['tavg_smooth'],
        mode='lines+markers',
        name='Average Temperature (Smoothed)',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Avg Temperature: %{y:.2f}<extra></extra>',
        yaxis='y1'  # Primary y-axis
    ))

    # Add sales line plot with the secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_day_smooth'],
        mode='lines+markers',
        name='Sales (Smoothed)',
        marker=dict(color='red'),
        line=dict(color='red'),
        hovertemplate='Date: %{x}<br>Sales: %{y:.2f}<extra></extra>',
        yaxis='y2'  # Secondary y-axis
    ))

    # Customize layout with dual y-axis
    fig.update_layout(
        title='Average Temperature and Daily Sales (Smoothed)',
        xaxis_title='Date',
        yaxis=dict(title='Avg Temperature (°C)', side='left'),  # Left y-axis for temperature
        yaxis2=dict(title='Sales (£)', side='right', overlaying='y'),  # Right y-axis for sales
        hovermode='x',
        template='plotly_white'
    )
    fig.write_html(os.path.join(output_directory, 'temp_sales_plot.png'))
    fig.show()



    # Create a list of weather factors and their corresponding correlations
    weather_factors = ['Precipitation', 'Wind Gust', 'Wind Speed', 'Avg Temperature']
    correlations = [corrprcp, corrwpgt, corrwspd, corrtavg]

    # Create a bar chart using Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weather_factors,
        y=correlations,
        text=[f'{corr:.2f}' for corr in correlations],  # Add correlation values as text labels
        textposition='auto',
        marker=dict(color=['blue', 'green', 'purple', 'orange']),  # Different colors for each bar
        name='Correlation'
    ))

    # Customize the layout of the chart
    fig.update_layout(
        title='Correlation between Weather Factors and Sales',
        xaxis_title='Weather Factors',
        yaxis_title='Correlation Coefficient',
        yaxis=dict(range=[-1, 1]),  # Set y-axis to range from -1 to 1 to reflect correlation bounds
        template='plotly_white'
    )
    fig.write_html(os.path.join(output_directory, 'weatherfactor_sales_plot.png'))
    # Show the plot
    fig.show()



    # Prepare the data for quadratic regression
    X = df[['prcp']].values
    y = df['total_day'].values

    # Create polynomial features (quadratic)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Fit the quadratic regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict values using the quadratic model
    y_pred = model.predict(X_poly)

    # Plot the results
    plt.scatter(df['prcp'], df['total_day'], color='blue', label='Data')
    plt.plot(df['prcp'], y_pred, color='red', label='Quadratic Fit')
    plt.title('Sales vs Precipitation (Quadratic Regression)')
    plt.savefig(os.path.join(output_directory, 'sales_vs_precipitation.png'))  # Save Matplotlib figure as PNG
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Sales (£)')
    plt.legend()
    plt.show()
    plt.close()  # Close the plot to free memory

    # Coefficients of the model (for interpretation)
    intercept = model.intercept_
    linear_coeff = model.coef_[1]  # Coefficient for linear term
    quadratic_coeff = model.coef_[2]  # Coefficient for quadratic term

    # Create a DataFrame to save the results
    results_df = pd.DataFrame({
        'Value': [intercept, linear_coeff, quadratic_coeff, corrprcp, corrwpgt, corrwspd, corrtavg]
    }, index=['Intercept', 'Linear Coefficient', 'Quadratic Coefficient', 'Precipitation Correlation', 'Wind Gust Correlation', 'Wind Speed Correlation', 'Avg Temp Correlation'])

    # Save results to CSV
    results_df.to_csv(os.path.join(output_directory, 'regression_results.csv'))










except psycopg2.Error as e:
    print(f'Error: {e}')  # Corrected error printing

finally:
    # Ensure cursor and connection are closed properly
    if 'cur' in locals():
        cur.close()
    if 'con' in locals():
        con.close()
    print('PostgreSQL connection closed')
