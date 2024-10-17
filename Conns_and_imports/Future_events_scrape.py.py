import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to scrape Hearts FC game schedule
def scrape_hearts_game_schedule():
    # URL of the Hearts FC fixtures page
    url = 'https://www.heartsfc.co.uk/pages/first-team-fixtures'
    
    # Send a GET request to fetch the page content
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all the relevant divs containing the date and location
        game_divs = soup.find_all('div', class_='lg:hidden max-lg:col-start-1 max-lg:row-start-3 flex flex-col justify-center max-lg:mt-2 text-base-color')
        
        # Lists to store the extracted data
        dates = []
        locations = []
        
        # Loop over each game div
        for game_div in game_divs:
            date_element = game_div.find('span', class_='text-xs leading-[12px]')
            location_element = game_div.find_all('span', class_='text-xs')  # Change to find_all to capture multiple spans
            
            # Debugging: Print the raw HTML of the game div
            print(game_div.prettify())  # Print the prettified HTML of each game div
            
            # Extract text if both elements are found
            if date_element and location_element:
                dates.append(date_element.text.strip())  # Get date
                locations.append(location_element[1].text.strip())  # Get location (second span)
        
        # Convert the extracted data into a DataFrame
        games_df = pd.DataFrame({
            'Date': dates,
            'Location': locations
        })
        
        # Filter for only Tynecastle Park games
        tynecastle_games = games_df[games_df['Location'].str.contains('Tynecastle Park', case=False, na=False)]
        
        # Debugging: Print the DataFrame before filtering
        print("All Games DataFrame:")
        print(games_df)
        
        return tynecastle_games
    else:
        print(f"Failed to fetch the schedule. Status code: {response.status_code}")
        return None

# Example usage
tynecastle_games_df = scrape_hearts_game_schedule()

# Check the extracted data
if tynecastle_games_df is not None:
    print("Filtered Tynecastle Games DataFrame:")
    print(tynecastle_games_df)
    # You can save this data to a CSV file or insert it into a database
    tynecastle_games_df.to_csv('tynecastle_games_schedule.csv', index=False)
