import pandas as pd
import ast

# Load your CSV file
input_file = 'C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data2.csv'  # Your input CSV file name
output_file = 'C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data2_updated.csv'  # New CSV file with the popularity column
df = pd.read_csv(input_file)

# Function to extract the artist name from the dictionary-like string
def extract_artist(artist_str):
    try:
        # Convert the string to a dictionary
        artist_dict = ast.literal_eval(artist_str)
        # Return the artist name (assuming the dictionary has one entry)
        return list(artist_dict.values())[0]
    except:
        return artist_str  # In case the format isn't as expected

# Apply the function to the 'artists' column
df['artists'] = df['artists'].apply(extract_artist)

# Save the modified CSV
df.to_csv(output_file, index=False)