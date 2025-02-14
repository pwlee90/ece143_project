import pandas as pd
import math
# Define the function to calculate Popularity from Rank
def calculate_popularity(rank, n=2):
    # Popularity formula with max popularity set to 95
    return math.floor(95 - (95 - 30) * ((rank - 1) / 99) ** n)

# Read the CSV file into a DataFrame
input_file = 'C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data1.csv'  # Path to your input CSV file
df = pd.read_csv(input_file)

# Ensure the 'Rank' column exists in the DataFrame
if 'Rank' in df.columns:
    # Apply the formula to calculate the Popularity column
    df['Popularity'] = df['Rank'].apply(lambda rank: calculate_popularity(rank, n=2))

    # Drop the original 'Rank' column (optional)
    df = df.drop(columns=['Rank'])

    # Save the updated DataFrame to a new CSV file
    output_file = 'C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data1_updated.csv'  # Path to save the updated CSV file
    df.to_csv(output_file, index=False)
    print(f"Updated file saved as {output_file}")
else:
    print("The 'Rank' column was not found in the input file.")
