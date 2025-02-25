<<<<<<< HEAD
import pandas as pd
import numpy as np
from math import floor

# Function to calculate popularity based on weighted average of Views and Likes
"""def weighted_popularity(views, likes, weight_views=0.3, weight_likes=0.7):
    # Normalize the views and likes to a range between 0 and 1
    min_views = np.min(views)
    max_views = np.max(views)
    min_likes = np.min(likes)
    max_likes = np.max(likes)

    # Normalize Views and Likes
    normalized_views = [(v - min_views) / (max_views - min_views) for v in views]
    normalized_likes = [(l - min_likes) / (max_likes - min_likes) for l in likes]
    
    # Weighted average of normalized views and likes
    combined_score = [weight_views * v + weight_likes * l for v, l in zip(normalized_views, normalized_likes)]
    
    # Normalize combined score to a range between 5 and 95
    min_score = np.min(combined_score)
    max_score = np.max(combined_score)
    popularity = [(95 - 5) * (score - min_score) / (max_score - min_score) + 5 for score in combined_score]
    
    # Apply floor to round popularity scores down to the nearest integer
    popularity = [floor(score) if not np.isnan(score) else 1 for score in popularity]
    print(popularity)
    return popularity
"""

def log_popularity(views, likes):
    # Apply logarithmic scaling to views and likes
    log_views = np.log1p(views)  # log1p ensures that log(0) is handled
    log_likes = np.log1p(likes)
    
    # Combine them (e.g., average of log-transformed values)
    combined_log = [(lv + ll) / 2 for lv, ll in zip(log_views, log_likes)]
    
    # Normalize the combined log scores to a range between 5 and 95
    min_log = np.min(combined_log)
    max_log = np.max(combined_log)
    popularity = [(95 - 5) * (score - min_log) / (max_log - min_log) + 5 for score in combined_log]
    popularity = [floor(score) for score in popularity]
    return popularity

# Function to read the CSV, calculate popularity, remove description column, and save the updated CSV
def add_popularity_column(input_file, output_file, weight_views=0.3, weight_likes=0.7):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Drop rows where both Views and Likes are missing or NaN
    df = df.dropna(subset=['Views', 'Likes'], how='all')

    # Set Likes = 0 if Views is present but Likes is missing
    df['Likes'] = df.apply(lambda row: 0 if pd.isna(row['Likes']) else row['Likes'], axis=1)
    
    # Set Views = Likes if only Likes is present (i.e., Views is missing)
    df['Views'] = df.apply(lambda row: row['Likes'] if pd.isna(row['Views']) else row['Views'], axis=1)

    # Drop the 'Description' column if it exists
    if 'Description' in df.columns:
        df = df.drop(columns=['Description'])
    
    # Extract Views and Likes columns
    views = df['Views'].tolist()
    likes = df['Likes'].tolist()

    # Calculate the popularity score
    popularity_scores = log_popularity(views, likes)
    
    # Add popularity as a new column to the dataframe
    df['Popularity'] = popularity_scores
    
    # Save the updated dataframe to a new CSV file
    df.to_csv(output_file, index=False)

# Example usage
input_file = 'C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data4.csv'  # Your input CSV file name
output_file = 'C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data4_updated.csv'  # New CSV file with the popularity column

# Call the function to add the popularity column
add_popularity_column(input_file, output_file)
=======
import pandas as pd
import numpy as np
from math import floor

# Function to calculate popularity based on weighted average of Views and Likes
"""def weighted_popularity(views, likes, weight_views=0.3, weight_likes=0.7):
    # Normalize the views and likes to a range between 0 and 1
    min_views = np.min(views)
    max_views = np.max(views)
    min_likes = np.min(likes)
    max_likes = np.max(likes)

    # Normalize Views and Likes
    normalized_views = [(v - min_views) / (max_views - min_views) for v in views]
    normalized_likes = [(l - min_likes) / (max_likes - min_likes) for l in likes]
    
    # Weighted average of normalized views and likes
    combined_score = [weight_views * v + weight_likes * l for v, l in zip(normalized_views, normalized_likes)]
    
    # Normalize combined score to a range between 5 and 95
    min_score = np.min(combined_score)
    max_score = np.max(combined_score)
    popularity = [(95 - 5) * (score - min_score) / (max_score - min_score) + 5 for score in combined_score]
    
    # Apply floor to round popularity scores down to the nearest integer
    popularity = [floor(score) if not np.isnan(score) else 1 for score in popularity]
    print(popularity)
    return popularity
"""

def log_popularity(views, likes):
    # Apply logarithmic scaling to views and likes
    log_views = np.log1p(views)  # log1p ensures that log(0) is handled
    log_likes = np.log1p(likes)
    
    # Combine them (e.g., average of log-transformed values)
    combined_log = [(lv + ll) / 2 for lv, ll in zip(log_views, log_likes)]
    
    # Normalize the combined log scores to a range between 5 and 95
    min_log = np.min(combined_log)
    max_log = np.max(combined_log)
    popularity = [(95 - 5) * (score - min_log) / (max_log - min_log) + 5 for score in combined_log]
    popularity = [floor(score) for score in popularity]
    return popularity

# Function to read the CSV, calculate popularity, remove description column, and save the updated CSV
def add_popularity_column(input_file, output_file, weight_views=0.3, weight_likes=0.7):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Drop rows where both Views and Likes are missing or NaN
    df = df.dropna(subset=['Views', 'Likes'], how='all')

    # Set Likes = 0 if Views is present but Likes is missing
    df['Likes'] = df.apply(lambda row: 0 if pd.isna(row['Likes']) else row['Likes'], axis=1)
    
    # Set Views = Likes if only Likes is present (i.e., Views is missing)
    df['Views'] = df.apply(lambda row: row['Likes'] if pd.isna(row['Views']) else row['Views'], axis=1)

    # Drop the 'Description' column if it exists
    if 'Description' in df.columns:
        df = df.drop(columns=['Description'])
    
    # Extract Views and Likes columns
    views = df['Views'].tolist()
    likes = df['Likes'].tolist()

    # Calculate the popularity score
    popularity_scores = log_popularity(views, likes)
    
    # Add popularity as a new column to the dataframe
    df['Popularity'] = popularity_scores
    
    # Save the updated dataframe to a new CSV file
    df.to_csv(output_file, index=False)

# Example usage
input_file = 'C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data4.csv'  # Your input CSV file name
output_file = 'C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data4_updated.csv'  # New CSV file with the popularity column

# Call the function to add the popularity column
add_popularity_column(input_file, output_file)
>>>>>>> 65036df0fa401b2f794301a7f65372ab463333b2
