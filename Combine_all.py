import pandas as pd

# Read the CSV files
file1 = pd.read_csv(
    "C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data1_updated.csv"
)
file2 = pd.read_csv(
    "C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data2_updated.csv"
)
file3 = pd.read_csv(
    "C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data3_updated.csv"
)
file4 = pd.read_csv(
    "C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/new_data4_updated.csv"
)

# Rename columns to make them consistent across all files
# file4.rename(columns={'Track': 'song_name'}, inplace=True)
file2.rename(columns={"artists": "Artist", "song_type": "Song_type"}, inplace=True)
# file1.rename(columns={'Song Title': 'song_name'}, inplace=True)
file3.rename(
    columns={
        "acousticness": "Acousticness",
        "available_markets": "Available_markets",
        "lyrics": "Lyrics",
        "country": "Country",
        "danceability": "Danceability",
        "duration_ms": "Duration_ms",
        "instrumentalness": "Instrumentalness",
        "key": "Key",
        "liveness": "Liveness",
        "loudness": "Loudness",
        "mode": "Mode",
        "speechiness": "Speechiness",
        "tempo": "Tempo",
        "time_signature": "Time_signature",
        "type": "Song_type",
    },
    inplace=True,
)
# Merge all the files on the common columns
merged_df = pd.concat([file1, file2, file3, file4], axis=0, ignore_index=True)

# Combine columns 'song_name', 'name', 'Track', 'Song Title' into one column
merged_df["song_name"] = (
    merged_df[["song_name", "name", "Track", "Song Title"]].bfill(axis=1).iloc[:, 0]
)
merged_df.drop(columns=["name", "Track", "Song Title"], inplace=True)

merged_df["Popularity"] = (
    merged_df[["Popularity", "popularity"]].bfill(axis=1).iloc[:, 0]
)
merged_df.drop(columns=["popularity"], inplace=True)

# Group by 'song_name' and aggregate by filling NaN values with the available data
merged_df = merged_df.groupby("song_name", as_index=False).first()

# Now you can save the merged CSV file
merged_df.to_csv(
    "C:/Users/Karmanya Pandey/Desktop/UCSD/Winter Quarter 24/ECE 143/Project/data_files/combine_all.csv",
    index=False,
)
