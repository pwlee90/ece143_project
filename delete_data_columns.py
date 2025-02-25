from idlelib.squeezer import count_lines_with_wrapping

import pandas as pd
import re


def delete_data_columns(file: str, columns_to_drop: list, new_file: str):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    df = df.drop(columns=columns_to_drop)

    # Save the updated DataFrame back to a CSV file (if you want to overwrite it or create a new one)
    df.to_csv(new_file, index=False)
    print(df)


file = "data_files/data1.csv"
columns_to_drop = [
    "Album",
    "Album URL",
    "Featured Artists",
    "Media",
    "Song URL",
    "Writers",
]
new_file = "data_files/new_data1.csv"
delete_data_columns(file, columns_to_drop, new_file)
file = "data_files/data3.csv"
columns_to_drop = [
    "album_id",
    "analysis_url",
    "artists_id",
    "disc_number",
    "href",
    "id",
    "playlist",
    "preview_url",
    "track_href",
    "track_name_prev",
    "track_number",
    "uri",
    "valence",
]
new_file = "data_files/new_data3.csv"
delete_data_columns(file, columns_to_drop, new_file)
file = "data_files/data4.csv"
columns_to_drop = [
    "Url_spotify",
    "Track",
    "Album",
    "Album_type",
    "Uri",
    "Valence",
    "Url_youtube",
    "Channel",
    "Licensed",
    "official_video",
]
new_file = "data_files/new_data4.csv"
delete_data_columns(file, columns_to_drop, new_file)
