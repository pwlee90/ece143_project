"""
This file contains a series of functions used to extract, clean, and process the data.
The functions are used to merge the source CSV files, drop irrelevant columns, extract datetime information, and extract genres.

When the program is run as main, the data is loaded, cleaned, and saved in a new CSV file.
Author: Henri Schulz
"""

import pandas as pd
from functools import reduce
import ast


def merge_src_files() -> pd.DataFrame:
    """
    Merges the source CSV files into a single DataFrame.
    """

    # Load CSVs
    lyrics = pd.read_csv("data/lyrics_features.csv")
    audio = pd.read_csv("data/low_level_audio_features.csv")
    albums = pd.read_csv(
        "data/spotify_albums.csv"
    )  # needs id to be renamed to album_id
    artists = pd.read_csv(
        "data/spotify_artists.csv"
    )  # needs id to be renamed to artist_id
    tracks = pd.read_csv("data/spotify_tracks.csv")  # Contains id instead of track_id

    datasets = [lyrics, audio, albums, artists, tracks]
    for dataset in datasets:
        dataset.drop(columns=["Unnamed: 0"], inplace=True)

    # Rename columns to match the schema
    albums.rename(columns={"id": "album_id", "name": "album_name"}, inplace=True)
    artists.rename(columns={"id": "artist_id", "name": "artist_name"}, inplace=True)
    tracks.rename(columns={"id": "track_id", "name": "track_name"}, inplace=True)

    # Merge albums, artists, and tracks
    merged_data = reduce(
        lambda left, right: pd.merge(left, right, on="track_id", how="inner"),
        [tracks, albums, artists, audio, lyrics],
    )

    # Remove _x, _y suffixes from the merge
    merged_data.columns = merged_data.columns.str.replace("_x", "")
    merged_data.columns = merged_data.columns.str.replace("_y", "")

    # Reset the index
    merged_data.index = merged_data["track_id"].reset_index(drop=True)

    return merged_data


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops irrelevant columns from the merged data DataFrame.
    """
    irrelevant_columns = [
        "album_id",
        "available_markets",
        "country",
        "type",
        "mode",
        "analysis_url",
        "artists_id",
        "disc_number",
        "href",
        "playlist",
        "preview_url",
        "track_href",
        "track_name_prev",
        "track_number",
        "uri",
        "artist_id",
        "external_urls",
        "href",
        "album_id",
        "images",
        "track_name_prev",
        "total_tracks",
        "uri",
        "artist_popularity",
        "followers",
        "artist_id",
    ]

    return df.drop(columns=irrelevant_columns)


def format_release_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a datetime column to the DataFrame using the release_date and release_precison columns.
    Any song that does not contain day, month, or year information will be dropped.
    """
    if "release_date" not in df.columns:
        print("release_date column not found in the DataFrame")
        return df
    if "release_date_precision" not in df.columns:
        print("release_date_precision column not found in the DataFrame")
        return df

    fully_specified_dates = df[df["release_date_precision"] == "day"].copy()
    fully_specified_dates["release_date"] = pd.to_datetime(
        fully_specified_dates["release_date"], format="%Y-%m-%d"
    )
    fully_specified_dates.drop(columns=["release_date_precision"], inplace=True)
    return fully_specified_dates


def format_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updates the genre column for each song to contain a list of genres.
    """
    if "genres" not in df.columns:
        print("genres column not found in the DataFrame")
        return df

    df["genres"] = df["genres"].apply(
        lambda genre_string: (
            ast.literal_eval(genre_string)
            if ast.literal_eval(genre_string) != [""]
            else []
        )
    )
    return df


if __name__ == "__main__":
    merged_data = merge_src_files()
    cleaned_data = drop_irrelevant_columns(merged_data)
    cleaned_data = format_release_date(cleaned_data)
    cleaned_data = format_genres(cleaned_data)
    cleaned_data.to_csv("data/cleaned_data.csv")
    print("Cleaned data saved in data/cleaned_data.csv")
