# ECE 143 Group Project

## Problem Statement

TODO

## Environment Setup
To setup your environment, create a new local venv of your choosing (we recommend Pip or Anaconda).

Next, open the root directory of the project, where you should see [requirements.txt](./requirements.txt).

To install all necessary packages, run
```bash
python -m pip install -r requirements.txt
```

Our work is done from the command line using the IPython interpreter. To launch an instance of IPython, run:
```bash
python -m IPython
```

Due to Github file size limits, we recommend downloading the dataset's CSV files locally and adding them to .gitignore.

## Dataset Selection

**[SpotGenTrack](https://data.mendeley.com/datasets/4m2x4zngny/1)**
TODO Explain dataset selection

*lyrics_features.csv*

- 

*spotify_albums.csv*

- 

*spotify_tracks.csv*

- 

*spotify_artists.csv*

-

*low_level_audio_features.csv*

- 

**Note:** This dataset can be downloaded from [SpotGenTrack](https://data.mendeley.com/datasets/4m2x4zngny/1) and should be placed in the following directory structure:

- ece143 project root
    - data
        - low_level_audio_features.csv
        - lyrics_features.csv
        - spotify_albums.csv
        - spotify_artists.csv
        - spotify_tracks.csv
    - README.md
    - Source
        - DataProcessing.py
        - EDA.py
        - etc.
    - DataVisualizations.ipynb
    - requirements.txt

## Data Processing

**To skip over this section and load the merged and cleaned data into `cleaned_data` run**
```IPython
run DataProcessing.py
```

### Merging Data 
After selecting our datasets, we decided to combine them all into a larger meta-dataset. We were able to do this by first loading individual the CSV files into individual DataFrames in Pandas. To prepare these datasets for merging, we renamed the `id` columns in `spotify_artists.csv`, `spotify_albums.csv`, and `spotify_tracks.csv` to `artist_id`, `album_id`, and `track_id` respectively.

After doing so, each of the five CSV files contained the column `track_id`, so we were able to merge them using an inner join into a meta-dataset which we saved as `cleaned_data.csv`.

We then set the index as track_id so that each song can be easily accessed through its unique track id.

To execute our data merging code and load the merged datasets into a DataFrame `merged_data`, from an IPython terminal running in the project root directory, run

```IPython
import DataProcessing
merged_data = DataProcessing.merge_src_files()
```

### Drop Unneccessary Columns

From the merged DataFrame, we have identified the following columns as irrelevant to determining the relationship between the raw audio of a song and its popularity:

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
        "artist_id".

To execute this, run
```IPython
cleaned_data = DataProcessing.drop_irrelevant_columns(merged_data)
```

### Date Extraction

In our resultant merged and cleaned DataFrame, we have two columns `release_date` and `release_date_precison`. Using these columns, we convert the string representation of the date and given precison of `day`, `month`, or `year` to convert each release date into a `datetime.date` object. Any songs that do not have a complete qualification on date (i.e. day, month, and year) are dropped.

```IPython
cleaned_data = DataProcessing.format_release_data(cleaned_data)
```

### Genre Extraction

The genres are stored by default as a csv-like string of genres which makes it difficult to filter based off genre. So, to simplify this filtering process we refactor the `genre` column to contain a list of strings, with each string representing a genre.

```IPython
cleaned_data = DataProcessing.format_genres(cleaned_data)
```

## NLP Lyric Feature Engineering

TODO

## Explorartory Data Analysis

TODO

## Feature Selection

## Summary of Results