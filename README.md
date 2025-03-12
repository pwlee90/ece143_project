# ECE 143 Group Project - Music Popularity Prediction
---

## Problem Statement

**Is the popularity of songs predictable from just lyrical and audio features?**

## Environment Setup
To setup your environment, create a new local venv of your choosing (we recommend Pip or Anaconda).

Next, open the root directory of the project, where you should see [requirements.txt](./requirements.txt).

To install all necessary packages, run
```bash
python -m pip install -r requirements.txt
```

**We are using the following 3rd part packages:**
- Jupyter
- Numpy
- Pandas
- Scikit-Learn
- Scipy
- XGBoost
- IPython
- Seaborn
- Matplotlib

Our work is done from the command line using the IPython interpreter. To launch an instance of IPython, run:
```bash
python -m IPython
```

Due to Github file size limits, we recommend downloading the dataset's CSV files locally and adding them to .gitignore.

## Dataset Selection

**[SpotGenTrack](https://data.mendeley.com/datasets/4m2x4zngny/1)**
- Repository of data for 109,393
- Scraped from Spotify and Genius using public APIs
- Highlighted features:
    - Song name
    - Song artist
    - Song ID
    - Low level audio features
        - MEL
        - Chroma
        - MFCC
        - Spectral decomposition metrics
    - Lyrics
    - Popularity score (1-100)
    - Song duration (in ms)
    - Descriptive statistics
        - Danceability
        - Loudness
        - etc.

**Note:** This dataset can be downloaded from [SpotGenTrack](https://data.mendeley.com/datasets/4m2x4zngny/1) and should be placed in the following directory structure:

### Project Structure

- ece143 project root
    - README.md
    - requirements.txt
    - data
        - low_level_audio_features.csv
        - lyrics_features.csv
        - spotify_albums.csv
        - spotify_artists.csv
        - spotify_tracks.csv
        - cleaned_data.csv (generated via user code)
        - cleaned_data_with_emotions.csv (generate via user code)
    - DataProcessing.py
    - NLP.py
    - EDA.py
    - linear_regression.py
    - polynomial_regression.py
    - random_forest.py
    - gradient_boosting.py
    - ErrorAnalysis.py
    - DataVisualizations.ipynb
    - Project_Presentation_Slides.pdf

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

We extracted additonal sentiment-based features from the lyrics of the songs. Our process for doingn so can be seen in NLP.py, which outputs a new csv file `cleaned_data_with_emotions.csv` which can be found in the above directory structure.

## Exploratory Data Analysis (EDA)

Our (EDA.py)[EDA.py] serves as a reusable module for common EDA functions, including computing correlation scores, plotting correlations, and clustering by genres and timeframes. 

While most of our work was done in the IPython command-line interpreter, the main of EDA.py serves as an example of a potential usage of some of the module's functions.
