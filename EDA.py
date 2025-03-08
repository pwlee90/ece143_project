"""
This file is where we conducted our EDA. For our EDA, we load in the combined data file and
then we calculate the correlation between the popularity and the other columns.

Author: Henri Schulz
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
from datetime import datetime
import ast
from collections import Counter
import typing

def load_cleaned_data() -> pd.DataFrame:
    """
    Loads the cleaned data from the CSV file.
    """
    assert os.path.exists(
        "data/cleaned_data.csv"
    ), "The cleaned data file does not exist at './data/cleaned_data.csv'."
    data = pd.read_csv("data/cleaned_data.csv")
    data["release_date"] = pd.to_datetime(data["release_date"])
    data["genres"] = data["genres"].apply(ast.literal_eval)
    return data


def setup_output_dir(output_dir : str) -> str:
    """
    Creates the output directory if it does not exist.
    Returns the output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_correlation(df: pd.DataFrame, column1: str, column2: str) -> None:
    """
    Given a DataFrame and two column names, this function plots a scatter plot of the two columns.
    """

    plt.figure()
    sns.regplot(
        x=df[column1],
        y=df[column2],
        scatter=False,
        line_kws={"color": "red"},
    )
    plt.hexbin(
        x=df[column1],
        y=df[column2],
        gridsize=50,
        cmap="Blues",
        alpha=0.8,
    )
    sns.kdeplot(
        x=df[column1], 
        y=df[column2],
        cmap="Greys",
        fill=False, 
        alpha=0.4
    )
    plt.title(f"{column1} vs {column2}\n Pearson Correlation: {df[column1].corr(df[column2])}\n Spearman Correlation: {df[column1].corr(df[column2], method='spearman')}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show(block=False)


import pandas as pd

def report_correlation(df: pd.DataFrame, ref_column: str) -> tuple[pd.Series, pd.Series]:
    """
    Given a DataFrame, this function computes the correlation between a ref_column and all other columns in the dataset.
    It returns the correlation results as two Series: Pearson and Spearman correlations.
    """

    # Extract numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Compute Pearson correlation
    pearson_correlation = df[numeric_columns].corrwith(df[ref_column]).drop(ref_column).abs().sort_values(ascending=False)

    # Compute Spearman correlation
    spearman_correlation = df[numeric_columns].corrwith(df[ref_column], method="spearman").drop(ref_column).abs().sort_values(ascending=False)

    return pearson_correlation, spearman_correlation


def determine_null_counts(df : pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing the proportion of null counts for each column in the given DataFrame.
    Saves the output as a CSV in the 'eda_data' directory with column names preserved.
    """
    null_proportions = df.isna().mean()

    return null_proportions


def get_genre_subset(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the rows that contain the given genre.
    """
    return df[
        df["genres"].apply(
            lambda genre_list: any(keyword in genre for genre in genre_list)
        )
    ]


def get_timeframe_subset(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the rows that fall within the given timeframe.
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    return df[(df["release_date"] >= start_date) & (df["release_date"] <= end_date)]


def list_all_genres(df: pd.DataFrame) -> dict:
    """
    Returns a dict containing the frequency of each genre in the given DataFrame.
    """
    genres = Counter()
    for genre_list in df["genres"]:
        genres.update(genre_list)
    return dict(sorted(genres.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    data = load_cleaned_data()

    # Plot the top 5 correlations in hip hop
    hip_hop_data = get_genre_subset(data, "hip hop")
    pearson, spearman = report_correlation(hip_hop_data, "popularity")
    for i in range(5):
        plot_correlation(hip_hop_data, pearson.index[i], "popularity")
        input("Press Enter to display the next plot...")

