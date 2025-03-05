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


def load_cleaned_data():
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


CorrelationResult = namedtuple(
    "CorrelationResult",
    ["Column1", "Column2", "Correlation", "P_value", "Significance"],
)
""" Named tuple to store correlation results between two columns. """


def compute_correlation(df: pd.DataFrame, col1: str, col2: str):
    """
    Computes and returns the Pearson correlation coefficient and its p-value.
    """
    x = df[col1].dropna()
    y = df[col2].dropna()

    # Ensure both columns have the same length after dropping NaNs
    common_idx = x.index.intersection(y.index)
    x, y = x.loc[common_idx], y.loc[common_idx]

    if len(x) > 1:  # Ensure we have at least two data points
        corr, p_value = stats.pearsonr(x, y)
        return corr, p_value
    else:
        return None, None  # Not enough data to compute correlation


def setup_output_dir(output_dir):
    """
    Creates the output directory if it does not exist.
    Returns the output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def is_significant(p_value, alpha=0.05):
    """
    Determines if the p-value is significant based on a given alpha threshold and returns a string,
    either "Significant" or "Not Significant".
    """
    significance = "Significant" if p_value < alpha else "Not Significant"
    return significance


def plot_correlation(df: pd.DataFrame, column1: str, column2: str):
    """
    Given a DataFrame and two column names, this function plots a scatter plot of the two columns.
    """

    sns.regplot(
        x=df[column1],
        y=df[column2],
        scatter=False,
        line_kws={"color": "red"},
    )
    sns.kdeplot(
        x=df[column1], 
        y=df[column2], 
        cmap="Blues", 
        fill=True, 
        alpha=0.6
    )
    plt.title(f"{column1} vs {column2}\n Correlation: {df[column1].corr(df[column2])}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()


def report_correlation(df: pd.DataFrame, ref_column: str) -> pd.DataFrame:
    """
    Given a DataFrame, this function computes the correlation between a ref_column and all other columns in the dataset.
    It returns the correlation results as a DataFrame.
    """

    # Collect numerical correlation results
    numeric_columns = df.select_dtypes(include=["number"]).columns
    correlation_results = []

    for col in numeric_columns:
        if col != ref_column:  # Avoid self-correlation
            corr, p_value = compute_correlation(df, col, ref_column)

            if corr is not None:
                alpha = 0.05  # Significance level
                significance = is_significant(p_value, alpha)
                result = CorrelationResult(col, ref_column, corr, p_value, significance)
                correlation_results.append(result)
            else:
                print(f"Insufficient data to compute correlation for column '{col}'")

    # Sort results by statistical significance (p-value)
    correlation_results.sort(key=lambda x: abs(x.Correlation), reverse=True)

    # Save the correlation results to a DataFrame
    correlation_df = pd.DataFrame(
        correlation_results, columns=CorrelationResult._fields
    )
    return correlation_df


def determine_null_counts(df):
    """
    Returns a DataFrame containing the proportion of null counts for each column in the given DataFrame.
    Saves the output as a CSV in the 'eda_data' directory with column names preserved.
    """
    null_proportions = df.isna().mean()

    return null_proportions


def get_genre_subset(df: pd.DataFrame, keyword: str):
    """
    Returns a DataFrame containing only the rows that contain the given genre.
    """
    return df[
        df["genres"].apply(
            lambda genre_list: any(keyword in genre for genre in genre_list)
        )
    ]


def get_timeframe_subset(df: pd.DataFrame, start_date: str, end_date: str):
    """
    Returns a DataFrame containing only the rows that fall within the given timeframe.
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    return df[(df["release_date"] >= start_date) & (df["release_date"] <= end_date)]


def list_all_genres(df: pd.DataFrame):
    """
    Returns a dict containing the frequency of each genre in the given DataFrame.
    """
    genres = Counter()
    for genre_list in df["genres"]:
        genres.update(genre_list)
    return dict(sorted(genres.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    data = load_cleaned_data()
    correlations = report_correlation(data, "popularity")

    # Example examination of the correlation results
    twenty_tens = get_timeframe_subset(data, "2010-01-01", "2019-12-31")
    twenty_tens_correlations = report_correlation(twenty_tens, "popularity")
    print("Twenty tens correlations")
    print(twenty_tens_correlations)
    plot_correlation(twenty_tens, "entropy_energy", "popularity")

    hip_hop_data = get_genre_subset(data, "hip hop")
    hip_hop_correlations = report_correlation(hip_hop_data, "popularity")
    print("Hip hop correlations")
    print(hip_hop_correlations)
    plot_correlation(hip_hop_data, "popularity", "danceability")
