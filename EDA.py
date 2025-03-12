"""
This file is where we conducted our EDA. For our EDA, we load in the combined data file and
then we calculate the correlation between the popularity and the other columns.

Authors: Henri Schulz, Karl Hernandez
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import namedtuple
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
from datetime import datetime
import ast
from collections import Counter
from typing import List

def load_cleaned_data() -> pd.DataFrame:
    """
    Loads the cleaned data from the CSV file.
    """
    assert os.path.exists(
        "data/cleaned_data_with_emotions.csv"
    ), "The cleaned data file does not exist at './data/cleaned_data_with_emotions.csv'."
    data = pd.read_csv("data/cleaned_data_with_emotions.csv")
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

def plot_top_ten_features(correlation: pd.Series, correlation_name: str) -> List[str]:
    """Plots the top 10 correlated features as a horizontal bar plot."""
    assert len(correlation) >= 10, "The correlation Series must have at least 10 elements."

    top_10_features = correlation.index[:10]

    print(f"Top 10 Features for {correlation_name} Correlation:")
    print(correlation[:10])

    sns.set_style("whitegrid")
    pastel_colors = sns.color_palette("pastel")

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(y=top_10_features, x=correlation[:10], hue=top_10_features, palette=pastel_colors, legend=False)

    plt.title(f'Top 10 Features for {correlation_name} Correlation', fontsize=14, fontweight='bold')
    plt.xlabel(f'{correlation_name} Correlation', fontsize=12)
    plt.ylabel("Features", fontsize=12)

    # Add value labels inside bars
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2), 
                    ha='left', va='center', fontsize=10, color='black')

    plt.show()

    return top_10_features.tolist()

def generate_emotional_profile(df, title="Emotional Profile", genre=None):
    """
    Generates and displays an emotional profile visualization from song sentiment features.
    
    Parameters:
        df (DataFrame): The dataframe containing song data with sentiment features
        title (str): Title for the visualization
        genre (str): Optional genre to filter by (None for all songs)
        
    Returns:
        None (displays visualizations)
    """
    # Define the sentiment features to visualize
    sentiment_features = [
        'negative', 'disgust', 'fear', 'sadness', 'positive', 
        'anger', 'trust', 'anticipation', 'joy', 'surprise'
    ]

    # Filter dataframe if needed
    filtered_df = df.copy()
    filter_description = []

    if genre:
        filtered_df = get_genre_subset(df, genre)
        filter_description.append(f"Emotional Profile of {genre}")

        
    # Make sure we have all the needed features and data
    available_features = [f for f in sentiment_features if f in filtered_df.columns]
    if not available_features:
        print("Error: No sentiment features found in the dataframe")
        return
        
    if len(filtered_df) == 0:
        print("Error: No songs match the specified filters")
        return
    
    # Calculate average sentiment values
    sentiment_means = filtered_df[available_features].mean().values
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 10))
    
    # Create the main radar chart
    ax = fig.add_subplot(111, polar=True)
    
    # Set up angles for each feature
    angles = np.linspace(0, 2*np.pi, len(available_features), endpoint=False).tolist()
    
    # Complete the loop for the radar chart
    sentiment_means = np.append(sentiment_means, sentiment_means[0])
    angles += angles[:1]
    
    # Plot radar chart
    ax.plot(angles, sentiment_means, 'o-', linewidth=2, color='#3274A1')
    ax.fill(angles, sentiment_means, alpha=0.25)
    
    # Set labels and styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_features, size=12)
    ax.set_yticklabels([])
    ax.grid(True)
    
    # Create descriptive title
    if filter_description:
        full_title = f"{title}\n({', '.join(filter_description)}, n={len(filtered_df)})"
    else:
        full_title = f"{title}\n(n={len(filtered_df)})"
    
    ax.set_title(full_title, size=16, pad=20, fontweight='bold')
    
    # Display the plot
    plt.tight_layout()
    plt.show()
    
    return


if __name__ == "__main__":
    data = load_cleaned_data()

    # Plot the top 5 correlations in hip hop
    hip_hop_data = get_genre_subset(data, "hip hop")
    pearson, spearman = report_correlation(hip_hop_data, "popularity")
    for i in range(5):
        plot_correlation(hip_hop_data, pearson.index[i], "popularity")
        input("Press Enter to display the next plot...")

