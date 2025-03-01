"""
This file is where we conducted our EDA. For our EDA, we load in the combined data file and
then we calculate the correlation between the popularity and the other columns.

These correlation results are saved in a CSV file in the folder eda_data.

We then plot the relationship between the popularity and the other columns using hexbin plots. We then save
these plots in the folder eda_plots.

Author: Henri Schulz
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns

CorrelationResult = namedtuple(
    "CorrelationResult",
    ["Column1", "Column2", "Correlation", "P_value", "Significance"],
)
""" Named tuple to store correlation results between two columns. """


def compute_correlation(df : pd.DataFrame, col1 : str, col2 : str):
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

def generate_correlation_plot(df: pd.DataFrame, corr_result, output_dir, hue=None):
    """
    Generates and saves hexbin plot given a DataFrame and a correlation result 
    derived from two columns in that DataFrame. Supports hue-based segmentation.
    """
    print(f"Generating correlation plot for {corr_result.Column1} vs {corr_result.Column2}")

    fig, ax = plt.subplots(figsize=(7, 5))

    if hue is not None:
        hue_values = df[hue].unique()  # Get unique hue categories
        colors = plt.cm.get_cmap("tab10", len(hue_values))  # Generate a colormap with distinct colors
        
        for i, val in enumerate(hue_values):
            subset = df[df[hue] == val]  # Filter dataset by hue category
            
            hb = ax.hexbin(
                subset[corr_result.Column1], 
                subset[corr_result.Column2], 
                gridsize=30, 
                mincnt=1, 
                alpha=0.5,  # Transparency for overlap visibility
                edgecolors="none",  # Removes hex edges
            )
            # Manually set color
            hb.set_facecolor(colors(i))  # Assign a color from colormap
    else:
        hb = ax.hexbin(
            df[corr_result.Column1], df[corr_result.Column2], 
            cmap="Blues", mincnt=1
        )
        plt.colorbar(hb, ax=ax).set_label("Count in bin")

    ax.set_xlabel(corr_result.Column1)
    ax.set_ylabel(corr_result.Column2)
    ax.set_title(
        f"{corr_result.Column1} vs. {corr_result.Column2} \n"
        f"Corr: {corr_result.Correlation:.3f}, p: {corr_result.P_value:.3g} ({corr_result.Significance})"
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, f"hexbin_{corr_result.Column1}_{corr_result.Column2}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def compute_raw_correlation(df : pd.DataFrame, ref_column : str, hue=None) -> pd.DataFrame:
    """
    Given a DataFrame, this function computes the correlation between the popularity and all other columns in the dataset.
    It saves the correlation results in a CSV file and generates and saves hexbin plots for each correlation.
    It returns the correlation results as a DataFrame.
    """
    # Load the combined data
    combined_data = df

    # Collect numerical correlation results
    numeric_columns = combined_data.select_dtypes(include=["number"]).columns
    correlation_results = []

    for col in numeric_columns:
        if col != ref_column:  # Avoid self-correlation
            print("Processing:", col)
            corr, p_value = compute_correlation(combined_data, col, ref_column)

            if corr is not None:
                alpha = 0.05  # Significance level
                significance = is_significant(p_value, alpha)
                result = CorrelationResult(
                    col, ref_column, corr, p_value, significance
                )
                correlation_results.append(result)
            else:
                print(f"Insufficient data to compute correlation for column '{col}'")

    # Sort results by statistical significance (p-value)
    correlation_results.sort(key=lambda x: abs(x.Correlation), reverse=True)

    # Save the correlation results to a CSV file
    data_output_dir = setup_output_dir("eda_data")
    correlation_df = pd.DataFrame(
        correlation_results, columns=CorrelationResult._fields
    )
    correlation_df.to_csv(
        os.path.join(data_output_dir, "correlation_results.csv"), index=False
    )
    print(f"Correlation results saved in {data_output_dir}/")

    # Plot and save hexbin
    plot_dir = setup_output_dir("eda_plots/correlation_plots")
    for corr_result in correlation_results:
        generate_correlation_plot(combined_data, corr_result, plot_dir, hue)
    print(f"Plots saved in {plot_dir}/")

    return correlation_df

def K_means_cluster(df : pd.DataFrame, column : str, K:int):
    """
    Performs a K-means clustering of a column in a provided DataFrame
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    kmeans = KMeans(n_clusters=K)
    df[f'{column}_cluster'] = kmeans.fit_predict(df[[column]])
       
    for cluster in range(K):
        feature_cluster = df[column].where(df[f'{column}_cluster'] == cluster)
        feature_cluster.plot(kind='kde')
    plt.show()


def DBSCAN_cluster(df, column):
    """
    Performs a density-based clustering of a column in a provided DataFrame
    """
    dbscan = DBSCAN(eps=5) 
    df[f'{column} Cluster'] = dbscan.fit_predict(df[[column]])
    print("Done predicting")
       
    for cluster in df['f{column} Cluster'].unique_values:
        feature_cluster = df[column].where(df[f'{column} Cluster'] == cluster)
        feature_cluster.plot(kind='kde')
    plt.show()

def cluster_data(df:pd.DataFrame, column:str, K_range:tuple=(3,10)) -> None:
    """
    Cluster data according to a give column in the dataframe.
    Attempts to perform k-means clustering for each k value in K_range (low inclusive, high-non inclusive)
    Saves a kde plot of the distribution to eda_plots
    """
    # Create output directories
    plot_dir = setup_output_dir(f"eda_plots/{column}_clustering")
    data_dir = setup_output_dir("eda_data")

    # First, we will examine the distribution of the column using a kernel density plot
    df[column].describe().to_csv(f"{data_dir}/{column}_stats.csv")
    df[column].plot(kind='kde', title=f'{column} Distribution', legend=True)
    plt.show()
    plt.savefig(f"{plot_dir}/{column}_distribution.png")

    mean_popularity = df['popularity'].mean()
    std_popularity = df['popularity'].std()

    df['popularity_segment'] = pd.cut(df['popularity'], 
                                    bins=[-float('inf'), mean_popularity - std_popularity, 
                                            mean_popularity + std_popularity, float('inf')],
                                    labels=['Low', 'Average', 'High'])

   
    # DBSCAN_cluster(df, column)
    for k in range(K_range[0], K_range[1]):
        K_means_cluster(df, column, k)

def determine_null_counts(df):
    """
    Returns a DataFrame containing the proportion of null counts for each column in the given DataFrame.
    Saves the output as a CSV in the 'eda_data' directory with column names preserved.
    """
    null_proportions = df.isna().mean()

    return null_proportions  # Return the DataFrame for inspection
        
if __name__ == "__main__":
    data = pd.read_csv("data/cleaned_data.csv")

    # Cluster data according to popularity and view the distribution of the popularity column
    cluster_data(data, 'popularity', (0,0))

    # Generate correlation plots between popularity and other columns
    compute_raw_correlation(data, "popularity")

    


