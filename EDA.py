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

CorrelationResult = namedtuple(
    "CorrelationResult",
    ["Column1", "Column2", "Correlation", "P_value", "Significance"],
)
""" Named tuple to store correlation results between two columns. """


def compute_correlation(df, col1, col2):
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


def generate_correlation_plot(df, corr_result):
    """
    Generates and saves hexbin plot given a DataFrame and a correlation result derived from two columns in that DataFrame.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(
        df[corr_result.Column1], df[corr_result.Column2], cmap="Blues", mincnt=1
    )
    plt.colorbar(hb, ax=ax).set_label("Count in bin")
    ax.set_xlabel(corr_result.Column1)
    ax.set_ylabel(corr_result.Column2)
    ax.set_title(
        f"{corr_result.Column1} vs. {corr_result.Column2} \nCorr: {corr_result.Correlation:.3f}, p: {corr_result.P_value:.3g} ({corr_result.Significance})"
    )

    plt.savefig(
        os.path.join(
            plot_output_dir, f"hexbin_{corr_result.Column1}_{corr_result.Column2}.png"
        ),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

def compute_raw_correlation_with_popularity():
    """
    Given a DataFrame, this function computes the correlation between the popularity and all other columns in the dataset.
    It saves the correlation results in a CSV file and generates and saves hexbin plots for each correlation.
    """
    # Load the combined data
    combined_data = pd.read_csv("./data/combine_all.csv")

    # Collect numerical correlation results
    numeric_columns = combined_data.select_dtypes(include=["number"]).columns
    correlation_results = []

    for col in numeric_columns:
        if col != "Popularity":  # Avoid self-correlation
            corr, p_value = compute_correlation(combined_data, col, "Popularity")

            if corr is not None:
                alpha = 0.05  # Significance level
                significance = is_significant(p_value, alpha)
                result = CorrelationResult(
                    col, "Popularity", corr, p_value, significance
                )
                correlation_results.append(result)
            else:
                print(f"Insufficient data to compute correlation for column '{col}'")

    # Sort results by statistical significance (p-value)
    correlation_results.sort(key=lambda x: x.P_value)

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
    plot_output_dir = setup_output_dir("eda_plots/correlation_plots")
    for corr_result in correlation_results:
        generate_correlation_plot(combined_data, corr_result)
    print(f"Plots saved in {plot_output_dir}/")

def cluster_data_by_popularity():
    """
    TODO
    """
    combined_data = pd.read_csv("data/combine_all.csv")

    # Create output directories
    plot_dir = setup_output_dir("eda_plots")
    data_dir = setup_output_dir("eda_data")

    # First, we will examine the distribution of the 'Popularity' column using a kernel density plot
    combined_data["Popularity"].describe().to_csv(f"{data_dir}/popularity_stats.csv")
    combined_data['Popularity'].plot(kind='kde', title='Popularity Distribution', legend=True)
    plt.show()
    plt.savefig(f"{plot_dir}/popularity_distribution.png")


   


if __name__ == "__main__":
    # compute_raw_correlation_with_popularity()

    cluster_data_by_popularity()


