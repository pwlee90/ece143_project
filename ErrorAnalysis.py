import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_residuals(y_test: pd.Series, y_pred: pd.Series, title: str):
    """Plot residuals to visualize prediction errors."""
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution: {title}")
    plt.show()

def plot_actual_vs_pred(y_test: pd.Series, y_pred: pd.Series, title: str):
    """Plot actual vs. predicted values to assess prediction accuracy."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect predictions line
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs. Predicted: {title}")
    plt.show()

def compute_accuracy_with_tolerance(y_test: pd.Series, y_pred: pd.Series, tolerance: float) -> float:
    """Compute the accuracy of the predictions within a given tolerance."""
    accuracy = (abs(y_test - y_pred) / y_test <= tolerance).mean() * 100
    return accuracy

import pandas as pd
import numpy as np

def compute_accuracy_with_std(y_test: pd.Series, y_pred: pd.Series, num_std: float = 2) -> float:
    """Compute the accuracy of predictions based on residuals within a given number of standard deviations.
    
    Params:
        y_test (pd.Series): Actual values.
        y_pred (pd.Series): Predicted values.
        num_std (float): Number of standard deviations to consider for accuracy.
        
    Returns:
        float: Accuracy percentage within the range.
    """
    residuals = y_test - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    lower_bound = mean_residual - num_std * std_residual
    upper_bound = mean_residual + num_std * std_residual

    # Count the number of residuals within the range
    within_range = ((residuals >= lower_bound) & (residuals <= upper_bound)).mean() * 100

    print(f"Residual Range ({num_std} SD): [{lower_bound:.2f}, {upper_bound:.2f}]")

    return within_range
