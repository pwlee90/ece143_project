import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error

def compute_rmse(y_test: pd.Series, y_pred: pd.Series) -> float:
    """Compute the Root Mean Squared Error (RMSE) between the actual and predicted values."""
    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse

def compute_r2(y_test: pd.Series, y_pred: pd.Series) -> float:
    """Compute the R2 score between the actual and predicted values."""
    r2 = r2_score(y_test, y_pred)
    return r2

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
    """Plot actual vs. predicted values with additional statistical insights."""
    
    # Compute Standard Deviation of Errors
    residuals = y_test - y_pred
    std_dev = np.std(residuals)
    
    # Compute RMSE for display
    rmse = root_mean_squared_error(y_test, y_pred)
    
    # Set style and figure size
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    # Scatterplot of datapoints
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="mediumslateblue", edgecolor="black")

    # Add perfect prediction reference
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit")

    # Labels and Title
    plt.xlabel("Actual Popularity", fontsize=12)
    plt.ylabel("Predicted Popularity", fontsize=12)
    plt.title(f"Actual vs. Predicted Popularity: {title}", fontsize=14, fontweight='bold')

    # Display RMSE as annotation
    plt.annotate(f"RMSE: {rmse:.2f}", 
                 xy=(0.05, 0.9), xycoords='axes fraction', 
                 fontsize=12, color='darkblue', fontweight='bold')

    plt.legend()
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

def compare_distributions(y_test: pd.Series, y_pred: pd.Series, title: str):
    """Compare actual vs. predicted values and plot residuals."""
    plt.hist(y_test, bins=30, alpha=0.5, label='Actual Popularity Distribution')
    plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted Popularity Distribution')
    plt.title(f"{title} Popularity Distribution")
    plt.legend()
    plt.show()

def plot_bland_altman(y_test: pd.Series, y_pred: pd.Series, num_std: float = 2):
    """Generate a Bland-Altman plot to assess agreement between actual and predicted values.
    
    Params:
        y_test (pd.Series): Actual values.
        y_pred (pd.Series): Predicted values.
        num_std (float): Number of standard deviations for limits of agreement.
    
    Returns:
        None (Displays the plot).
    """
    mean_values = (y_test + y_pred) / 2  # Mean of actual and predicted values
    diff_values = y_test - y_pred  # Differences (Actual - Predicted)
    
    mean_diff = np.mean(diff_values)  # Bias (average difference)
    std_diff = np.std(diff_values)    # Standard deviation of differences

    upper_limit = mean_diff + num_std * std_diff  # Upper limit of agreement
    lower_limit = mean_diff - num_std * std_diff  # Lower limit of agreement

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=mean_values, y=diff_values, alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean Diff: {mean_diff:.2f}")
    plt.axhline(upper_limit, color='blue', linestyle='--', label=f"+{num_std} SD: {upper_limit:.2f}")
    plt.axhline(lower_limit, color='blue', linestyle='--', label=f"-{num_std} SD: {lower_limit:.2f}")
    plt.xlabel("Mean of Actual & Predicted")
    plt.ylabel("Difference (Actual - Predicted)")
    plt.title(f"Bland-Altman Plot ({num_std} SD)")
    plt.legend()
    plt.show()

    print(f"Limits of Agreement ({num_std} SD): [{lower_limit:.2f}, {upper_limit:.2f}]")

def compute_smape(y_test: pd.Series, y_pred: pd.Series) -> float:
    """Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) for two Series."""
    smape = np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + 1e-5)) * 100
    return smape

def analyze_predictions(y_test : pd.Series, y_pred : pd.Series, title : str) -> None:
    rmse = compute_rmse(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    smape = compute_smape(y_test, y_pred)

    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print("SMAPE: ", smape)
    print()

    accuracy = compute_accuracy_with_tolerance(y_test, y_pred, 0.05)
    print(f"Using tolerance of 5%, accuracy is: {accuracy}%")

    accuracy = compute_accuracy_with_std(y_test, y_pred, 1)
    print(f"The percentage of residuals within 1 std devs of the mean of residuals is: {accuracy}%")

    compare_distributions(y_test, y_pred, title)
    plot_bland_altman(y_test, y_pred, 2)
    plot_residuals(y_test, y_pred, title)
    plot_actual_vs_pred(y_test, y_pred, title)

def generate_slide_graphics(y_test : pd.Series, y_pred : pd.Series, title : str) -> None:
    """Generate graphics for the slides."""
    rmse = compute_rmse(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results = pd.DataFrame({"RMSE:": [rmse], "R2:": [r2]})
    print(results)
    plot_actual_vs_pred(y_test, y_pred, title)

    