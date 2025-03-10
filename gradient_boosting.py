import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ErrorAnalysis import analyze_predictions

df = pd.read_csv("cleaned_data_with_emotions.csv")

# Display all rows/columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

# Define a function to train & evaluate a predictor
def train_and_evaluate_popularity_predictor(df, genre=None, top_n_features=20):
    """
    Trains an XGBoost model (with GPU support) to predict song popularity.
    Uses Spearman correlation (monotonic) instead of Pearson.
    
    Parameters:
        df (DataFrame): The full dataset.
        genre (str or None): If provided, filters the dataset for rows where 
                             the 'genres' column contains this substring (case-insensitive).
                             If None, uses the entire dataset.
        top_n_features (int): Number of top features (by absolute Spearman correlation with popularity)
                              to use as predictors.
                              
    Returns:
        model: The trained XGBoost model.
        X_test, y_test: The test set (for further evaluation if needed).
        y_pred: The predicted values for the test set.
    """
    # If genre is specified, filter rows where the genres column contains the substring.
    if genre:
        df_filtered = df[df["genres"].str.contains(genre, case=False, na=False)]
        print(f"Filtered dataset to genre '{genre}' (n = {len(df_filtered)} rows).")
    else:
        df_filtered = df.copy()
        print(f"Using entire dataset (n = {len(df_filtered)} rows).")
    
    # Feature Selection Based on Spearman Correlation
    # Get numerical columns; remove target column 'popularity'
    numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
    if "popularity" in numeric_cols:
        numeric_cols.remove("popularity")

    # Compute Spearman correlation manually for each feature
    spearman_corr = {}
    for col in numeric_cols:
        spearman_corr[col], _ = spearmanr(df_filtered[col], df_filtered["popularity"], nan_policy='omit')

    # Convert to Pandas Series and sort by absolute value
    spearman_corr = pd.Series(spearman_corr)
    abs_corr = spearman_corr.abs().sort_values(ascending=False)

    print("\nAll features ranked by absolute Spearman correlation with popularity:")
    print(abs_corr.to_string())

    # Select the top_n_features predictors
    top_features = abs_corr.head(top_n_features).index.tolist()
    print(f"\nSelected top {top_n_features} features for prediction:")
    print(top_features)

    # Prepare Data Splits
    X = df_filtered[top_features]
    y = df_filtered["popularity"]

    # First split off the test set (20% of data)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # Then split the remaining into training and validation sets (60% train, 20% val overall)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    print(f"\nData split: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples.")

    # Train XGBoost Model
    # Setting up parameters to use CUDA
    params = {
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist",       
        "predictor": "gpu_predictor",     
        "gpu_id": 0,
        "random_state": 42,
        "eval_metric": "rmse"
    }

    # Initialize the model with the parameters.
    model = xgb.XGBRegressor(**params, n_estimators=200, max_depth=5, learning_rate=0.1)

    # Set up evaluation set to monitor training and validation error.
    eval_set = [(X_train, y_train), (X_val, y_val)]
    print("\nTraining XGBoost model with GPU support...")

    # Train the model
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    # Retrieve evaluation results (training and validation metrics)
    results = model.evals_result()
    eval_metric = params["eval_metric"]

    # Evaluate Model on Test Set
    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    # Compute SMAPE
    smape = np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + 1e-5)) * 100
    accuracy = 100 - smape

    print(f"\nTest RMSE: {test_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test SMAPE: {smape:.2f}%")
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot training and validation RMSE over boosting rounds
    epochs = len(results["validation_0"][eval_metric])
    x_axis = range(epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results["validation_0"][eval_metric], label="Train")
    plt.plot(x_axis, results["validation_1"][eval_metric], label="Validation")
    plt.xlabel("Boosting Round")
    plt.ylabel("RMSE")
    plt.title("XGBoost Training and Validation RMSE")
    plt.legend()
    plt.show()

    # Plot Actual vs Predicted Popularity on Test Set
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title("Actual vs. Predicted Popularity on Test Set")
    plt.show()

    return model, X_test, y_test, y_pred

# Run the Predictor on the Entire Dataset or a Specific Genre
# To run on the entire dataset, set genre=None.
# To run on a specific genre (e.g., "hip hop"), provide that as the argument.
trained_model, X_test, y_test, y_pred = train_and_evaluate_popularity_predictor(df, genre=None, top_n_features=20)

analyze_predictions(y_test, y_pred, title="Hip Hop Popularity Prediction using XGBoost")