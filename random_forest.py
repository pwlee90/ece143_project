import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

def run_random_forest(df : pd.DataFrame, features : List[str], target : str) -> Tuple[RandomForestRegressor, float, float]:
    """"
    Params:
        df : pd.DataFrame - Training data
        features : List[str] - List of features to train the model on
        target : str - Target variable to predict
    Returns:
        Tuple[RandomForestRegressor, float, float] - Tuple containing the trained model, RMSE, and R2 score
    """
    
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=143)

    model = RandomForestRegressor(n_estimators=100, random_state=143)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    return model, rmse, r2, y_test, y_pred

# Example usage
if __name__ == "__main__":
    import EDA
    data = EDA.load_cleaned_data()
    import ErrorAnalysis

    # # Example with all songs
    # print("All Songs")
    # pearson, spearman = EDA.report_correlation(data, "popularity")
    
    # print("Selected Features:")
    # selected_features = spearman[:10]# Extract the top 10 features
    # print(selected_features)
    
    # model, rmse, r2, y_test, y_pred  = run_random_forest(data, selected_features.index.tolist(), "popularity")
    # print(f"RMSE: {rmse}")
    # print(f"R2: {r2}")
    # print()

    # # Plot distribution of predictions
    # plt.hist(y_test, bins=30, alpha=0.5, label='Actual Popularity Distribution')
    # plt.show()

    # ErrorAnalysis.plot_residuals(y_test, y_pred, "All Songs")
    # ErrorAnalysis.plot_actual_vs_pred(y_test, y_pred, "All Songs")
    # print("Using tolerance of 5%")
    # accuracy = ErrorAnalysis.compute_accuracy_with_tolerance(y_test, y_pred, 0.05)
    # print(f"Accuracy: {accuracy}%")

    # print ("Using 2 standard deviations")
    # accuracy = ErrorAnalysis.compute_accuracy_with_std(y_test, y_pred, 2)
    # print(f"Accuracy: {accuracy}%")


    # Example with hip hop songs
    print("Hip Hop Songs")
    hip_hop_songs = EDA.get_genre_subset(data, "hip hop")
    pearson, spearman = EDA.report_correlation(hip_hop_songs, "popularity")
    
    selected_features = spearman[:10]# Extract the top 10 features
    print("Selected Features:")
    print(selected_features)
    
    model, rmse, r2, y_test, y_pred  = run_random_forest(hip_hop_songs, selected_features.index.tolist(), "popularity")
    ErrorAnalysis.analyze_predictions(y_test, y_pred, "Hip Hop Songs")


    # # Example with pop songs
    # print("Pop Songs")
    # pop_songs = EDA.get_genre_subset(data, "pop")
    # pearson, spearman = EDA.report_correlation(pop_songs, "popularity")
    
    # selected_features = spearman[:10]# Extract the top 10 features
    # print("Selected Features:")
    # print(selected_features)
    
    # model, rmse, r2  = run_random_forest(pop_songs, selected_features.index.tolist(), "popularity")
    # print(f"RMSE: {rmse}")
    # print(f"R2: {r2}")
    # print()