import pandas as pd
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def run_linear_regression(df : pd.DataFrame, features : List[str], target : str) -> Tuple[LinearRegression, float, float, float]:
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=143)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Accuracy check within 5% tolerance
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= 0.05) * 100

    return model, rmse, r2, accuracy

def find_optimal_features(data, correlation_df, target):
    min_rmse = float('inf')
    max_accuracy = 0
    min_rmse_accuracy = 0
    max_accuracy_rmse = float('inf') 
    optimal_features = []
    best_accuracy_features = []

    for n_features in range(1, len(correlation_df) + 1):
        selected_features = correlation_df[:n_features].index.tolist()
        model, rmse, r2, accuracy = run_linear_regression(data, selected_features, target)

        if rmse < min_rmse:
            min_rmse = rmse
            optimal_features = selected_features
            min_rmse_accuracy = accuracy

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_accuracy_features = selected_features
            max_accuracy_rmse = rmse

    print(f"\nOptimal Number of Features for min RMSE: {len(optimal_features)}")
    print(f"Min RMSE: {min_rmse}")
    print(f"Accuracy for optimal RMSE: {min_rmse_accuracy}%")

    print(f"\nOptimal Number of Features for Best Accuracy: {len(best_accuracy_features)}")
    print(f"Best Accuracy: {max_accuracy}%")
    print(f"RMSE for Best Accuracy: {max_accuracy_rmse}")

# Example usage
if __name__ == "__main__":
    import EDA
    data = EDA.load_cleaned_data()

    print("All Songs")
    pearson, spearman = EDA.report_correlation(data, "popularity")
    print("Selected Features:")
    find_optimal_features(data, pearson, "popularity")
    ''' 
    print("Selected Features:")
    selected_features = pearson[:10]  # Extract the top 10 features
    print(selected_features)
    
    model, rmse, r2 = run_linear_regression(data, selected_features.index.tolist(), "popularity")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    '''
    print()
    print("Hip Hop Songs")
    hip_hop_songs = EDA.get_genre_subset(data, "hip hop")
    pearson, spearman = EDA.report_correlation(data, "popularity")
    print("Selected Features:")
    find_optimal_features(hip_hop_songs, pearson, "popularity")

    '''
    selected_features = pearson[:10]  # Extract the top 10 features
    print("Selected Features:")
    print(selected_features)
    
    model, rmse, r2 = run_linear_regression(hip_hop_songs, selected_features.index.tolist(), "popularity")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    '''
    print()
    print("Pop Songs")
    pop_songs = EDA.get_genre_subset(data, "pop")
    pearson, spearman = EDA.report_correlation(data, "popularity")
    print("Selected Features:")
    find_optimal_features(pop_songs, pearson, "popularity")

    '''
    selected_features = pearson[:10]  # Extract the top 10 features
    print("Selected Features:")
    print(selected_features)    
    model, rmse, r2 = run_linear_regression(pop_songs, selected_features.index.tolist(), "popularity")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    '''
    print()


'''
All Songs
Selected Features:

Optimal Number of Features for min RMSE: 245
Min RMSE: 14.328092207009009
Accuracy for optimal RMSE: 10.79465159256441%

Optimal Number of Features for Best Accuracy: 176
Best Accuracy: 11.392542667681269%
RMSE for Best Accuracy: 14.39503735944937

Hip Hop Songs
Selected Features:

Optimal Number of Features for min RMSE: 42
Min RMSE: 14.36805676648163
Accuracy for optimal RMSE: 12.333965844402277%

Optimal Number of Features for Best Accuracy: 203
Best Accuracy: 14.800759013282732%
RMSE for Best Accuracy: 17.060877192860136

Pop Songs
Selected Features:

Optimal Number of Features for min RMSE: 160
Min RMSE: 14.675526575613976
Accuracy for optimal RMSE: 13.57592722183345%

Optimal Number of Features for Best Accuracy: 174
Best Accuracy: 14.485654303708886%
RMSE for Best Accuracy: 14.737798613218105

'''
