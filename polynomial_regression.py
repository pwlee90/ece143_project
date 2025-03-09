import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def run_poly_regression(df : pd.DataFrame, features : List[str], target : str, d : int) -> Tuple[LinearRegression, float, float, float]:
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=143)

    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Accuracy check within 5% tolerance
    accuracy = np.mean(np.abs((y_test - y_pred) / y_test) <= 0.05) * 100

    return model, rmse, r2, accuracy


# Example usage
if __name__ == "__main__":
    import EDA
    data = EDA.load_cleaned_data()

    results_all = []
    results_hip_hop = []  

    pearson, spearman = EDA.report_correlation(data, "popularity")

    for i in range(1, 11): 
      for j in range(2, 5): 

        print(f"All Songs: {i} features selected, {j} poly degree")
        selected_features = pearson[:i]  # Extract the top i features
        print("Selected Features:")
        print(selected_features)
        
        model, rmse, r2, accuracy = run_poly_regression(data, selected_features.index.tolist(), "popularity", j)
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}")
        print(f"Accuracy: {accuracy}")
        print()

        results_all.append({
            "num_features": i,
            "poly_degree": j,
            "RMSE": rmse,
            "R2": r2,
            "Accuracy": accuracy
        })

        print(f"Hip Hop Songs: {i} features selected, {j} poly degree")
        hip_hop_songs = EDA.get_genre_subset(data, "hip hop")
        pearson, spearman = EDA.report_correlation(data, "popularity")
        
        selected_features = pearson[:i]  # Extract the top i features
        #print("Selected Features:")
        #print(selected_features)
        
        model, rmse, r2, accuracy = run_poly_regression(hip_hop_songs, selected_features.index.tolist(), "popularity", j)
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}")
        print(f"Accuracy: {accuracy}")
        print()

        results_hip_hop.append({
            "num_features": i,
            "poly_degree": j,
            "RMSE": rmse,
            "R2": r2,
            "Accuracy": accuracy
        })

    df_all = pd.DataFrame(results_all)
    df_hip_hop = pd.DataFrame(results_hip_hop)
    
    def plot_metric_by_features(df, metric, title_prefix):
      plt.figure(figsize=(8,6))
      # Loop over each unique number of features and plot the metric across polynomial degrees
      for nf in sorted(df['num_features'].unique()):
          subset = df[df['num_features'] == nf]
          plt.plot(subset['poly_degree'], subset[metric], marker='o', label=f"{nf} features")
      plt.title(f"{title_prefix}: {metric} vs. Polynomial Degree")
      plt.xlabel("Polynomial Degree")
      plt.ylabel(metric)
      plt.legend(title="Selected Features")
      plt.grid(True)
      plt.show()

    # Plot for All Songs
    for metric in ['RMSE', 'R2', 'Accuracy']:
        plot_metric_by_features(df_all, metric, "All Songs")

    # Plot for Hip Hop Songs
    for metric in ['RMSE', 'R2', 'Accuracy']:
        plot_metric_by_features(df_hip_hop, metric, "Hip Hop Songs")
