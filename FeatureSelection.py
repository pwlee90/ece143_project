import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_PCA(df : pd.DataFarme, target : str, variance_threshold, correlation_threshold = 0.1):
    """
    Performs PCA on the given DataFrame excluding the column target and returns the transformed data.
    Returns a DataFrame with the principal components.
    """
    correlation = df.corr()[target].drop(target)

    relevant_features = correlation[correlation.abs() > variance_threshold].index.tolist()

    if len(relevant_features) == 0:
        raise ValueError("No relevant features found. Please adjust the variance threshold.")
    
    # Standardize the data
    scaler = StandardScaler()
    train_df = scaler.fit_transform(train_df)


    pca = PCA(n_components=variance_threshold)
    components = pca.fit_transform(train_df)
    pca_df = pd.DataFrame(data = components, columns = [f"PC{i}" for i in range(components.shape[1])])
    
    return pca_df

if __name__ == "__main__":
    pass