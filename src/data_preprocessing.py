import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    return df

def select_features(df):
    # Selecting important features
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    return features

def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled
'''def preprocess_data(path):
    df = load_data(path)
    df = clean_data(df)
    features = select_features(df)
    scaled_features = scale_features(features)
    return scaled_features
if __name__ == "__main__":
    path = 'customer_data.csv'
    processed_data = preprocess_data(path)
    print(processed_data)'''