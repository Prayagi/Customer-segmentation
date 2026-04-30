import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_preprocessing import load_data, clean_data, select_features, scale_features
from src.model import find_optimal_clusters, train_model
from src.visualization import plot_elbow, plot_clusters

def run_pipeline():
    # Load data
    df = load_data('data/Mall_Customers.csv')

    # Clean data
    df = clean_data(df)

    # Feature selection
    features = select_features(df)

    # Scaling
    scaled_data = scale_features(features)

    # Find optimal clusters
    inertia = find_optimal_clusters(scaled_data)
    plot_elbow(inertia)

    # Train model
    model, labels = train_model(scaled_data, n_clusters=5)

    # Visualization
    plot_clusters(scaled_data, labels)

if __name__ == "__main__":
    run_pipeline()