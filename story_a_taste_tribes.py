"""
MovieLens Data Mining - Story A: Taste Tribes
This module performs user clustering based on their training-period ratings and features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from yellowbrick.cluster import KElbowVisualizer
    HAS_YELLOWBRICK = True
except ImportError:
    HAS_YELLOWBRICK = False

# Import modular utilities
from app_utils.config import DATA_DIR, FIGURES_OUT
from app_utils.data_loader import generate_artifacts
from app_utils.visualizations import apply_dimensionality_reduction, create_radar_chart

def setup_dirs():
    """Create necessary directories for outputs."""
    from app_utils.config import TABLES_OUT, REPORTS_OUT, FIGURES_OUT
    os.makedirs(TABLES_OUT, exist_ok=True)
    os.makedirs(REPORTS_OUT, exist_ok=True)
    os.makedirs(FIGURES_OUT, exist_ok=True)

def load_data():
    """Load inputs from the core pipeline."""
    print("Loading datasets...")
    features_path = os.path.join(DATA_DIR, "user_features_train.parquet")
    movies_path = os.path.join(DATA_DIR, "dim_movies_clean.parquet")
    interactions_path = os.path.join(DATA_DIR, "interactions_train.parquet")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing input file: {features_path}.")

    user_features = pd.read_parquet(features_path)
    movies = pd.read_parquet(movies_path) if os.path.exists(movies_path) else None
    interactions = pd.read_parquet(interactions_path) if os.path.exists(interactions_path) else None
    
    return user_features, movies, interactions

def preprocess_features(df):
    """Clean data, apply log transform, and scale."""
    print("Preprocessing features...")
    df_clean = df.copy()
    exclude_cols = ["userId", "first_dt", "last_dt"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
    skewed_cols = [c for c in feature_cols if "n_ratings" in c or "count" in c]
    for c in skewed_cols:
        df_clean[c] = np.log1p(df_clean[c])
        
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[feature_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols, index=df_clean.index)
    
    return df_clean, df_scaled, feature_cols, scaler

def find_optimal_k(X_scaled, max_k=10):
    """Find optimal K using Silhouette Score (and Elbow if YellowBrick available)."""
    print(f"Finding optimal K up to {max_k}...")
    sample_size = min(20000, len(X_scaled))
    X_sample = X_scaled.sample(sample_size, random_state=42) if sample_size < len(X_scaled) else X_scaled
        
    if HAS_YELLOWBRICK:
        model = KMeans(random_state=42)
        visualizer = KElbowVisualizer(model, k=(2, max_k), metric='distortion', timings=False)
        visualizer.fit(X_sample)
        visualizer.show(outpath=os.path.join(FIGURES_OUT, "elbow_visualizer.png"), clear_figure=True)
        
    best_k, best_score, scores = 2, -1, []
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        scores.append(score)
        print(f"K={k} -> Silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
            
    # Save silhouette plot
    plt.figure(figsize=(8,5))
    plt.plot(range(2, max_k+1), scores, marker='o')
    plt.title('Silhouette Score vs. K')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_OUT, "silhouette_scores.png"))
    plt.close()
    
    return best_k

def main():
    print("=== MovieLens Data Mining | Story A: Taste Tribes ===")
    setup_dirs()
    try:
        df_features, df_movies, df_interactions = load_data()
        df_clean, df_scaled, feature_cols, scaler = preprocess_features(df_features)
        
        best_k = find_optimal_k(df_scaled, max_k=8)
        
        print(f"Executing K-Means with K={best_k}...")
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(df_scaled)
        
        apply_dimensionality_reduction(df_scaled, labels)
        create_radar_chart(kmeans.cluster_centers_, feature_cols)
        generate_artifacts(df_clean, df_scaled, labels, kmeans, feature_cols, df_movies, df_interactions, scaler)
        
        print("\nAll done!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
