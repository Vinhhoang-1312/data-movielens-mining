"""
MovieLens Data Mining - Story A: Taste Tribes
This module performs user clustering based on their training-period ratings and features.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
try:
    from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
    HAS_YELLOWBRICK = True
except ImportError:
    HAS_YELLOWBRICK = False
    print("Warning: yellowbrick is not installed. Will use standard silhouette score for K selection.")

# Configure paths
DATA_DIR = "movielens-parquet-build-2026"
ARTIFACTS_DIR = "artifacts/story_A"
TABLES_OUT = os.path.join(ARTIFACTS_DIR, "tables")
REPORTS_OUT = os.path.join(ARTIFACTS_DIR, "reports")
FIGURES_OUT = os.path.join(ARTIFACTS_DIR, "figures")

def setup_dirs():
    """Create necessary directories for outputs."""
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
        raise FileNotFoundError(f"Missing input file: {features_path}. Please ensure paths match data/processed/tables/.")

    user_features = pd.read_parquet(features_path)
    movies = pd.read_parquet(movies_path) if os.path.exists(movies_path) else None
    interactions = pd.read_parquet(interactions_path) if os.path.exists(interactions_path) else None
    
    return user_features, movies, interactions

def preprocess_features(df):
    """Clean data, apply log transform, and scale."""
    print("Preprocessing features...")
    df_clean = df.copy()
    
    # Identify identifier and date columns to exclude from scaling
    exclude_cols = ["userId", "first_dt", "last_dt"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Fill missing values:
    df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
    
    # Identify skewed metrics that need log-transform (activity features usually have count/n_ratings)
    skewed_cols = [c for c in feature_cols if "n_ratings" in c or "count" in c]
    for c in skewed_cols:
        # log1p handles 0s safely
        df_clean[c] = np.log1p(df_clean[c])
        
    print(f"Applied log transform to: {skewed_cols}")
        
    # Standardize remaining features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[feature_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols, index=df_clean.index)
    
    return df_clean, df_scaled, feature_cols, scaler

def find_optimal_k(X_scaled, max_k=10):
    """Find optimal K using Silhouette Score (and Elbow if YellowBrick available)."""
    print(f"Finding optimal K up to {max_k}...")
    
    # Sample down if dataset is large for quicker silhouette computation
    sample_size = min(20000, len(X_scaled))
    if sample_size < len(X_scaled):
        print(f"Sampling {sample_size} records for silhouette score calculation...")
        X_sample = X_scaled.sample(sample_size, random_state=42)
    else:
        X_sample = X_scaled
        
    if HAS_YELLOWBRICK:
        # Elbow Method Visualizer
        model = KMeans(random_state=42)
        visualizer = KElbowVisualizer(model, k=(2, max_k), metric='distortion', timings=False)
        visualizer.fit(X_sample)
        visualizer.show(outpath=os.path.join(FIGURES_OUT, "elbow_visualizer.png"), clear_figure=True)
        elbow_k = visualizer.elbow_value_
        print(f"KElbowVisualizer suggests: {elbow_k}")
        
    best_k = 2
    best_score = -1
    scores = []
    
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        scores.append(score)
        print(f"K={k} -> Silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Optimal K selected: {best_k} (Highest Silhouette: {best_score:.4f})")
    
    # Save silhouette plot
    plt.figure(figsize=(8,5))
    plt.plot(range(2, max_k+1), scores, marker='o')
    plt.title('Silhouette Score vs. K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_OUT, "silhouette_scores.png"))
    plt.close()
    
    return best_k

def apply_dimensionality_reduction(X_scaled, labels):
    """Apply PCA and t-SNE for 2D visualization."""
    print("Applying Dimensionality Reduction (PCA & t-SNE)...")
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    print(f"PCA explained variance ratio (2 comps): {pca.explained_variance_ratio_.sum():.4f}")
    
    # 2. t-SNE for plotting (sample to 1,000 for performance)
    sample_size = min(1000, len(X_scaled))
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled.iloc[idx]
    labels_sample = labels[idx]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(X_sample)
    
    df_tsne = pd.DataFrame({'x': tsne_result[:, 0], 'y': tsne_result[:, 1], 'Cluster': labels_sample})
    df_tsne['Cluster'] = df_tsne['Cluster'].astype(str)
    
    fig = px.scatter(
        df_tsne, x='x', y='y', color='Cluster',
        title='Taste Tribes (t-SNE 2D Projection)',
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.write_html(os.path.join(FIGURES_OUT, "cluster_scatter_tsne.html"))

def create_radar_chart(centroids, feature_cols):
    """Create radar chart comparing genre preferences across clusters."""
    print("Generating Radar Chart...")
    
    # Extract only genre-related features
    genre_cols = [c for c in feature_cols if "genre" in c.lower()]
    if not genre_cols:
        genre_cols = feature_cols[:8] # fallback
        
    top_n = min(8, len(genre_cols))
    selected_genres = genre_cols[:top_n]
    
    fig = go.Figure()
    
    for i, centroid in enumerate(centroids):
        # Retrieve the centroid scores for the selected genres
        values = [centroid[feature_cols.index(c)] for c in selected_genres]
        values.append(values[0]) # close polygon
        
        # Clean up labels
        labels = [g.replace('genre_', '').replace('genre:', '').title() for g in selected_genres]
        labels.append(labels[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=f'Cluster {i}'
        ))
        
    fig.update_layout(
      polar=dict(radialaxis=dict(visible=True, title="Z-Score")),
      showlegend=True,
      title="Cluster Central Genre Preferences (Standardized)",
      template="plotly_white"
    )
    fig.write_html(os.path.join(FIGURES_OUT, "genre_radar.html"))
    
def generate_artifacts(df_original, df_scaled, labels, kmeans_model, feature_cols, movies, interactions, scaler):
    """Generate Parquet data, JSON profiles, and Markdown narrative."""
    print("Exporting Artifacts...")
    
    # 1. cluster_labels_users.parquet
    df_original['cluster_id'] = labels
    labels_df = df_original[['userId', 'cluster_id']].copy()
    labels_df['method'] = 'kmeans'
    labels_df.to_parquet(os.path.join(TABLES_OUT, "cluster_labels_users.parquet"), index=False)
    
    # 2. cluster_profiles.json & Narrative
    profiles = {}
    centroids = kmeans_model.cluster_centers_
    
    md_lines = [
        "# Story A: Taste Tribes - Cluster Narratives\n",
        "This document describes each identified taste tribe (cluster) and its behavioral profile.\n"
    ]
    
    for c in np.unique(labels):
        cluster_data = df_original[df_original['cluster_id'] == c]
        size = len(cluster_data)
        percentage = (size / len(df_original)) * 100
        
        centroid = centroids[c]
        genre_cols = [col for col in feature_cols if "genre" in col.lower()]
        
        # Profile top genres
        genre_scores = {col: centroid[feature_cols.index(col)] for col in genre_cols}
        top_genres = sorted(genre_scores, key=genre_scores.get, reverse=True)[:5]
        
        # Extract Human Readable Name
        if top_genres:
            top2 = [g.replace('genre_','').replace('genre:','').title() for g in top_genres[:2]]
            profile_name = " & ".join(top2) + " Enthusiasts"
        else:
            profile_name = f"General Group {c}"
            
        profiles[str(c)] = {
            "profile_name": profile_name,
            "size": int(size),
            "percentage": float(percentage),
            "top_genres": top_genres,
            "centroid_scores": {f: float(v) for f, v in zip(feature_cols, centroid)}
        }
        
        md_lines.append(f"## Cluster {c}: {profile_name}")
        md_lines.append(f"- **Size**: {size} users ({percentage:.1f}%)")
        clean_genres = [g.replace('genre_','').replace('genre:','').title() for g in top_genres]
        md_lines.append(f"- **Top Preferences**: {', '.join(clean_genres)}")
        
        # 3. Top 10 representative movies
        md_lines.append("\n### Representative Movies")
        if interactions is not None and movies is not None:
            # Subset interactions for this cluster
            c_users = cluster_data['userId'].unique()
            c_interactions = interactions[interactions['userId'].isin(c_users)]
            
            # Find highly rated + popular movies within cluster
            movie_stats = c_interactions.groupby('movieId').agg(
                c_mean_rating=('rating', 'mean'),
                c_rating_count=('rating', 'count')
            ).reset_index()
            
            # Filter somewhat popular in the cluster to drop noisy 5.0s
            min_count = max(5, np.percentile(movie_stats['c_rating_count'], 75))
            top_movies = movie_stats[movie_stats['c_rating_count'] >= min_count].nlargest(10, 'c_mean_rating')
            
            # Join titles
            top_movies = top_movies.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
            
            for _, row in top_movies.iterrows():
                md_lines.append(f"- **{row['title']}** (Mean: {row['c_mean_rating']:.2f}, Views: {row['c_rating_count']}) - _{row['genres']}_")
        else:
            md_lines.append("*Interactions or Movies metadata not fully available to compute top movies.*")
            
        md_lines.append("\n---\n")
        
    # Write JSON
    with open(os.path.join(REPORTS_OUT, "cluster_profiles.json"), "w", encoding='utf-8') as f:
        json.dump(profiles, f, indent=4)
        
    # Write Model Metadata for Streamlit Demo
    metadata = {
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    with open(os.path.join(REPORTS_OUT, "model_metadata.json"), "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    # Write Markdown
    with open(os.path.join(REPORTS_OUT, "cluster_cards.md"), "w", encoding='utf-8') as f:
        f.write("\n".join(md_lines))
        
    print(f"Artifacts successfully exported to: {ARTIFACTS_DIR}")

def main():
    print("=== MovieLens Data Mining | Story A: Taste Tribes ===")
    setup_dirs()
    try:
        # 1. Loading
        df_features, df_movies, df_interactions = load_data()
        
        # 2. Preprocessing
        df_clean, df_scaled, feature_cols, scaler = preprocess_features(df_features)
        
        # 3. Optimal K
        best_k = find_optimal_k(df_scaled, max_k=8)
        
        # 4. Clustering
        print(f"Executing K-Means with K={best_k}...")
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(df_scaled)
        
        # 5. Vis
        apply_dimensionality_reduction(df_scaled, labels)
        create_radar_chart(kmeans.cluster_centers_, feature_cols)
        
        # 6. Output
        generate_artifacts(df_clean, df_scaled, labels, kmeans, feature_cols, df_movies, df_interactions, scaler)
        
        print("\nAll done!")
    except FileNotFoundError as e:
        print(f"Execution Stopped: {e}")
        print("Note: Run from the project root where 'data/processed/tables/' exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
