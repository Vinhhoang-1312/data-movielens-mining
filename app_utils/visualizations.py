import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from .config import FIGURES_OUT

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
