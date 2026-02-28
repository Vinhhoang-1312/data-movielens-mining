"""
Streamlit Dashboard for Story A: Taste Tribes
Run with: streamlit run app_story_a.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Taste Tribes Dashboard", layout="wide")

ARTIFACTS_DIR = "artifacts/story_A"
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
TABLES_DIR = os.path.join(ARTIFACTS_DIR, "tables")
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")

st.title("🎬 Story A: Taste Tribes (User Segmentation)")
st.markdown("Explore the clustering of MovieLens users based on their preferences and behavior.")

@st.cache_data
def load_data():
    try:
        with open(os.path.join(REPORTS_DIR, "cluster_profiles.json"), "r") as f:
            profiles = json.load(f)
            
        labels_df = pd.read_parquet(os.path.join(TABLES_DIR, "cluster_labels_users.parquet"))
        
        with open(os.path.join(REPORTS_DIR, "cluster_cards.md"), "r", encoding='utf-8') as f:
            cards_md = f.read()
            
        try:
            with open(os.path.join(REPORTS_DIR, "model_metadata.json"), "r", encoding='utf-8') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = None
            
        return profiles, labels_df, cards_md, metadata
    except FileNotFoundError:
        return None, None, None, None

profiles, labels_df, cards_md, metadata = load_data()

if not profiles:
    st.warning("Artifacts not found! Please run `python story_a_taste_tribes.py` first to generate data.")
    st.stop()

# Create Tabs
tab1, tab2 = st.tabs(["📊 Overview & Analytics", "🚀 Interactive Cold-Start Demo"])

with tab1:
    # 1. Overview Section
    st.header("1. Overview of Tribes")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Cluster Distribution")
        dist_data = [{"Cluster": k, "Name": v["profile_name"], "Size": v["size"], "Pct": v["percentage"]} 
                     for k, v in profiles.items()]
        dist_df = pd.DataFrame(dist_data)
        
        fig_pie = px.pie(dist_df, values='Size', names='Name', hole=0.4, 
                         title="User Distribution across Tribes")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Global Narrative")
        st.markdown("A quick descriptive summary pulled directly from our analytical pipeline.")
        with st.expander("Read Cluster Cards", expanded=True):
            st.markdown(cards_md)
    
    # 2. Radar Chart
    st.header("2. Genre Fingerprints")
    st.markdown("Review the raw standardized values (z-scores) characterizing each cluster's genre affinity.")
    try:
        with open(os.path.join(FIGURES_DIR, "genre_radar.html"), 'r', encoding='utf-8') as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=500)
    except FileNotFoundError:
        st.info("Interactive Radar chart HTML not found (fallback available).")
    
    # 3. t-SNE Scatter
    st.header("3. 2D Manifold Projection (t-SNE)")
    st.markdown("Observe topological separation between taste clusters in the sampled reduced space.")
    try:
        with open(os.path.join(FIGURES_DIR, "cluster_scatter_tsne.html"), 'r', encoding='utf-8') as f:
            html_data2 = f.read()
        st.components.v1.html(html_data2, height=600)
    except FileNotFoundError:
        st.info("Interactive t-SNE scatter plot not found.")

with tab2:
    st.header("New User Onboarding (Cold-Start)")
    st.markdown("Select a few of your favorite genres to see which Taste Tribe you belong to, and get instant recommendations!")
    
    if metadata:
        feature_cols = metadata["feature_cols"]
        scaler_mean = np.array(metadata["scaler_mean"])
        scaler_scale = np.array(metadata["scaler_scale"])
        
        # Extract available genres
        genre_cols = [c for c in feature_cols if "genre_pref__" in c]
        display_genres = [c.replace("genre_pref__", "").title() for c in genre_cols]
        
        selected_genres = st.multiselect(
            "What genres do you absolutely love?", 
            options=display_genres,
            max_selections=5
        )
        
        if st.button("Discover My Taste Tribe!", type="primary"):
            if not selected_genres:
                st.warning("Please select at least one genre to find your tribe.")
            else:
                with st.spinner("Analyzing your tastes..."):
                    # 1. Build synthetic vector
                    user_vector = np.zeros(len(feature_cols))
                    
                    # Assume typical activity for a new user
                    if "n_ratings" in feature_cols:
                        user_vector[feature_cols.index("n_ratings")] = np.log1p(20) # 20 hypothetical ratings
                    if "rating_mean" in feature_cols:
                        user_vector[feature_cols.index("rating_mean")] = 4.0
                    
                    # Add high scores to selected genres
                    for g in selected_genres:
                        raw_col = "genre_pref__" + g.lower()
                        if raw_col in feature_cols:
                            user_vector[feature_cols.index(raw_col)] = 5.0 # Max rating
                            
                    # 2. Scale it
                    user_scaled = (user_vector - scaler_mean) / scaler_scale
                    
                    # 3. Compute distances to centroids
                    best_cluster = None
                    min_dist = float('inf')
                    
                    for k, v in profiles.items():
                        centroid = np.array([v["centroid_scores"][col] for col in feature_cols])
                        dist = np.linalg.norm(user_scaled - centroid)
                        if dist < min_dist:
                            min_dist = dist
                            best_cluster = k
                            
                    # 4. Show Result
                    chosen_profile = profiles[best_cluster]
                    st.success(f"🎉 You belong to **Cluster {best_cluster}**: {chosen_profile['profile_name']}!")
                    
                    st.markdown("### Your Top Representative Movies:")
                    
                    # Extract from markdown
                    lines = cards_md.split("\n")
                    in_cluster = False
                    movie_bullets = []
                    
                    for line in lines:
                        if line.startswith(f"## Cluster {best_cluster}:"):
                            in_cluster = True
                        elif in_cluster and line.startswith("## Cluster"):
                            break
                        elif in_cluster and line.startswith("- **") and "(Mean:" in line:
                            movie_bullets.append(line)
                            
                    if movie_bullets:
                        for m in movie_bullets:
                            st.markdown(m)
                    else:
                        st.info("Check back later or view the overview tab to see what this tribe watches!")
    else:
        st.warning("Model metadata not found. Please run the clustering pipeline again to enable the demo.")

st.sidebar.title("Configuration & Exports")
st.sidebar.info("Model details:")
st.sidebar.write("- **Method:** K-Means")
st.sidebar.write(f"- **Optimal K:** {len(profiles)}")
st.sidebar.download_button(
    label="Download cluster labels (Parquet)",
    data=open(os.path.join(TABLES_DIR, "cluster_labels_users.parquet"), "rb").read(),
    file_name="cluster_labels_users.parquet",
    mime="application/octet-stream"
)
