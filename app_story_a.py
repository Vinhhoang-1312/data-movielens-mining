"""
Streamlit Dashboard for Story A: Taste Tribes
Run with: streamlit run app_story_a.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Taste Tribes | MovieLens",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# ── Custom Modules ────────────────────────────────────────────────────────────
from app_utils.config import TABLES_DIR, FIGURES_DIR, TMDB_API_KEY
from app_utils.data_loader import load_artifacts, load_movie_lookup
from app_utils.logic import build_synthetic_user_vector, find_nearest_cluster, parse_movies_from_markdown
import app_utils.ui_components as ui

# Inject CSS styles
ui.inject_custom_css()

# ── Load Data ─────────────────────────────────────────────────────────────────
profiles, labels_df, cards_md, metadata = load_artifacts()
movie_lookup = load_movie_lookup()

if not profiles:
    st.warning("Artifacts not found! Run `python story_a_taste_tribes.py` first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎬 Taste Tribes")
    st.markdown("---")
    st.write(f"- **Method:** K-Means")
    st.write(f"- **Optimal K:** {len(profiles)}")
    if TMDB_API_KEY:
        st.success("✅ TMDB API connected")
    else:
        st.warning("⚠️ Set TMDB_API_KEY in .env")
    st.markdown("---")
    st.download_button(
        "📥 Download Cluster Labels",
        data=open(os.path.join(TABLES_DIR, "cluster_labels_users.parquet"), "rb").read(),
        file_name="cluster_labels_users.parquet",
        mime="application/octet-stream",
        use_container_width=True
    )

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎬 Story A: Taste Tribes")
st.markdown("User segmentation demo — cluster users into interpretable preference groups.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Overview & Analytics", "🚀 Cold-Start Demo"])

# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Tribe Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        ui.render_pie_chart(profiles)
    with col2:
        with st.expander("📄 Cluster Cards", expanded=True):
            st.markdown(cards_md)

    st.markdown('<div class="section-header">Genre Fingerprints (Radar)</div>', unsafe_allow_html=True)
    try:
        with open(os.path.join(FIGURES_DIR, "genre_radar.html"), 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=500)
    except FileNotFoundError:
        st.info("Radar chart not found.")

    st.markdown('<div class="section-header">2D Manifold Projection (t-SNE)</div>', unsafe_allow_html=True)
    try:
        with open(os.path.join(FIGURES_DIR, "cluster_scatter_tsne.html"), 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=600)
    except FileNotFoundError:
        st.info("t-SNE scatter not found.")

# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎯 What kind of movies do you love?")
    st.markdown("Click on the genres you enjoy — just like Netflix onboarding. Then discover your Taste Tribe!")
    st.markdown("---")

    if not metadata:
        st.warning("Model metadata not found. Please re-run the clustering pipeline.")
        st.stop()

    # ── Prep Session State ──
    if "selected_genres" not in st.session_state:
        st.session_state.selected_genres = set()

    def toggle_genre(g):
        if g in st.session_state.selected_genres:
            st.session_state.selected_genres.discard(g)
        else:
            if len(st.session_state.selected_genres) < 5:
                st.session_state.selected_genres.add(g)

    # ── Render Genre Selector ──
    feature_cols = metadata["feature_cols"]
    genre_cols = [c for c in feature_cols if "genre_pref__" in c]
    display_genres = [c.replace("genre_pref__", "").replace("_", "-").title() for c in genre_cols]
    
    ui.render_genre_selector_grid(display_genres, toggle_genre)

    # ── Perform Calculation ──
    n_selected = len(st.session_state.selected_genres)
    if n_selected > 0:
        pills = "  ".join([f"`{g}`" for g in st.session_state.selected_genres])
        st.markdown(f"**Selected ({n_selected}/5):** {pills}")
    else:
        st.caption("No genre selected yet — pick at least 1.")

    st.markdown("---")
    go = st.button(
        "🔍 Find My Taste Tribe!",
        type="primary",
        disabled=(n_selected == 0),
        use_container_width=False
    )

    if go and n_selected > 0:
        with st.spinner("Crunching the numbers..."):
            user_scaled = build_synthetic_user_vector(
                st.session_state.selected_genres,
                feature_cols,
                metadata["scaler_mean"],
                metadata["scaler_scale"]
            )
            
            best_cluster, chosen_profile = find_nearest_cluster(user_scaled, profiles, feature_cols)
            movie_items = parse_movies_from_markdown(cards_md, best_cluster)

        st.markdown("---")
        st.markdown(f'<div class="tribe-badge">Cluster {best_cluster}</div>', unsafe_allow_html=True)
        st.markdown(f"## 🎉 You belong to: **{chosen_profile['profile_name']}**")
        
        top_prefs = chosen_profile.get("top_preferences", [])
        if top_prefs:
            st.caption("Top characteristics: " + ", ".join(top_prefs))

        st.markdown('<div class="section-header">🎬 Your Recommended Movies</div>', unsafe_allow_html=True)
        
        # ── Render Final Movies ──
        ui.render_recommended_movies(movie_items, movie_lookup)
