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
from app_utils.data_loader import load_artifacts, load_movie_lookup, load_projection_data
from app_utils.logic import (
    build_synthetic_user_vector,
    find_nearest_cluster,
    parse_movies_from_markdown,
    project_user_into_charts,
)
from app_utils.visualizations import build_tsne_fig_with_user, build_pca3d_fig_with_user
import app_utils.ui_components as ui

# Inject CSS styles
ui.inject_custom_css()

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("🔄 Reload artifacts"):
        from app_utils.data_loader import load_artifacts_cache_clear
        load_artifacts_cache_clear()
        st.rerun()

profiles, labels_df, cards_md, metadata = load_artifacts()
movie_lookup = load_movie_lookup()

if not profiles:
    st.warning("Artifacts not found! Run `python story_a_taste_tribes.py` first.")
    st.stop()

df_tsne_base, df_pca3d_base = load_projection_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# Taste Tribes")
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
        width='stretch'
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
            display_cards_md = cards_md.replace('Pref__', '').replace('genre_', '')
            st.markdown(display_cards_md)

    st.markdown('<div class="section-header">Genre Fingerprints (Radar)</div>', unsafe_allow_html=True)
    try:
        with open(os.path.join(FIGURES_DIR, "genre_radar.html"), 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=500)
    except FileNotFoundError:
        st.info("Radar chart not found.")

    st.markdown('<div class="section-header">2D Manifold Projection (t-SNE)</div>', unsafe_allow_html=True)
    if df_tsne_base is not None:
        st.plotly_chart(build_tsne_fig_with_user(df_tsne_base), use_container_width=True, key="overview_tsne")
    else:
        try:
            with open(os.path.join(FIGURES_DIR, "cluster_scatter_tsne.html"), 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=600)
        except FileNotFoundError:
            st.info("t-SNE scatter not found. Re-run the clustering pipeline.")

    st.markdown('<div class="section-header">3D Cluster Projection (PCA)</div>', unsafe_allow_html=True)
    if df_pca3d_base is not None:
        st.plotly_chart(build_pca3d_fig_with_user(df_pca3d_base), use_container_width=True, key="overview_pca3d")
    else:
        try:
            with open(os.path.join(FIGURES_DIR, "cluster_scatter_pca_3d.html"), 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=650)
        except FileNotFoundError:
            st.info("3D scatter plot not found. Re-run the clustering pipeline.")

# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎯 Cold-Start User Simulation")
    st.markdown("Select genre categories below to construct a synthetic user vector and observe real-time cluster assignment.")
    st.markdown("---")

    if not metadata:
        st.warning("Model metadata not found. Please re-run the clustering pipeline.")
        st.stop()

    # ── Prep Session State ──
    if "selected_genres" not in st.session_state:
        st.session_state.selected_genres = set()
    if "user_tsne" not in st.session_state:
        st.session_state.user_tsne = None
    if "user_pca3d" not in st.session_state:
        st.session_state.user_pca3d = None

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
        width='content'
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

            # Project user into chart coordinate spaces
            user_tsne, user_pca3d = project_user_into_charts(user_scaled, FIGURES_DIR)
            st.session_state.user_tsne  = user_tsne
            st.session_state.user_pca3d = user_pca3d
            st.session_state.best_cluster = best_cluster

        st.markdown("---")
        st.markdown(f'<div class="tribe-badge">Cluster {best_cluster}</div>', unsafe_allow_html=True)
        display_profile_name = chosen_profile['profile_name'].replace('Pref__', '').replace('genre_', '').title()
        st.markdown(f"## 🎉 You belong to: **{display_profile_name}**")
        
        top_prefs = chosen_profile.get("top_preferences", [])
        if top_prefs:
            display_prefs = [p.replace('Pref__', '').replace('genre_', '').title() for p in top_prefs]
            st.caption("Top characteristics: " + ", ".join(display_prefs))

        st.markdown('<div class="section-header">🎬 Your Recommended Movies</div>', unsafe_allow_html=True)
        ui.render_recommended_movies(movie_items, movie_lookup)

    # ── Chart Overlay Section (always shown, updated after submit) ──────────────
    st.markdown("---")
    st.markdown("### 📍 Where do you sit among the Tribes?")
    st.caption("Submit your genres above to see your position (⭐ gold star) appear in the charts below.")

    if df_tsne_base is not None or df_pca3d_base is not None:
        col_2d, col_3d = st.columns(2)
        with col_2d:
            st.markdown("**2D t-SNE Projection**")
            if df_tsne_base is not None:
                fig_t = build_tsne_fig_with_user(
                    df_tsne_base,
                    user_tsne=st.session_state.get("user_tsne")
                )
                st.plotly_chart(fig_t, use_container_width=True, key="coldstart_tsne")
            else:
                st.info("Re-run clustering pipeline to enable this chart.")
        with col_3d:
            st.markdown("**3D PCA Projection**")
            if df_pca3d_base is not None:
                fig_p = build_pca3d_fig_with_user(
                    df_pca3d_base,
                    user_pca3d=st.session_state.get("user_pca3d")
                )
                st.plotly_chart(fig_p, use_container_width=True, key="coldstart_pca3d")
            else:
                st.info("Re-run clustering pipeline to enable this chart.")
    else:
        st.info("Chart data not available. Re-run `python story_a_taste_tribes.py` first.")
