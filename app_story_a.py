"""
Streamlit Dashboard for Story A: Taste Tribes
Run with: streamlit run app_story_a.py
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Taste Tribes | MovieLens",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (Netflix-inspired) ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Netflix+Sans,Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Title */
h1 { color: #E50914 !important; letter-spacing: -0.5px; }

/* Genre card grid + Skeleton loader */
.genre-card {
    position: relative; border-radius: 8px; overflow: hidden;
    cursor: pointer; width: 100%; aspect-ratio: 2/3;
    border: 3px solid transparent;
    transition: transform 0.2s ease, border-color 0.2s ease;
    background: #33353B;
}
.genre-card:hover { transform: scale(1.04); border-color: #aaa; z-index: 10; }
.genre-card.selected { border-color: #E50914 !important; transform: scale(1.04); }
.genre-card img { width: 100%; height: 100%; object-fit: cover; display: block; position: absolute; top:0; left:0; }
.genre-card .label {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.95));
    color: #fff; font-size: 14px; font-weight: 700;
    padding: 30px 8px 10px; text-align: center;
    text-shadow: 1px 1px 2px black;
}
.genre-card .check {
    position: absolute; top: 6px; right: 6px;
    background: #E50914; border-radius: 50%;
    width: 26px; height: 26px; display: flex;
    align-items: center; justify-content: center;
    font-size: 14px; color: white; font-weight: bold;
    box-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

/* Button container alignment */
div[data-testid="column"] {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
}

/* Movie poster grid */
.movie-card {
    position: relative; border-radius: 6px; overflow: hidden;
    width: 100%; aspect-ratio: 2/3;
    background: #33353B;
    transition: transform 0.2s;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.movie-card:hover { transform: scale(1.04); z-index: 10; }
.movie-card img { width: 100%; height: 100%; object-fit: cover; display: block; position: absolute; top:0; left:0; }

/* Tribe badge */
.tribe-badge {
    display: inline-block; background: #E50914;
    color: white; font-size: 14px; font-weight: 700;
    padding: 6px 16px; border-radius: 20px; margin-bottom: 8px;
}

/* Section headers */
.section-header {
    font-size: 22px; font-weight: 700; color: #fff;
    border-left: 4px solid #E50914; padding-left: 12px;
    margin: 24px 0 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts/story_A"
REPORTS_DIR   = os.path.join(ARTIFACTS_DIR, "reports")
TABLES_DIR    = os.path.join(ARTIFACTS_DIR, "tables")
FIGURES_DIR   = os.path.join(ARTIFACTS_DIR, "figures")
DATA_DIR      = "movielens-parquet-build-2026"

TMDB_API_KEY  = os.getenv("TMDB_API_KEY", "")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_API_BASE = "https://api.themoviedb.org/3"

# Representative movies per genre (well-known titles for poster cards)
GENRE_MOVIES = {
    "Action":      "The Dark Knight",
    "Adventure":   "Indiana Jones and the Raiders of the Lost Ark",
    "Animation":   "Spirited Away",
    "Children":    "Toy Story",
    "Comedy":      "The Grand Budapest Hotel",
    "Crime":       "Goodfellas",
    "Documentary": "Planet Earth II",
    "Drama":       "The Shawshank Redemption",
    "Fantasy":     "The Lord of the Rings: The Fellowship of the Ring",
    "Film-Noir":   "Chinatown",
    "Horror":      "Get Out",
    "Imax":        "Interstellar",
    "Musical":     "La La Land",
    "Mystery":     "Knives Out",
    "Romance":     "Pride & Prejudice",
    "Sci-Fi":      "The Matrix",
    "Thriller":    "Parasite",
    "War":         "Saving Private Ryan",
    "Western":     "Django Unchained",
}

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_artifacts():
    try:
        with open(os.path.join(REPORTS_DIR, "cluster_profiles.json"), "r", encoding='utf-8') as f:
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

@st.cache_data
def load_movie_lookup():
    try:
        links  = pd.read_parquet(os.path.join(DATA_DIR, "dim_links_clean.parquet"))[["movieId","tmdbId"]]
        movies = pd.read_parquet(os.path.join(DATA_DIR, "dim_movies_clean.parquet"))[["movieId","title","genres"]]
        return pd.merge(movies, links, on="movieId", how="left")
    except Exception:
        return None

@st.cache_data(ttl=86400)
def fetch_poster_by_name(title: str) -> str | None:
    """Search TMDB by movie title, return poster URL."""
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            f"{TMDB_API_BASE}/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "language": "en-US"},
            timeout=5
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results and results[0].get("poster_path"):
                return TMDB_IMG_BASE + results[0]["poster_path"]
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_poster_by_tmdb_id(tmdb_id) -> str | None:
    if not TMDB_API_KEY or pd.isna(tmdb_id):
        return None
    try:
        r = requests.get(
            f"{TMDB_API_BASE}/movie/{int(tmdb_id)}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        if r.status_code == 200:
            path = r.json().get("poster_path")
            if path:
                return TMDB_IMG_BASE + path
    except Exception:
        pass
    return None

# ── Load data ─────────────────────────────────────────────────────────────────
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
        dist_data = [{"Name": v["profile_name"], "Size": v["size"]} for v in profiles.values()]
        fig_pie = px.pie(
            pd.DataFrame(dist_data), values='Size', names='Name', hole=0.4,
            color_discrete_sequence=["#E50914", "#B81D24", "#F5F5F1", "#999"]
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#fff", legend_font_color="#fff")
        st.plotly_chart(fig_pie, use_container_width=True)
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
    else:
        feature_cols = metadata["feature_cols"]
        scaler_mean  = np.array(metadata["scaler_mean"])
        scaler_scale = np.array(metadata["scaler_scale"])

        genre_cols     = [c for c in feature_cols if "genre_pref__" in c]
        display_genres = [c.replace("genre_pref__", "").replace("_", "-").title() for c in genre_cols]

        # ── Session state for selected genres ────────────────────────
        if "selected_genres" not in st.session_state:
            st.session_state.selected_genres = set()

        def toggle_genre(g):
            if g in st.session_state.selected_genres:
                st.session_state.selected_genres.discard(g)
            else:
                if len(st.session_state.selected_genres) < 5:
                    st.session_state.selected_genres.add(g)

        # ── Netflix-style genre card grid ─────────────────────────────
        # Pad display_genres to 20 so it makes a perfect 5x4 grid 
        genre_list = display_genres.copy()
        if len(genre_list) % 7 != 0:
            padded_slots = 7 - (len(genre_list) % 7)
            genre_list.extend([None] * padded_slots)

        COLS = 7
        genre_rows = [genre_list[i:i+COLS] for i in range(0, len(genre_list), COLS)]

        for row in genre_rows:
            cols = st.columns(COLS)
            for col, genre in zip(cols, row):
                with col:
                    if genre is None:
                        # Empty placeholder to keep the grid perfectly aligned
                        st.markdown('<div style="width:100%; aspect-ratio:2/3; background:transparent;"></div>', unsafe_allow_html=True)
                        continue

                    is_selected = genre in st.session_state.selected_genres
                    poster_url  = fetch_poster_by_name(GENRE_MOVIES.get(genre, genre + " movie"))

                    # Build the HTML card
                    selected_class = "selected" if is_selected else ""
                    check_html     = '<div class="check">✓</div>' if is_selected else ""

                    if poster_url:
                        img_html = f'<img src="{poster_url}" alt="{genre}">'
                    else:
                        img_html = f'<div style="width:100%;height:100%;background:#33353B;display:flex;align-items:center;justify-content:center;position:absolute;top:0;left:0;">🎥</div>'

                    st.markdown(f"""<div class="genre-card {selected_class}">
{img_html}
{check_html}
<div class="label">{genre}</div>
</div>""", unsafe_allow_html=True)

                    btn_label = "✓ Selected" if is_selected else "+ Select"
                    btn_type  = "primary" if is_selected else "secondary"
                    if st.button(btn_label, key=f"btn_{genre}", use_container_width=True, type=btn_type):
                        toggle_genre(genre)
                        st.rerun()

        # ── Selected genres pill display ──────────────────────────────
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
                # Build synthetic feature vector
                user_vector = np.zeros(len(feature_cols))
                if "n_ratings" in feature_cols:
                    user_vector[feature_cols.index("n_ratings")] = np.log1p(20)
                if "rating_mean" in feature_cols:
                    user_vector[feature_cols.index("rating_mean")] = 4.0

                for g in st.session_state.selected_genres:
                    raw_col = "genre_pref__" + g.lower().replace("-", "_")
                    if raw_col in feature_cols:
                        user_vector[feature_cols.index(raw_col)] = 5.0

                user_scaled = (user_vector - scaler_mean) / scaler_scale

                # Nearest centroid
                best_cluster, min_dist = None, float('inf')
                for k, v in profiles.items():
                    centroid = np.array([v["centroid_scores"][col] for col in feature_cols])
                    d = np.linalg.norm(user_scaled - centroid)
                    if d < min_dist:
                        min_dist, best_cluster = d, k

                chosen = profiles[best_cluster]

            st.markdown("---")
            st.markdown(f'<div class="tribe-badge">Cluster {best_cluster}</div>', unsafe_allow_html=True)
            st.markdown(f"## 🎉 You belong to: **{chosen['profile_name']}**")
            top_prefs = chosen.get("top_preferences", [])
            if top_prefs:
                st.caption("Top characteristics: " + ", ".join(top_prefs))

            st.markdown('<div class="section-header">🎬 Your Recommended Movies</div>', unsafe_allow_html=True)

            # Parse movies from cluster_cards.md
            lines = cards_md.split("\n")
            in_cluster, movie_items = False, []
            for line in lines:
                s = line.strip()
                if s.startswith(f"## Cluster {best_cluster}:"):
                    in_cluster = True
                elif in_cluster and s.startswith("## Cluster"):
                    break
                elif in_cluster and s.startswith("- **") and "(Mean:" in s:
                    try:
                        title_raw = s.split("**")[1]
                        genre_tag = s.split("_")[-1].strip() if "_" in s else ""
                        info      = s.split("(Mean:")[1].split(")")[0].strip() if "(Mean:" in s else ""
                        movie_items.append({"title": title_raw, "info": info, "genre": genre_tag, "raw": s})
                    except Exception:
                        pass

            if not movie_items:
                st.info("No representative movies found.")
            else:
                MOVIE_COLS = 7
                movie_rows = [movie_items[i:i+MOVIE_COLS] for i in range(0, len(movie_items), MOVIE_COLS)]
                for row in movie_rows:
                    cols = st.columns(MOVIE_COLS)
                    for col, movie in zip(cols, row):
                        with col:
                            poster_url = None
                            if movie_lookup is not None:
                                search_name = movie["title"].split("(")[0].strip()[:20]
                                matches = movie_lookup[
                                    movie_lookup["title"].str.contains(search_name, case=False, na=False, regex=False)
                                ]
                                if not matches.empty:
                                    tmdb_id = matches.iloc[0]["tmdbId"]
                                    poster_url = fetch_poster_by_tmdb_id(tmdb_id)

                            if poster_url:
                                img_html = f'<img src="{poster_url}" alt="{movie["title"]}">'
                            else:
                                img_html = f'<div style="width:100%;height:100%;background:#33353B;display:flex;align-items:center;justify-content:center;position:absolute;top:0;left:0;">🎥</div>'

                            st.markdown(f"""<div class="movie-card">
{img_html}
</div>""", unsafe_allow_html=True)

                            st.markdown(f"**{movie['title']}**")
                            if movie["info"]:
                                st.caption(f"⭐ {movie['info']}")
