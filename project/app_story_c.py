"""
Streamlit Demo App for Story C: Behavioral Weirdness
Run with: streamlit run app_story_c.py

Displays anomaly detection results (unusual users + polarizing movies) from the
Story C mining pipeline (story_c_behavioral_weirdness.py).
"""

import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Behavioral Weirdness | MovieLens",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
STORY_C_DIR = os.path.join("artifacts", "story_C")
TABLES_DIR  = os.path.join(STORY_C_DIR, "tables")
REPORTS_DIR = os.path.join(STORY_C_DIR, "reports")
FIGURES_DIR = os.path.join(STORY_C_DIR, "figures")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 4px solid #FF4500;
    color: #FF8C69;
    padding: 10px 18px;
    border-radius: 6px;
    margin: 20px 0 12px 0;
    font-size: 1.1rem;
    font-weight: 700;
}
.metric-card {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_story_c_artifacts():
    user_path  = os.path.join(TABLES_DIR, "user_anomaly_scores.parquet")
    movie_path = os.path.join(TABLES_DIR, "movie_anomaly_scores.parquet")
    manifest_path = os.path.join(REPORTS_DIR, "run_manifest.json")
    summary_path  = os.path.join(REPORTS_DIR, "summary.md")
    cases_path    = os.path.join(REPORTS_DIR, "case_studies.md")

    user_scores  = pd.read_parquet(user_path)  if os.path.exists(user_path)  else None
    movie_scores = pd.read_parquet(movie_path) if os.path.exists(movie_path) else None
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    summary  = open(summary_path).read()      if os.path.exists(summary_path) else ""
    cases    = open(cases_path).read()        if os.path.exists(cases_path)   else ""
    return user_scores, movie_scores, manifest, summary, cases

user_scores, movie_scores, manifest, summary_md, cases_md = load_story_c_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔍 Behavioral Weirdness")
    st.markdown("**Story C** — Anomaly Detection")
    st.markdown("---")
    if manifest:
        st.caption("Last run:")
        st.write(manifest.get("timestamp", "?")[:19])
        metrics = manifest.get("metrics", {})
        st.metric("Users analysed", f"{metrics.get('n_users_sampled', '?'):,}")
        st.metric("Anomalies (ISO)", f"{metrics.get('n_anomalous_iso', '?'):,}")
        st.metric("Movies analysed", f"{metrics.get('n_movies_analyzed', '?'):,}")
    else:
        st.warning("Run `python story_c_behavioral_weirdness.py` first.")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔍 Story C: Behavioral Weirdness")
st.markdown("Identifies **unusual users** (via Isolation Forest + LOF) and **polarizing movies** (robust std ranking).")

if user_scores is None or movie_scores is None:
    st.error("Artifacts not found. Run `python story_c_behavioral_weirdness.py` first.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👤 User Anomalies", "🎬 Polarizing Movies", "📄 Reports"])

# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">User Anomaly Scatter</div>', unsafe_allow_html=True)
    scatter_path = os.path.join(FIGURES_DIR, "user_anomaly_scatter.html")
    if os.path.exists(scatter_path):
        with open(scatter_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=520)
    else:
        st.info("Scatter chart not found — re-run the pipeline.")

    st.markdown('<div class="section-header">Top Anomalous Users</div>', unsafe_allow_html=True)
    n_top = st.slider("Show top N", 5, 50, 10)
    cols_to_show = ["rank", "userId", "combined_score", "iso_forest_score", "lof_score", "iso_forest_label"]
    cols_available = [c for c in cols_to_show if c in user_scores.columns]
    df_display = user_scores.head(n_top)[cols_available].copy()
    df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
    st.dataframe(df_display, use_container_width=True)

    st.download_button(
        "📥 Download User Anomaly Scores",
        data=open(os.path.join(TABLES_DIR, "user_anomaly_scores.parquet"), "rb").read(),
        file_name="user_anomaly_scores.parquet",
        mime="application/octet-stream",
        width="content",
    )

# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Polarization Score Histogram</div>', unsafe_allow_html=True)
    hist_path = os.path.join(FIGURES_DIR, "movie_polarization_hist.html")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=480)
    else:
        st.info("Histogram not found — re-run the pipeline.")

    st.markdown('<div class="section-header">Top Polarizing Movies</div>', unsafe_allow_html=True)
    n_movies = st.slider("Show top N movies", 10, 100, 20)
    movie_cols = ["rank", "title", "genres", "rating_mean", "rating_std", "n_ratings", "polarization_score"]
    movie_cols_avail = [c for c in movie_cols if c in movie_scores.columns]
    df_movies_display = movie_scores.head(n_movies)[movie_cols_avail].copy()
    for col in ["rating_mean", "rating_std", "polarization_score"]:
        if col in df_movies_display.columns:
            df_movies_display[col] = df_movies_display[col].round(3)
    st.dataframe(df_movies_display, use_container_width=True)

    st.download_button(
        "📥 Download Movie Anomaly Scores",
        data=open(os.path.join(TABLES_DIR, "movie_anomaly_scores.parquet"), "rb").read(),
        file_name="movie_anomaly_scores.parquet",
        mime="application/octet-stream",
        width="content",
    )

# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Summary Report</div>', unsafe_allow_html=True)
    st.markdown(summary_md)

    st.markdown('<div class="section-header">Case Studies</div>', unsafe_allow_html=True)
    st.markdown(cases_md)
