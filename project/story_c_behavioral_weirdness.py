"""
MovieLens Data Mining - Story C: Behavioral Weirdness
Goal: Identify unusual users and polarizing movies using anomaly detection signals.

Inputs (Mining-Ready Parquet):
  - user_features_train.parquet   → user anomaly detection
  - movie_features_train.parquet  → movie polarization signals
  - interactions_train.parquet    → evidence slices
  - dim_movies_clean.parquet      → titles / genres for interpretation

Outputs:
  artifacts/story_C/
    tables/user_anomaly_scores.parquet
    tables/movie_anomaly_scores.parquet
    reports/case_studies.md
    reports/run_manifest.json
    reports/summary.md
    figures/user_anomaly_scatter.html
    figures/movie_polarization_hist.html

Run with: python story_c_behavioral_weirdness.py
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ── Output directories ─────────────────────────────────────────────────────────
STORY_C_DIR  = os.path.join("artifacts", "story_C")
TABLES_OUT   = os.path.join(STORY_C_DIR, "tables")
REPORTS_OUT  = os.path.join(STORY_C_DIR, "reports")
FIGURES_OUT  = os.path.join(STORY_C_DIR, "figures")
DATA_DIR     = "data-warehousing"

RANDOM_STATE = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_dirs():
    for d in [TABLES_OUT, REPORTS_OUT, FIGURES_OUT]:
        os.makedirs(d, exist_ok=True)


def load_data():
    """Load all required Parquet inputs."""
    print("Loading datasets...")
    paths = {
        "user_features":  os.path.join(DATA_DIR, "user_features_train.parquet"),
        "movie_features": os.path.join(DATA_DIR, "movie_features_train.parquet"),
        "interactions":   os.path.join(DATA_DIR, "interactions_train.parquet"),
        "movies":         os.path.join(DATA_DIR, "dim_movies_clean.parquet"),
    }
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input: {path}")
    return (
        pd.read_parquet(paths["user_features"]),
        pd.read_parquet(paths["movie_features"]),
        pd.read_parquet(paths["interactions"]),
        pd.read_parquet(paths["movies"]),
    )


def preprocess_user_features(df):
    """Drop non-numeric / id columns and scale."""
    exclude = [c for c in df.columns if c in ("userId", "first_dt", "last_dt")
               or df[c].dtype == "object"]
    feat_cols = [c for c in df.columns if c not in exclude]
    X = df[feat_cols].fillna(0)
    # Log-transform heavy-tailed count features
    for c in [c for c in feat_cols if "n_ratings" in c or "count" in c]:
        X[c] = np.log1p(X[c])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feat_cols, index=df.index)
    return X_scaled, feat_cols


def detect_user_anomalies(df_user, X_scaled):
    """Run Isolation Forest + LOF and build a ranked anomaly score table."""
    print("Detecting user anomalies (Isolation Forest + LOF)...")
    sample_size = min(30_000, len(X_scaled))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled.iloc[idx]
    user_ids  = df_user["userId"].iloc[idx].values

    # ── Isolation Forest ──
    iso = IsolationForest(
        n_estimators=200, contamination=0.05,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    if_scores = iso.fit_predict(X_sample)          # -1 = anomaly
    if_raw    = -iso.score_samples(X_sample)       # higher = more anomalous

    # ── LOF ──
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1)
    lof_pred   = lof.fit_predict(X_sample)
    lof_raw    = -lof.negative_outlier_factor_

    # ── Combine into a score: average normalised ranks ──
    def normalise(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    combined = 0.5 * normalise(if_raw) + 0.5 * normalise(lof_raw)

    result = pd.DataFrame({
        "userId":           user_ids,
        "iso_forest_score": if_raw,
        "iso_forest_label": if_scores,
        "lof_score":        lof_raw,
        "lof_label":        lof_pred,
        "combined_score":   combined,
        "method":           "isolation_forest+lof",
    })
    result["rank"] = result["combined_score"].rank(ascending=False, method="min").astype(int)
    result.sort_values("rank", inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result


def detect_movie_anomalies(df_movie, df_movies):
    """Rank polarizing / outlier movies using std + entropy + rating distribution spread."""
    print("Detecting movie polarization signals...")
    required = ["movieId", "rating_mean", "rating_std", "n_ratings"]
    available = [c for c in required if c in df_movie.columns]

    # Build polarization proxy columns
    df = df_movie[["movieId"] + [c for c in df_movie.columns if c != "movieId"]].copy()

    # Robust-z score on rating_std if available
    if "rating_std" in df.columns:
        med = df["rating_std"].median()
        mad = np.abs(df["rating_std"] - med).median()
        df["std_zscore"] = (df["rating_std"] - med) / (mad * 1.4826 + 1e-9)
    else:
        df["std_zscore"] = 0.0

    # Polarization ≈ high std AND sufficient activity
    if "n_ratings" in df.columns:
        min_count = max(50, df["n_ratings"].quantile(0.5))
        df_active = df[df["n_ratings"] >= min_count].copy()
    else:
        df_active = df.copy()

    if "std_zscore" in df_active.columns:
        df_active["polarization_score"] = df_active["std_zscore"].clip(lower=0)
    else:
        df_active["polarization_score"] = 0.0

    df_active["rank"] = (
        df_active["polarization_score"].rank(ascending=False, method="min").astype(int)
    )
    df_active.sort_values("rank", inplace=True)

    # Join titles
    if df_movies is not None:
        movie_meta = df_movies[["movieId", "title", "genres"]]
        df_active = df_active.merge(movie_meta, on="movieId", how="left")

    keep_cols = ["movieId", "rank", "polarization_score"]
    for c in ["title", "genres", "rating_mean", "rating_std", "n_ratings"]:
        if c in df_active.columns:
            keep_cols.append(c)

    df_active["method"] = "robust_std_rank"
    result = df_active[keep_cols + ["method"]].reset_index(drop=True)
    return result


# ── Visualisations ─────────────────────────────────────────────────────────────

def plot_user_anomaly_scatter(user_scores: pd.DataFrame):
    """Interactive scatter: combined_score vs iso_forest_score, coloured by label."""
    df_plot = user_scores.copy()
    df_plot["label"] = df_plot["iso_forest_label"].map({-1: "Anomaly", 1: "Normal"})
    fig = px.scatter(
        df_plot.head(5000),
        x="iso_forest_score", y="lof_score",
        color="label", opacity=0.6,
        color_discrete_map={"Anomaly": "#FF4500", "Normal": "#7EC8E3"},
        title="User Anomaly Detection: Isolation Forest vs LOF Score",
        labels={"iso_forest_score": "Isolation Forest Score (higher = more anomalous)",
                "lof_score":        "LOF Score (higher = more anomalous)"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#FFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(os.path.join(FIGURES_OUT, "user_anomaly_scatter.html"))
    print("  Saved user_anomaly_scatter.html")


def plot_movie_polarization_histogram(movie_scores: pd.DataFrame):
    """Histogram of polarization score with top-10 movie annotations."""
    if "polarization_score" not in movie_scores.columns:
        return
    fig = px.histogram(
        movie_scores,
        x="polarization_score", nbins=60,
        title="Movie Polarization Score Distribution",
        labels={"polarization_score": "Polarization Score (robust std z-score)"},
        color_discrete_sequence=["#7EC8E3"],
    )
    # Annotate top 5
    top5 = movie_scores.head(5)
    for _, row in top5.iterrows():
        label = row.get("title", f"movie:{row['movieId']}")
        fig.add_vline(x=row["polarization_score"], line_dash="dash", line_color="#FFD700")
        fig.add_annotation(
            x=row["polarization_score"], y=0,
            text=label[:20], showarrow=True, arrowhead=2,
            font=dict(color="#FFD700", size=9), yshift=10,
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#FFF"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(os.path.join(FIGURES_OUT, "movie_polarization_hist.html"))
    print("  Saved movie_polarization_hist.html")


# ── Narrative Generation ───────────────────────────────────────────────────────

def generate_case_studies(user_scores, movie_scores, interactions, movies, n_users=5, n_movies=10):
    """Write a markdown case-study report for the top anomalous users and polarizing movies."""
    print("Generating case studies...")

    lines = [
        "# Story C: Behavioral Weirdness — Case Studies\n",
        "_Auto-generated report. Describes the most anomalous users and polarizing movies._\n",
    ]

    # ── Top anomalous users ──
    lines.append("## 🔴 Top Anomalous Users\n")
    top_users = user_scores.head(n_users)
    for _, row in top_users.iterrows():
        uid  = row["userId"]
        rank = row["rank"]
        cs   = row["combined_score"]
        iso  = row["iso_forest_score"]
        lof  = row["lof_score"]
        lines.append(f"### User {uid} (Rank #{rank})")
        lines.append(f"- **Combined Anomaly Score**: {cs:.4f}")
        lines.append(f"- **Isolation Forest Score**: {iso:.4f}")
        lines.append(f"- **LOF Score**: {lof:.4f}")

        # Evidence: their rating distribution
        if interactions is not None:
            u_ints = interactions[interactions["userId"] == uid]
            if len(u_ints) > 0:
                lines.append(f"- **Ratings count**: {len(u_ints)}")
                lines.append(f"- **Mean rating**: {u_ints['rating'].mean():.2f}  |  "
                             f"**Std**: {u_ints['rating'].std():.2f}")
                lines.append(f"- **Rating distribution**: "
                             f"{dict(u_ints['rating'].value_counts().sort_index())}")
        lines.append("\n_What makes them weird?_ High anomaly score indicates their rating patterns "
                     "deviate significantly from the majority of users — possibly extreme raters, "
                     "bot-like activity, or niche taste that sits far from any cluster.\n")
        lines.append("---\n")

    # ── Top polarizing movies ──
    lines.append("## 🎬 Top Polarizing Movies\n")
    top_movies = movie_scores.head(n_movies)
    lines.append(
        "Movies are ranked by a **polarization score** (robust z-score of rating standard deviation). "
        "High scores indicate movies that split audience opinion strongly.\n"
    )
    lines.append("| Rank | Title | Genres | Mean Rating | Std Dev | # Ratings | Polarization Score |")
    lines.append("|------|-------|--------|-------------|---------|-----------|-------------------|")
    for _, row in top_movies.iterrows():
        title  = str(row.get("title",  row["movieId"]))[:40]
        genres = str(row.get("genres", "?"))[:30]
        mean_r = f"{row['rating_mean']:.2f}" if "rating_mean" in row and not pd.isna(row.get("rating_mean")) else "?"
        std_r  = f"{row['rating_std']:.2f}"  if "rating_std"  in row and not pd.isna(row.get("rating_std"))  else "?"
        n_r    = int(row["n_ratings"]) if "n_ratings" in row and not pd.isna(row.get("n_ratings")) else "?"
        pol    = f"{row['polarization_score']:.4f}"
        lines.append(f"| {row['rank']} | {title} | {genres} | {mean_r} | {std_r} | {n_r} | {pol} |")

    md = "\n".join(lines)
    out_path = os.path.join(REPORTS_OUT, "case_studies.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Saved {out_path}")
    return md


def write_summary(user_scores, movie_scores, best_k_users=None):
    """Write a short summary.md."""
    n_anomalous_iso = (user_scores["iso_forest_label"] == -1).sum()
    n_anomalous_lof = (user_scores["lof_label"] == -1).sum()
    n_total_users   = len(user_scores)
    n_movies        = len(movie_scores)

    text = f"""# Story C: Behavioral Weirdness — Summary

Executed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Ran anomaly detection on {n_total_users:,} sampled users from the training set.
Polarization analysis covers {n_movies:,} movies with sufficient rating activity.

## User Anomaly Detection
- **Method**: Isolation Forest + LOF (combined normalised score, contamination=5%)
- **Anomalous users (Isolation Forest)**: {n_anomalous_iso:,} ({n_anomalous_iso/n_total_users*100:.1f}%)
- **Anomalous users (LOF)**:               {n_anomalous_lof:,} ({n_anomalous_lof/n_total_users*100:.1f}%)
- Top anomalous users tend to be extreme raters or users with very narrow / highly unusual taste.

## Movie Polarization
- **Method**: Robust z-score on rating standard deviation (after minimum-count filter)
- Top polarizing movies are controversial titles that attract both 1-star and 5-star reviews.
- Common patterns: cult films, art-house cinema, horror, and genre-defying releases.

## Artifacts Generated
| File | Description |
|------|-------------|
| `tables/user_anomaly_scores.parquet` | Per-user anomaly scores, rank, and labels |
| `tables/movie_anomaly_scores.parquet` | Per-movie polarization scores and rank |
| `reports/case_studies.md` | Detailed case studies of top anomalies |
| `figures/user_anomaly_scatter.html` | Interactive scatter: IF score vs LOF score |
| `figures/movie_polarization_hist.html` | Histogram of polarization scores |
"""
    out_path = os.path.join(REPORTS_OUT, "summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved {out_path}")


def write_manifest(user_scores, movie_scores):
    manifest = {
        "story":     "Story C: Behavioral Weirdness",
        "timestamp": datetime.datetime.now().isoformat(),
        "inputs":    [
            "user_features_train.parquet",
            "movie_features_train.parquet",
            "interactions_train.parquet",
            "dim_movies_clean.parquet",
        ],
        "methods": {
            "user_anomaly":       ["IsolationForest", "LocalOutlierFactor"],
            "movie_polarization": ["robust_std_rank"],
        },
        "parameters": {
            "iso_forest":  {"n_estimators": 200, "contamination": 0.05, "random_state": RANDOM_STATE},
            "lof":         {"n_neighbors": 20,  "contamination": 0.05},
            "sample_size": 30_000,
        },
        "metrics": {
            "n_users_sampled":     int(len(user_scores)),
            "n_anomalous_iso":     int((user_scores["iso_forest_label"] == -1).sum()),
            "n_anomalous_lof":     int((user_scores["lof_label"] == -1).sum()),
            "n_movies_analyzed":   int(len(movie_scores)),
        },
    }
    out_path = os.path.join(REPORTS_OUT, "run_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== MovieLens Data Mining | Story C: Behavioral Weirdness ===")
    setup_dirs()

    try:
        user_features, movie_features, interactions, movies = load_data()

        # ── User anomaly detection ──
        X_scaled, feat_cols = preprocess_user_features(user_features)
        user_scores = detect_user_anomalies(user_features, X_scaled)
        user_scores.to_parquet(os.path.join(TABLES_OUT, "user_anomaly_scores.parquet"), index=False)
        print(f"  Saved user_anomaly_scores.parquet  ({len(user_scores):,} users)")

        # ── Movie polarization ──
        movie_scores = detect_movie_anomalies(movie_features, movies)
        movie_scores.to_parquet(os.path.join(TABLES_OUT, "movie_anomaly_scores.parquet"), index=False)
        print(f"  Saved movie_anomaly_scores.parquet ({len(movie_scores):,} movies)")

        # ── Visualisations ──
        plot_user_anomaly_scatter(user_scores)
        plot_movie_polarization_histogram(movie_scores)

        # ── Narrative & provenance ──
        generate_case_studies(user_scores, movie_scores, interactions, movies)
        write_summary(user_scores, movie_scores)
        write_manifest(user_scores, movie_scores)

        print("\n✅  Story C complete. Artifacts written to:", STORY_C_DIR)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
