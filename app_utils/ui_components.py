import streamlit as st
import pandas as pd
import plotly.express as px
from .config import GENRE_MOVIES
from .data_loader import fetch_poster_by_name, fetch_poster_by_tmdb_id

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Netflix+Sans,Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Title */
    h1 { color: #E50914 !important; letter-spacing: -0.5px; }

    /* Genre card grid */
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

def render_pie_chart(profiles):
    """Renders the cluster distribution pie chart."""
    dist_data = [{"Name": v["profile_name"], "Size": v["size"]} for v in profiles.values()]
    fig_pie = px.pie(
        pd.DataFrame(dist_data), values='Size', names='Name', hole=0.4,
        color_discrete_sequence=["#E50914", "#B81D24", "#F5F5F1", "#999"]
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fff", 
        legend_font_color="#fff",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

def render_genre_selector_grid(display_genres, toggle_func):
    """Renders the Netflix-style clickable genre grid."""
    # Pad display_genres to to make a perfect grid 
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
                    toggle_func(genre)
                    st.rerun()

def render_recommended_movies(movie_items, movie_lookup):
    """Renders the Netflix-style Recommended Movies output."""
    if not movie_items:
        st.info("No representative movies found.")
        return

    MOVIE_COLS = 7
    movie_rows = [movie_items[i:i+MOVIE_COLS] for i in range(0, len(movie_items), MOVIE_COLS)]
    
    for row in movie_rows:
        cols = st.columns(MOVIE_COLS)
        for col, movie in zip(cols, row):
            with col:
                poster_url = None
                if movie_lookup is not None:
                    # Search TMDB lookup table
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
