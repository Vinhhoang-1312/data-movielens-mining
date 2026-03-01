import numpy as np

def build_synthetic_user_vector(selected_genres, feature_cols, scaler_mean, scaler_scale):
    """
    Simulates a "Cold Start" user by placing high score values (5.0) in the user's
    selected genres, and default values for standard metrics like n_ratings.
    """
    user_vector = np.zeros(len(feature_cols))
    
    # Typical baseline activity metrics
    if "n_ratings" in feature_cols:
        user_vector[feature_cols.index("n_ratings")] = np.log1p(20)
    if "rating_mean" in feature_cols:
        user_vector[feature_cols.index("rating_mean")] = 4.0

    # Max out the selected genres
    for g in selected_genres:
        raw_col = "genre_pref__" + g.lower().replace("-", "_")
        if raw_col in feature_cols:
            user_vector[feature_cols.index(raw_col)] = 5.0

    # Standardize it based on how the KMeans model was trained
    user_scaled = (user_vector - scaler_mean) / scaler_scale
    return user_scaled

def find_nearest_cluster(user_scaled, profiles, feature_cols):
    """
    Computes Euclidean distance between the simulated user vector and all KMeans centroids.
    Returns the nearest cluster ID (best_cluster) and distance.
    """
    best_cluster, min_dist = None, float('inf')
    
    for k, v in profiles.items():
        centroid = np.array([v["centroid_scores"][col] for col in feature_cols])
        d = np.linalg.norm(user_scaled - centroid)
        if d < min_dist:
            min_dist, best_cluster = d, k
            
    return best_cluster, profiles[best_cluster]

def parse_movies_from_markdown(cards_md, cluster_id):
    """
    Parses `cluster_cards.md` and strips out the list of representative 
    movies for the selected cluster_id.
    """
    lines = cards_md.split("\n")
    in_cluster, movie_items = False, []
    
    for line in lines:
        s = line.strip()
        if s.startswith(f"## Cluster {cluster_id}:"):
            in_cluster = True
        elif in_cluster and s.startswith("## Cluster"):
            break
        elif in_cluster and s.startswith("- **") and "(Mean:" in s:
            try:
                title_raw = s.split("**")[1]
                genre_tag = s.split("_")[-1].strip() if "_" in s else ""
                info      = s.split("(Mean:")[1].split(")")[0].strip() if "(Mean:" in s else ""
                movie_items.append({
                    "title": title_raw, 
                    "info": info, 
                    "genre": genre_tag, 
                    "raw": s
                })
            except Exception:
                pass
                
    return movie_items
