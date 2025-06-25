from collections import Counter
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.debiased import get_unbiased_genre_embedding
from typing import List, Optional
# Load models
plot_genre_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load embeddings and metadata
plot_embeddings = np.load('../data/plot_embeddings.npy', mmap_mode='r')
genre_embeddings = np.load('../data/genre_embeddings.npy', mmap_mode='r')
topic_embeddings = np.load('../data/topic_embeddings.npy', mmap_mode='r')
metadata = pd.read_parquet("../data/movie_metadata.parquet")
udf = pd.read_parquet("../data/user_metadata.parquet")

# Load indices
combined_index = faiss.read_index("../data/combined_index.faiss")
user_index = faiss.read_index("../data/user_index.faiss")

# Embedding generation functions
def get_plot_embedding(text):
    return plot_genre_model.encode([text], convert_to_numpy=True)[0]

def get_topic_embedding(text):
    return topic_model.encode([text], convert_to_numpy=True)[0]

def get_genre_embedding(genres):
    genre_text = ", ".join(genres)
    return plot_genre_model.encode([genre_text], convert_to_numpy=True)[0]

# Recommendation function
def recommend(
    movie_names: List[str],
    provided_plots: Optional[List[str]] = None,
    provided_genres: Optional[List[List[str]]] = None,
    top_k: int = 20,
    language: Optional[str] = None
):
    user_plot_vecs = []
    user_genre_vecs = []
    user_topic_vecs = []
    user_all_genres=[]
    for idx, movie_name in enumerate(movie_names):
        # 1. Lookup metadata if not provided
        match = metadata[metadata['title'].str.lower() == movie_name.lower()]
        if match.empty:
            if not (provided_plots and idx < len(provided_plots)):
                continue

        plot = provided_plots[idx] if provided_plots and idx < len(provided_plots) else match.iloc[0]["plot"]
        genres = provided_genres[idx] if provided_genres and idx < len(provided_genres) else match.iloc[0]["genres"]

        # 2. Encode plot
        plot_vec = plot_genre_model.encode(plot, normalize_embeddings=True)
        user_plot_vecs.append(plot_vec)

        # 3. Encode genre tags
        genre_vecs = []
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",") if g.strip()]
        user_all_genres.extend(genres)
        #genre_avg = np.mean(genre_vecs, axis=0)
        #genre_avg = genre_avg / np.linalg.norm(genre_avg)
        #user_genre_vecs.append(genre_avg)

        # 4. Topic embedding from plot
        topic_vec = topic_model.encode(plot, normalize_embeddings=True)
        user_topic_vecs.append(topic_vec)
    if not user_plot_vecs:
        return []
    # 5. Average all user embeddings
    plot_avg = np.mean(user_plot_vecs, axis=0)
    genre_avg = get_unbiased_genre_embedding(user_all_genres, plot_genre_model)
    topic_avg = np.mean(user_topic_vecs, axis=0)

    # 6. Final combined embedding (1536)
    user_combined = np.concatenate([plot_avg, genre_avg, topic_avg])
    user_combined = user_combined / np.linalg.norm(user_combined)

    # 7. FAISS Search
    D, I = combined_index.search(np.expand_dims(user_combined, axis=0), 200)  # top 200 candidates
    matched_idxs = I[0]

    # 8. Score matched movies using individual sim
    plot_scores = np.dot(plot_embeddings[matched_idxs], plot_avg)
    genre_scores = np.dot(genre_embeddings[matched_idxs], genre_avg)
    topic_scores = np.dot(topic_embeddings[matched_idxs], topic_avg)
    combined_scores = (
        0.2 * D[0] +        # from FAISS combined similarity
        0.3 * plot_scores +
        0.3 * genre_scores +
        0.2 * topic_scores
    )

    # 9. Get matched metadata
    result = metadata.iloc[matched_idxs].copy()
    result["score"] = combined_scores
    result["final_score"] = result["score"]*0.8+result["ratings"]*0.02
    if language:
        result = result[result["lan"].str.lower() == language.lower()]
    # 10. Sort and return top_k
    watched_titles = set([t.lower() for t in movie_names])
    result = result[~result["title"].str.lower().isin(watched_titles)]
    result = result.sort_values("final_score", ascending=False).head(top_k)

    return result[["title", "genres", "final_score"]]

# Collaborative filtering

def recommend_from_users(
    movie_names: List[str],
    top_k_users: int = 10,
    top_k_movies: int = 20,
    language: Optional[str] = None,
    provided_plots: Optional[List[str]] = None,
    provided_genres: Optional[List[List[str]]] = None
):
    user_plot_vecs = []
    user_topic_vecs = []
    user_all_genres = []

    for idx, movie_name in enumerate(movie_names):
        match = metadata[metadata['title'].str.lower() == movie_name.lower()]
        # if match.empty:
        #     continue
        plot = provided_plots[idx] if provided_plots and idx < len(provided_plots) else match.iloc[0]["plot"]
        genres = provided_genres[idx] if provided_genres and idx < len(provided_genres) else match.iloc[0]["genres"]
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",") if g.strip()]
        user_all_genres.extend(genres)

        plot_vec = plot_genre_model.encode(plot, normalize_embeddings=True)
        topic_vec = topic_model.encode(plot, normalize_embeddings=True)
        user_plot_vecs.append(plot_vec)
        user_topic_vecs.append(topic_vec)

    if not user_plot_vecs:
        return []

    plot_avg = np.mean(user_plot_vecs, axis=0)
    genre_avg = get_unbiased_genre_embedding(user_all_genres, plot_genre_model)
    topic_avg = np.mean(user_topic_vecs, axis=0)

    user_combined = np.concatenate([plot_avg, genre_avg, topic_avg])
    user_combined = user_combined / np.linalg.norm(user_combined)

    # 🔁 Search top similar users in FAISS user index
    D, I = user_index.search(np.expand_dims(user_combined, axis=0), top_k_users)
    matched_user_idxs = I[0]

    # 🎯 Get user IDs from udf index mapping
    user_ids = udf.reset_index().iloc[matched_user_idxs]['user_id'].values

    # 🧠 Count movie frequency among similar users
    movie_counter = Counter()
    for uid in user_ids:
        user_movies = udf.loc[uid]['movies']
        for movie in user_movies:
            movie_lower = movie.lower()
            if movie_lower not in [m.lower() for m in movie_names]:
                movie_counter[movie] += 1

    if not movie_counter:
        return []

    # 📋 Select top N most frequent movies
    top_movies = [movie for movie, count in movie_counter.most_common(top_k_movies * 2)]

    # 📄 Filter metadata
    result = metadata[metadata['title'].isin(top_movies)].copy()

    # Filter by language
    if language:
        result = result[result["lan"].str.lower() == language.lower()]

    if result.empty:
        return []

    # 🏁 Add frequency score
    result['freq_score'] = result['title'].apply(lambda t: movie_counter.get(t, 0))

    # 🧮 Final score: combine freq + ratings (you can tweak weights)
    result['final_score'] = (
        0.8 * result['freq_score'] +
        0.2 * result['ratings']  # optional: or normalized rating
    )

    # 🔽 Sort and return top_k
    result = result.sort_values("final_score", ascending=False).head(top_k_movies)

    return result[["title", "genres","final_score"]]
