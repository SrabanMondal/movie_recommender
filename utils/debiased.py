from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_unbiased_genre_embedding(genres,model , similarity_threshold=0.75):
    if not genres:
        return np.zeros(model.get_sentence_embedding_dimension())

    # Step 1: Encode all genres directly
    genre_embeddings = model.encode(genres, convert_to_numpy=True)

    if len(genre_embeddings) == 1:
        return genre_embeddings[0]

    # Step 2: Compute similarity matrix
    sim_matrix = cosine_similarity(genre_embeddings)

    # Step 3: Initialize weights
    weights = np.ones(len(genres))
    assigned = [False] * len(genres)

    for i in range(len(genres)):
        if assigned[i]:
            continue

        cluster = [i]
        assigned[i] = True

        for j in range(i + 1, len(genres)):
            if not assigned[j] and sim_matrix[i, j] > similarity_threshold:
                cluster.append(j)
                assigned[j] = True

        # Down-weight if cluster has semantically redundant genres
        cluster_size = len(cluster)
        cluster_weight = 1.0 / cluster_size
        for idx in cluster:
            weights[idx] = cluster_weight

    # Step 4: Weighted average
    weighted_sum = np.sum(genre_embeddings * weights[:, np.newaxis], axis=0)
    norm = np.linalg.norm(weighted_sum)
    if norm == 0:
        return weighted_sum
    return weighted_sum / norm