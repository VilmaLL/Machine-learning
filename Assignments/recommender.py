import pandas as pd


def get_recommendations_nn(title, movie_df, tfidf_matrix, nn, top_n=50):
    indices = pd.Series(movie_df.index, index=movie_df["title"]).drop_duplicates()
    if title not in indices:
        raise ValueError(f"Title '{title}' not found in the dataset.")
    idx = indices[title]
    distances, neighbors = nn.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)

    neighbors = neighbors[0][1:]
    distances = distances[0][1:]

    recommendations = movie_df["title"].iloc[neighbors].tolist()

    return recommendations



def get_recommendations_ratings(mean_ratings,recommendations, title):
    
    target_rating = mean_ratings.get(title)

    recommendation_ratings = [(movie, mean_ratings.get(movie)) for movie in recommendations if movie != title]

    highest_rated_5 = sorted(recommendation_ratings, key=lambda x: x[1], reverse=True)[:5]
    similarily_rated_5 = sorted(recommendation_ratings, key=lambda x: abs(x[1] - target_rating))[:5]

    return highest_rated_5, similarily_rated_5
    

def get_title_rating(mean_ratings, title):
    title_rating = mean_ratings.get(title)
    return title_rating
# Add ratings to the recommendations, collaborative filtering, eller viktning