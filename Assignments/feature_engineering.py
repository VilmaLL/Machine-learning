from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def filter_out_suspect_users(df):
    grouped = df.groupby("title")["rating"]
    mean_ratings = grouped.mean()
    five_star_counts = df[df["rating"] == 5.0].groupby("title").size()
    top_5 = five_star_counts.sort_values(ascending=False).head(5).index
    
    bad_movies = [title for title in five_star_counts.index if five_star_counts[title] <= 1 and mean_ratings[title] < 1.5]

    contradict_top5 = df[(df["title"].isin(top_5)) & (df["rating"] < 1.0)]
    contradict_bad_movies = df[(df["title"].isin(bad_movies)) & (df["rating"] > 3.0)]
    
    user_ids = set(contradict_top5["userId"]) & set(contradict_bad_movies["userId"])

    cleaned_df = df[~df["userId"].isin(user_ids)].copy()
    df.drop(columns=["tags"], inplace=True)
    return cleaned_df, mean_ratings


def create_tfidf_matrix(cleaned_df):
    movie_df = cleaned_df.groupby("title")["tags"].apply(lambda x: " ".join(x)).reset_index()
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movie_df['tags'])
    movie_df.drop(columns=["tags"], inplace=True)
    cleaned_df.drop(columns=["tags"], inplace=True)
    return tfidf_matrix, movie_df


def nearest_neighbors_model(tfidf_matrix, n_neighbors=50):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(tfidf_matrix)
    return nn


def get_nearest_neighbors(movie_index, tfidf_matrix, nn, top_n=50):
    distances, indices = nn.kneighbors(tfidf_matrix[movie_index:movie_index+1], n_neighbors=top_n+1)
    return indices[0][1:], distances[0][1:]