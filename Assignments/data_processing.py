import pandas as pd

def get_df():
    path_movies = r"C:\Users\vilma\Machine-learning\Assignments\ml-latest\ml-latest\movies.csv"
    path_ratings = r"C:\Users\vilma\Machine-learning\Assignments\ml-latest\ml-latest\ratings.csv"
    path_tags = r"C:\Users\vilma\Machine-learning\Assignments\ml-latest\ml-latest\tags.csv"

    dtype = {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "tag": "string",
        "genre": "string",
        "title": "string"
    }

    movies = pd.read_csv(path_movies, dtype=dtype)
    ratings = pd.read_csv(path_ratings, usecols=["userId", "movieId", "rating"], dtype=dtype)
    tags = pd.read_csv(path_tags, usecols=["userId", "movieId", "tag"], dtype=dtype)

    movies["title"] = movies["title"].astype("category")

    movies_merged_dups = movies.groupby("title").agg({"movieId": "first", "genres": lambda x: ", ".join(sorted(set(x)))}).reset_index()

    df = pd.merge(ratings, movies_merged_dups, on="movieId", how="left")
    df = pd.merge(df, tags, on=["movieId", "userId"], how="left")

    df["tag"] = df["tag"].fillna('')
    df["genres"] = df["genres"].fillna('')
    df["tags"] = df["tag"] + " " + df["genres"].str.replace("|", " ", regex=False)

    df.drop(columns=["tag", "genres"], inplace=True)
    df.dropna(subset=["title", "rating", "userId", "movieId"], inplace=True)

    return df