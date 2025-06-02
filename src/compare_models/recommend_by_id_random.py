def recommend_by_id_random(spotify_id, df, n=5):
    if spotify_id not in df['spotify_id'].values:
        return None

    recs = df[df['spotify_id'] != spotify_id].sample(n=n)
    return recs
