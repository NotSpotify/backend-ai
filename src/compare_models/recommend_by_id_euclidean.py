from sklearn.metrics.pairwise import euclidean_distances

def recommend_by_id_euclidean(spotify_id, df, n=5, feature_cols=None):
    if spotify_id not in df['spotify_id'].values:
        return None
    
    if feature_cols is None:
        raise ValueError("Cần truyền vào feature_cols")

    idx = df[df['spotify_id'] == spotify_id].index[0]
    features = df[feature_cols].values
    distances = euclidean_distances([features[idx]], features)[0]

    df['euclidean_distance'] = distances
    recs = df[df['spotify_id'] != spotify_id].sort_values(by='euclidean_distance').head(n)
    return recs.drop(columns=['euclidean_distance'])
 