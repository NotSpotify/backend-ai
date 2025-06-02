def recommend_by_id_cluster(spotify_id, df, n=5):
    if 'music_genre_label' not in df.columns:
        raise ValueError("Chưa có cột 'music_genre_label' trong DataFrame")
    if spotify_id not in df['spotify_id'].values:
        return None

    song_row = df[df['spotify_id'] == spotify_id].iloc[0]
    genre_label = song_row['music_genre_label']

    candidates = df[
        (df['music_genre_label'] == genre_label) &
        (df['spotify_id'] != spotify_id)
    ].sample(n=min(n, df.shape[0]-1))  # tránh lỗi nếu ít hơn n bài

    return candidates
