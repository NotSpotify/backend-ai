import pandas as pd
import joblib
from src.data_loader import load_csv, load_config

def recommend_songs(df, cosine_sim, song_idx, n=5):
    """Gợi ý top N bài hát tương tự + thêm label thể loại."""
    sim_scores = list(enumerate(cosine_sim[song_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_n = sim_scores[1:n+1] 

    recommendations = df.iloc[[i[0] for i in top_n]][[
        'name', 'artist', 'spotify_id', 'preview', 'img', 'music_genre_label'
    ]].copy()
    recommendations['similarity_score'] = [i[1] for i in top_n]
    return recommendations
    
def recommend_by_id(sp_id, df, cosine_sim, n=5):
    """Gợi ý dựa trên spotify_id."""
    song_idx = df.index[df['spotify_id'] == sp_id].tolist()
    if not song_idx:
        return None
    return recommend_songs(df, cosine_sim, song_idx[0], n)

def main():
    config = load_config()
    df = load_csv(config['data']['processed'])
    cosine_sim = joblib.load(config['models']['cosine_sim'])
    
    # Ví dụ gợi ý
    example_id = df['spotify_id'].iloc[0]
    recommendations = recommend_by_id(example_id, df, cosine_sim, config['recommender']['top_n'])
    print(f"Gợi ý cho bài hát {example_id}:")
    print(recommendations)

if __name__ == "__main__":
    main()