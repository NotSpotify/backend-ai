from src.recommender import recommend_by_id
from src.data_loader import load_csv, load_config
import joblib
import random

def precision_at_k(recommended_ids, ground_truth_ids, k):
    recommended_at_k = recommended_ids[:k]
    hits = [1 for item in recommended_at_k if item in ground_truth_ids]
    return sum(hits) / k if k > 0 else 0

def evaluate_precision(df, cosine_sim, k=5, num_samples=100):
    total_precision = 0
    valid_samples = 0

    if 'music_genre_label' not in df.columns:
        raise ValueError("Cột 'music_genre_label' chưa tồn tại trong DataFrame")

    for _ in range(num_samples):
        song_row = df.sample(1).iloc[0]
        test_id = song_row['spotify_id']
        genre_label = song_row['music_genre_label']

        ground_truth_ids = df[
            (df['music_genre_label'] == genre_label) &
            (df['spotify_id'] != test_id)
        ]['spotify_id'].tolist()

        if len(ground_truth_ids) == 0:
            continue

        recs = recommend_by_id(test_id, df, cosine_sim, k)
        if recs is None or recs.empty:
            continue

        predicted_ids = recs['spotify_id'].tolist()
        precision = precision_at_k(predicted_ids, ground_truth_ids, k)
        total_precision += precision
        valid_samples += 1

    avg_precision = total_precision / valid_samples if valid_samples > 0 else 0
    print(f"🎯 Average Precision@{k} (on {valid_samples} samples): {avg_precision:.4f}")

def main():
    config = load_config()
    df = load_csv(config['data']['processed'])
    cosine_sim = joblib.load(config['models']['cosine_sim'])
    
    evaluate_precision(df, cosine_sim, k=5, num_samples=100)

if __name__ == "__main__":
    main()
