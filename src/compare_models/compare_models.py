import joblib
import pandas as pd
from src.data_loader import load_config, load_csv
from src.compare_models.recommender import recommend_by_id
from src.compare_models.recommend_by_id_euclidean import recommend_by_id_euclidean
from src.compare_models.recommend_by_id_cluster import recommend_by_id_cluster
from src.compare_models.recommend_by_id_random import recommend_by_id_random

def precision_at_k(recommended_ids, ground_truth_ids, k):
    hits = [1 for item in recommended_ids[:k] if item in ground_truth_ids]
    return sum(hits) / k if k > 0 else 0

def evaluate_model(df, recommender_func, k, num_samples, cosine_sim=None, feature_cols=None):
    total_precision = 0
    valid_samples = 0

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

        try:
            if recommender_func.__name__ == "recommend_by_id":
                recs = recommender_func(test_id, df, cosine_sim, n=k)
            elif recommender_func.__name__ == "recommend_by_id_euclidean":
                recs = recommender_func(test_id, df, n=k, feature_cols=feature_cols)
            else:
                recs = recommender_func(test_id, df, n=k)
        except:
            continue

        if recs is None or recs.empty:
            continue

        predicted_ids = recs['spotify_id'].tolist()
        precision = precision_at_k(predicted_ids, ground_truth_ids, k)
        total_precision += precision
        valid_samples += 1

    return total_precision / valid_samples if valid_samples > 0 else 0

def main():
    config = load_config()
    df = load_csv(config['data']['processed'])
    cosine_sim = joblib.load(config['models']['cosine_sim'])
    feature_cols = config['features']['numeric_cols'] + config['features']['additional_cols']

    models = {
        "Cosine Similarity": lambda sid, df_, n=5: recommend_by_id(sid, df_, cosine_sim, n),
        "Euclidean Distance": lambda sid, df_, n=5: recommend_by_id_euclidean(sid, df_, n, feature_cols),
        "Cluster-Based": recommend_by_id_cluster,
        "Random Recommend": recommend_by_id_random
    }

    results = []
    for name, model_func in models.items():
        print(f"⏳ Đang đánh giá mô hình: {name}")
        score = evaluate_model(df, model_func, k=5, num_samples=100, cosine_sim=cosine_sim, feature_cols=feature_cols)
        results.append((name, round(score, 4)))

    print("\n📊 KẾT QUẢ SO SÁNH PRECISION@5:")
    print("{:<20} {:>10}".format("Mô hình", "Precision@5"))
    print("-" * 32)
    for name, score in results:
        print("{:<20} {:>10}".format(name, score))

if __name__ == "__main__":
    main()
