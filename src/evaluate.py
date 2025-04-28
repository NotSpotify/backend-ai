from src.recommender import recommend_by_id
from src.data_loader import load_csv, load_config
import joblib

def evaluate_recommender(sp_id, df, cosine_sim, n=5):
    """Kiểm tra gợi ý cho một spotify_id."""
    recommendations = recommend_by_id(sp_id, df, cosine_sim, n)
    if recommendations is None:
        print(f"Không tìm thấy bài hát với spotify_id: {sp_id}")
    else:
        print(f"Gợi ý cho bài hát {sp_id}:")
        print(recommendations)

def main():
    config = load_config()
    df = load_csv(config['data']['processed'])
    cosine_sim = joblib.load(config['models']['cosine_sim'])
    
    # Kiểm tra vài bài hát ngẫu nhiên
    test_ids = df['spotify_id'].sample(3).tolist()
    for sp_id in test_ids:
        evaluate_recommender(sp_id, df, cosine_sim, config['recommender']['top_n'])

if __name__ == "__main__":
    main()