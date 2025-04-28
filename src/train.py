import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from src.data_loader import load_config, load_csv

def train_recommender(df, feature_cols, model_path):
    """Tính ma trận độ tương đồng."""
    available_cols = [col for col in feature_cols if col in df.columns]
    features = df[available_cols].values
    cosine_sim = cosine_similarity(features)
    joblib.dump(cosine_sim, model_path)
    print(f"Đã lưu ma trận độ tương đồng vào {model_path}")
    return cosine_sim

def main():
    config = load_config()
    df = load_csv(config['data']['processed'])
    feature_cols = config['features']['numeric_cols'] + config['features']['additional_cols']
    train_recommender(df, feature_cols, config['models']['cosine_sim'])

if __name__ == "__main__":
    main()