import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from src.data_loader import load_config


def handle_missing_values(df, numeric_cols):
    """Xử lý giá trị thiếu."""
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df['has_preview'] = df['preview'].apply(lambda x: False if pd.isna(x) or x == 'no' else True)
    df['preview'] = df['preview'].fillna('').replace('no', '')
    df['img'] = df['img'].fillna('')
    df = df.dropna(subset=['name', 'artist', 'spotify_id'])
    return df

def handle_outliers(df):
    """Xử lý giá trị bất thường."""
    cols_0_1 = ['danceability', 'energy', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence',
                'acousticness_artist', 'danceability_artist', 'energy_artist',
                'instrumentalness_artist', 'liveness_artist', 'speechiness_artist', 
                'valence_artist']
    df[cols_0_1] = df[cols_0_1].clip(lower=0, upper=1)
    df['loudness'] = df['loudness'].clip(lower=-60, upper=0)
    return df

def normalize_features(df, numeric_cols, scaler_path):
    """Chuẩn hóa đặc trưng số."""
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump(scaler, scaler_path)
    return df

def encode_categorical(df, encoder_path):
    """Mã hóa cột artist."""
    le = LabelEncoder()
    df['artist_encoded'] = le.fit_transform(df['artist'])
    joblib.dump(le, encoder_path)
    return df

def create_new_features(df):
    """Tạo đặc trưng mới."""
    df['avg_danceability'] = (df['danceability'] + df['danceability_artist']) / 2
    df['avg_energy'] = (df['energy'] + df['energy_artist']) / 2
    return df

def preprocess_data(input_path, output_path, scaler_path, encoder_path):
    """Tiền xử lý dữ liệu và lưu kết quả."""
    config = load_config()
    numeric_cols = config['features']['numeric_cols']

    df = pd.read_csv(input_path)

    # 🔥 Lọc chỉ các bài có preview link và có ảnh
    df = df[(df['preview'].notna()) & (df['preview'] != 'no') & (df['preview'] != '')]
    df = df[df['img'].notna() & (df['img'] != '')]

    # 🔥 Lấy tối đa 1500 bài
    if len(df) > 1500:
        df = df.sample(n=1500, random_state=42)

    # Các bước xử lý như cũ
    df = handle_missing_values(df, numeric_cols)
    df = handle_outliers(df)
    df = df.drop_duplicates(subset=['spotify_id'], keep='first')
    df = normalize_features(df, numeric_cols, scaler_path)
    df = encode_categorical(df, encoder_path)
    df = create_new_features(df)

    df.to_csv(output_path, index=False)
    print(f"Đã lưu dữ liệu vào {output_path}")
    return df

def main():
    """Hàm chính để chạy tiền xử lý."""
    config = load_config()
    preprocess_data(
        input_path=config['data']['raw'],
        output_path=config['data']['processed'],
        scaler_path=config['models']['scaler'],
        encoder_path=config['models']['label_encoder']
    )


if __name__ == "__main__":
    main()