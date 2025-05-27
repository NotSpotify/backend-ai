import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib
from src.data_loader import load_config

def handle_missing_values(df, numeric_cols):
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df['has_preview'] = df['preview'].apply(lambda x: False if pd.isna(x) or x == 'no' else True)
    df['preview'] = df['preview'].fillna('').replace('no', '')
    df['img'] = df['img'].fillna('')
    df = df.dropna(subset=['name', 'artist', 'spotify_id'])
    return df

def handle_outliers(df):
    cols_0_1 = ['danceability', 'energy', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence',
                'acousticness_artist', 'danceability_artist', 'energy_artist',
                'instrumentalness_artist', 'liveness_artist', 'speechiness_artist', 
                'valence_artist']
    df[cols_0_1] = df[cols_0_1].clip(lower=0, upper=1)
    df['loudness'] = df['loudness'].clip(lower=-60, upper=0)
    return df

def normalize_features(df, numeric_cols, scaler_path):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump(scaler, scaler_path)
    return df

def encode_categorical(df, encoder_path):
    le = LabelEncoder()
    df['artist_encoded'] = le.fit_transform(df['artist'])
    joblib.dump(le, encoder_path)
    return df

def create_new_features(df):
    df['avg_danceability'] = (df['danceability'] + df['danceability_artist']) / 2
    df['avg_energy'] = (df['energy'] + df['energy_artist']) / 2
    return df

def cluster_music(df, numeric_cols, n_clusters=6):
    kmeans_features = df[numeric_cols]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['music_cluster'] = kmeans.fit_predict(kmeans_features)
    cluster_labels = {
        0: "Energetic",
        1: "Chill",
        2: "Acoustic",
        3: "Dance",
        4: "Experimental",
        5: "Emotional"
    }
    df['music_genre_label'] = df['music_cluster'].map(cluster_labels)
    return df

def preprocess_data(input_path, output_path, scaler_path, encoder_path):
    config = load_config()
    numeric_cols = config['features']['numeric_cols']

    df = pd.read_csv(input_path)

    # Lọc bài có preview và ảnh
    df = df[(df['preview'].notna()) & (df['preview'] != 'no') & (df['preview'] != '')]
    df = df[df['img'].notna() & (df['img'] != '')]

    # Giới hạn 1500 bài hát
    if len(df) > 1500:
        df = df.sample(n=1500, random_state=42)

    df = handle_missing_values(df, numeric_cols)
    df = handle_outliers(df)
    df = df.drop_duplicates(subset=['spotify_id'], keep='first')
    df = normalize_features(df, numeric_cols, scaler_path)
    df = encode_categorical(df, encoder_path)
    df = create_new_features(df)
    df = cluster_music(df, numeric_cols)

    df.to_csv(output_path, index=False)
    print(f"\u2714\ufe0f Dã luu du lieu vao {output_path}")
    return df

def main():
    config = load_config()
    preprocess_data(
        input_path=config['data']['raw'],
        output_path=config['data']['processed'],
        scaler_path=config['models']['scaler'],
        encoder_path=config['models']['label_encoder']
    )

if __name__ == "__main__":
    main()
