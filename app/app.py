from fastapi import Query
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import numpy as np
from src.data_loader import load_csv, load_config
from src.recommender import recommend_by_id
import math

app = FastAPI(title="Music Recommender API")

config = load_config()
df = load_csv(config['data']['processed'])
cosine_sim = joblib.load(config['models']['cosine_sim'])

# async def recommend(sp_id: str, n: int = config['recommender']['top_n']):

@app.get("/recommend")
async def recommend(
    sp_id: str = Query(default="0b4KsOdnrZlfE5VUAAzxv1"),
    n: int = Query(default=10)
):
    """Recommend n bài hát tương tự sp_id."""
    recommendations = recommend_by_id(sp_id, df, cosine_sim, n)
    if recommendations is None:
        raise HTTPException(status_code=404, detail="Bài hát không tồn tại")

    # Replace NaN, inf
    recommendations = recommendations.replace([np.inf, -np.inf], 0).fillna('')
    for col in recommendations.columns:
        recommendations[col] = recommendations[col].apply(
            lambda x: 0 if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x
        )

    return recommendations.to_dict(orient='records')

@app.get("/random_songs")
async def get_random_songs(n: int = 20):
     """Lấy ngẫu nhiên n bài hát từ dataset."""
     try:
         random_songs = df.sample(n=n)[['spotify_id', 'name', 'artist', 'preview', 'img', 'music_genre_label']]
         random_songs = random_songs.fillna({'preview': '', 'img': ''})  # Thay NaN bằng chuỗi rỗng
         return random_songs.to_dict(orient='records')
     except ValueError:
         raise HTTPException(status_code=400, detail="Số lượng bài hát yêu cầu vượt quá kích thước dataset")


@app.get("/songs_by_genre")
async def get_songs_by_genre(
    genre: str = Query(..., description="Tên thể loại nhạc (ví dụ: Chill, Dance, ...)"),
    n: int = Query(20, description="Số lượng bài hát muốn lấy")
):
    """Lấy bài hát theo thể loại đã cluster."""
    filtered = df[df['music_genre_label'].str.lower() == genre.lower()]
    
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No songs found for genre '{genre}'")
    
    sampled = filtered.sample(n=min(n, len(filtered)))
    sampled = sampled[['spotify_id', 'name', 'artist', 'preview', 'img', 'music_genre_label']]
    
    return sampled.fillna('').to_dict(orient='records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)