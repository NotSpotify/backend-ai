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

@app.get("/random_songs")
async def get_random_songs(n: int = 20):
    """Lấy ngẫu nhiên n bài hát từ dataset."""
    try:
        random_songs = df.sample(n=n)[['spotify_id', 'name', 'artist', 'preview', 'img']]
        # Xử lý NaN/inf trong các cột
        random_songs = random_songs.fillna({'preview': '', 'img': ''})  # Thay NaN bằng chuỗi rỗng
        return random_songs.to_dict(orient='records')
    except ValueError:
        raise HTTPException(status_code=400, detail="Số lượng bài hát yêu cầu vượt quá kích thước dataset")

@app.get("/recommend")
async def recommend(sp_id: str, n: int = Query()):
    recommendations = recommend_by_id(sp_id, df, cosine_sim, n)
    if recommendations is None:
        raise HTTPException(status_code=404, detail="Bài hát không tồn tại")
    
    # Làm sạch toàn bộ dataframe để đảm bảo JSON hợp lệ
    recommendations = recommendations.replace([np.inf, -np.inf], 0).fillna(0)

    # Optional: lọc kỹ hơn nếu cần
    for col in recommendations.columns:
        recommendations[col] = recommendations[col].apply(
            lambda x: 0 if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x
        )
    
    return recommendations.to_dict(orient='records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)