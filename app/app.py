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
async def recommend(sp_id: str = Query(default="0b4KsOdnrZlfE5VUAAzxv1"), n: int = Query(default=10)):
    """Recommend n bài hát tương tự sp_id."""
    recommendations = recommend_by_id(sp_id, df, cosine_sim, n)
    if recommendations is None:
        raise HTTPException(status_code=404, detail="Bài hát không tồn tại")
    
    recommendations = recommendations.replace([np.inf, -np.inf], 0).fillna(0)
    for col in recommendations.columns:
        recommendations[col] = recommendations[col].apply(
            lambda x: 0 if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x
        )
    
    return recommendations.to_dict(orient='records')

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