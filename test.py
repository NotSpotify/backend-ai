# import pandas as pd
# df = pd.read_csv('data/processed/Music_processed.csv')
# print(df['spotify_id'].head(5).tolist())

import joblib
import numpy as np
cosine_sim = joblib.load('models/cosine_sim.pkl')
print(np.isnan(cosine_sim).sum())  # Số NaN
print(np.isinf(cosine_sim).sum())  # Số inf