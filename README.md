
# NotSpotify Backend

## 🚀 Giới thiệu
Backend cho hệ thống gợi ý nhạc tương tự Spotify, bao gồm pipeline xử lý dữ liệu, huấn luyện mô hình gợi ý, đánh giá mô hình và triển khai API bằng FastAPI.

## 🧱 Cấu trúc thư mục

```
NOTSPOTIFY_BACKEND/
│
├── app/
│   └── app.py                     # FastAPI app (triển khai API)
│
├── compare_models/               # So sánh các thuật toán gợi ý
│   ├── compare.py                # Đánh giá Precision@K giữa nhiều mô hình
│   ├── recommend_by_id_cluster.py
│   ├── recommend_by_id_euclidean.py
│   ├── recommend_by_id_random.py
│
├── data/
│   ├── raw/                      # Dữ liệu gốc (CSV từ Spotify)
│   └── processed/                # Dữ liệu đã tiền xử lý
│
├── models/                       # Lưu các artifact như scaler, encoder, cosine_sim,...
│
├── src/
│   ├── cluster_music.py          # Gán nhãn thể loại bằng clustering
│   ├── data_loader.py            # Hàm load file cấu hình và dữ liệu
│   ├── evaluate.py               # Tính precision@K
│   ├── preprocessing.py          # Tiền xử lý dữ liệu
│   ├── recommender.py            # Gợi ý bằng cosine similarity
│   ├── train.py                  # Huấn luyện mô hình (tính cosine_sim)
│
├── config.yaml                   # Cấu hình chính (đường dẫn, feature, mô hình)
├── requirements.txt              # Thư viện cần cài đặt
└── README.md
```

## 🛠️ Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
```

## 🔁 Quy trình chạy project

### Bước 1: Tiền xử lý dữ liệu

```bash
python -m src.preprocessing
```

Kết quả:
- File CSV đã xử lý tại: `data/processed/Music_processed.csv`
- Các mô hình: `scaler.pkl`, `label_encoder.pkl` trong thư mục `models/`

### Bước 2: Huấn luyện mô hình gợi ý (tính cosine similarity)

```bash
python -m src.train
```

Kết quả:
- Tạo file `cosine_sim.pkl` trong thư mục `models/`

### Bước 3: Chạy đánh giá mô hình

```bash
python -m src.evaluate
```

Kết quả:
- Precision@5 cho mô hình hiện tại.

### Bước 4: So sánh nhiều mô hình (Cosine, Euclidean, Cluster, Random)

```bash
python -m compare_models.compare
```

Kết quả ví dụ:

```
📊 KẾT QUẢ SO SÁNH PRECISION@5:
Mô hình              Precision@5
-------------------------------
Cosine Similarity          0.85
Euclidean Distance         0.81
Cluster-Based              0.76
Random Recommend           0.22
```

### Bước 5: Chạy API FastAPI

```bash
uvicorn app.app:app --reload
```

Truy cập Swagger docs tại: [http://localhost:8000/docs](http://localhost:8000/docs)

## ✨ Các API nổi bật

- `GET /recommend?sp_id=...`: Gợi ý bài hát tương tự
- `GET /random_songs`: Lấy n bài hát ngẫu nhiên
- `GET /songs_by_genre?genre=Chill`: Lấy bài theo thể loại
- `POST /generate_playlists_by_genres`: Tạo playlist từ nhiều thể loại

## 📌 Ghi chú

- Dữ liệu tối đa 1500 bài (đã được chọn lọc trước)
- Nếu lỗi module, đảm bảo thư mục `compare_models` đã có `__init__.py` và `PYTHONPATH=.` khi chạy

## 👨‍💻 Tác giả
Phan Minh Gia Huy - Đồ án hệ thống gợi ý âm nhạc sử dụng nội dung
