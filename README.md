# NotSpotify Backend

## Giới thiệu
Backend cho hệ thống gợi ý nhạc tương tự như Spotify, bao gồm tiền xử lý dữ liệu, huấn luyện mô hình và API triểu hồi gợi ý.

## Câu trúc project

```
NOTSPOTIFY_BACKEND/
|— app/
|   — app.py                # FastAPI app
|— data/
|   |— raw/                  # File CSV gốc
|   |— processed/            # File CSV sau khi preprocess
|— models/                  # Lưu scaler, label encoder, cosine similarity
|— notebooks/               # Notebook phát triển nhanh (nếu có)
|— src/
|   |— data_loader.py        # Load config và dữ liệu
|   |— preprocessing.py     # Tiền xử lý dữ liệu
|   |— train.py             # Huấn luyện model
|   |— recommender.py       # Triểu hồi gợi ý
|   |— evaluate.py         # Đánh giá mô hình (nếu có)
|— .env                     # Biến môi trường (nếu có)
|— config.yaml              # Cấu hình file
|— requirements.txt         # Thư viện cần thiết
|— test.py                  # Test nhanh (nếu có)
```

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate.bat   # Windows
source venv/bin/activate #Ubuntu

pip install -r requirements.txt
```

## Quy trình chạy project

### Bước 1: Tiền xử lý dữ liệu

```bash
python -m src.preprocessing
```

Kết quả:
- Tạo file CSV đã xử lý trong `data/processed/`
- Tạo scaler và label encoder trong `models/`

### Bước 2: Train mô hình gợi ý

```bash
python -m src.train
```

Kết quả:
- Lưu mảng cosine similarity vào `models/`

### Bước 3: Chạy API FastAPI

```bash
uvicorn app.app:app --reload
```

Sau đó vào trình duyệt:
- http://localhost:8000/docs để test API

## Ghi chú
- Nhớ sửa đúng đường dẫn trong `config.yaml` theo tổ chức thư mục.
- Dataset chỉ giới hạn tối đa 1500 bài đã được xử lý trong code preprocessing.

## Tác giả
- Project by Phan Minh Gia Huy
