import pandas as pd
import sqlite3
import yaml


def load_config(config_path='config.yaml'):
    """Đọc file cấu hình."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_csv(file_path):
    """Đọc dữ liệu từ CSV."""
    try:
        df = pd.read_csv(file_path)
        print(f"Đã đọc {len(df)} hàng từ {file_path}")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc CSV: {e}")
        return None