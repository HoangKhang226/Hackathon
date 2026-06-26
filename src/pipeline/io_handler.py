import json
import os

import pandas as pd

def load_test_data(data_dir=None):
    if data_dir is None:
        data_dir = "/data" if os.path.exists("/data") else "./data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Thư mục {data_dir} không tồn tại.")
        
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith(".csv"):
            return pd.read_csv(path)
        elif file.endswith(".json"):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                for key in ["questions", "data", "items"]:
                    if key in data:
                        return pd.DataFrame(data[key])
                return pd.DataFrame([data])
                
    raise FileNotFoundError(f"Không tìm thấy file .csv hay .json nào trong {data_dir}")



def load_test_data_json(data_dir="/data"):
    for name in ["private_test.json", "public_test.json"]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            return pd.DataFrame(payload)

        if isinstance(payload, dict):
            for key in ("questions", "data", "items", "records"):
                if key in payload and isinstance(payload[key], list):
                    return pd.DataFrame(payload[key])
            return pd.DataFrame([payload])

        raise ValueError(f"Định dạng JSON không hỗ trợ: {path}")




def save_predictions(results: list, output_dir=None):
    if output_dir is None:
        output_dir = "/output" if os.path.exists("/data") else "./output"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=["qid", "answer"])
    df.to_csv(os.path.join(output_dir, "pred.csv"), index=False)