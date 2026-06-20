import json
import os

import pandas as pd

def load_test_data(data_dir="/data"):
    for name in ["private_test.csv", "public_test.csv"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Không tìm thấy file test trong /data")



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




def save_predictions(results: list, output_dir="/output"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=["qid", "answer"])
    df.to_csv(os.path.join(output_dir, "pred.csv"), index=False)