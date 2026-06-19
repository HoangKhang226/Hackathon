import pandas as pd
import os

def load_test_data(data_dir="/data"):
    for name in ["private_test.csv", "public_test.csv"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Không tìm thấy file test trong /data")

def save_predictions(results: list, output_dir="/output"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=["qid", "answer"])
    df.to_csv(os.path.join(output_dir, "pred.csv"), index=False)