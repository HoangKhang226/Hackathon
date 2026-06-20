"""IO Handler — Đọc dữ liệu đầu vào, ghi kết quả pred.csv."""

import json
import os
from typing import List

import pandas as pd

from src.core.config import settings

import logging
logger = logging.getLogger("HackAIthon_Agent")


def load_test_data() -> List[dict]:
    """Đọc file test (ưu tiên private > public, CSV > JSON)."""
    data_dir = settings.paths.data_dir

    # Tìm file trong thư mục data
    candidates = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f.endswith(".csv"):
                candidates.append((full_path, "csv"))
            elif f.endswith(".json"):
                candidates.append((full_path, "json"))

    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu trong {data_dir}")

    # Ưu tiên: private > public, csv > json
    def priority(item):
        path, fmt = item
        name = os.path.basename(path).lower()
        score = 0
        if "private" in name:
            score += 10
        if fmt == "csv":
            score += 1
        return -score

    candidates.sort(key=priority)
    target_file, file_type = candidates[0]

    logger.info("[IO] Đang nạp dữ liệu từ: %s", target_file)

    if file_type == "json":
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # CSV format
    df = pd.read_csv(target_file)
    questions = []
    for _, row in df.iterrows():
        choices = []
        if "choices" in df.columns:
            choices = json.loads(row["choices"])
        else:
            for col in [chr(65 + i) for i in range(26)]:
                if col in df.columns and pd.notna(row.get(col)):
                    choices.append(str(row[col]))

        questions.append({
            "qid": str(row["qid"]),
            "question": str(row["question"]),
            "choices": choices,
        })

    logger.info("[IO] Đã nạp %d câu hỏi.", len(questions))
    return questions


def save_predictions(state: dict, output_dir: str = None):
    """Ghi kết quả cuối cùng ra pred.csv."""
    if output_dir is None:
        output_dir = settings.paths.output_dir

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pred.csv")

    rows = []
    for i, q in enumerate(state["questions"]):
        rows.append({"qid": q["qid"], "answer": state["final_answers"][i]})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("[IO] Đã ghi %d kết quả vào %s", len(rows), output_path)
