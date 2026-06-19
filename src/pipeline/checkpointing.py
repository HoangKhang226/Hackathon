import json, os

CHECKPOINT_PATH = "/output/pipeline_checkpoint.json"

def save_checkpoint(state: dict):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

def load_checkpoint() -> dict | None:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None