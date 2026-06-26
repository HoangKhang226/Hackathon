import json, os

def get_checkpoint_path():
    output_dir = "/output" if os.path.exists("/data") else "./output"
    return os.path.join(output_dir, "pipeline_checkpoint.json")

def save_checkpoint(state: dict):
    path = get_checkpoint_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

def load_checkpoint() -> dict | None:
    path = get_checkpoint_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None