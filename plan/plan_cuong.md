# Plan — Cường (Pipeline, Mapper & Docker)

Phụ trách ghép nối hệ thống, lọc đáp án, lưu checkpoint và đóng gói Docker nộp bài.

---

## 1. File cần tạo

| File | Chức năng |
|---|---|
| `src/agents/state.py` | Định nghĩa GraphState |
| `src/pipeline/io_handler.py` | Đọc CSV đầu vào, ghi pred.csv đầu ra |
| `src/pipeline/checkpointing.py` | Lưu/nạp tiến trình chống sập |
| `src/pipeline/dynamic_mapper.py` | Lọc rác LLM → chốt A/B/C/D |
| `src/main.py` | Entry point, nối tất cả module lại |
| `Dockerfile` | Đóng gói container |
| `requirements.txt` (root) | Danh sách thư viện |

---

## 2. `state.py` — GraphState

*(Nguồn: [report_pipeline.md L71-82](file:///d:/Competition/Hackathon/report/report_pipeline.md#L71))*

```python
from typing import List, TypedDict

class GraphState(TypedDict):
    questions: List[dict]        # {qid, question, choices}
    choice_counts: List[int]     # số phương án mỗi câu (4, 5,...)
    routes: List[str]            # FAST_QA, READING, CODEABLE
    execution_codes: List[str]   # code Python sinh ra (luồng CODEABLE)
    sandbox_results: List[dict]  # {success, stdout, stderr}
    final_answers: List[str]     # đáp án chốt cuối: A/B/C/D
```

---

## 3. `io_handler.py` — Đọc/Ghi file

*(Nguồn: [rule.txt](file:///d:/Competition/Hackathon/rule.txt))*

- **Input:** Đọc `/data/public_test.csv` hoặc `/data/private_test.csv`. Dùng Pandas.
- **Output:** Ghi `/output/pred.csv` với đúng 2 cột: `qid,answer`.

```python
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
```

---

## 4. `checkpointing.py` — Lưu tiến trình chống sập

*(Nguồn: [report_pipeline.md L81-82](file:///d:/Competition/Hackathon/report/report_pipeline.md#L81), [current_pipeline_architecture.md L143-146](file:///d:/Competition/Hackathon/report/current_pipeline_architecture.md#L143))*

Khi chạy 2000 câu, OOM hoặc lỗi GPU có thể xảy ra. Sau mỗi batch (10 câu), lưu state ra JSON. Khi restart, nạp lại và chỉ chạy tiếp câu chưa có `final_answers`.

```python
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
```

---

## 5. `dynamic_mapper.py` — Lọc rác LLM

*(Nguồn: [report_pipeline.md L259-270](file:///d:/Competition/Hackathon/report/report_pipeline.md#L259), [current_pipeline_architecture.md L135-139](file:///d:/Competition/Hackathon/report/current_pipeline_architecture.md#L135))*

BTC chỉ chấp nhận `A`, `B`, `C` hoặc `D`. LLM hay in thừa (VD: "Đáp án đúng là A").

**Bước 1:** Regex xóa ký tự đặc biệt.
```python
import re
def clean_text(text: str) -> str:
    return re.sub(r'[\$\{\}\\\^\_\(\)\*\/\+\-\=\s\[\]\,\.]', '', text.lower().strip())
```

**Bước 2:** Intersection Matching — so khớp ký tự giữa output LLM và từng choice gốc. Lấy choice trùng nhiều nhất.
```python
def map_answer(llm_output: str, choices: list, num_choices: int) -> str:
    clean_out = clean_text(llm_output)
    best_idx, best_score = 0, -1
    for i, choice in enumerate(choices):
        score = len(set(clean_text(choice)) & set(clean_out))
        if score > best_score:
            best_score = score
            best_idx = i
    # Kiểm tra không vượt quá số đáp án thực tế
    best_idx = min(best_idx, num_choices - 1)
    return chr(ord('A') + best_idx)
```

**Bước 3:** So khớp số (cho luồng Sandbox). Dùng sai số tương đối <= 1%.
```python
def match_number(stdout_val: str, choices: list) -> str | None:
    nums = re.findall(r'-?\b\d+\.?\d*', stdout_val)
    if not nums:
        return None
    val = float(nums[-1])
    for i, choice in enumerate(choices):
        choice_nums = re.findall(r'-?\b\d+\.?\d*', choice)
        for cn in choice_nums:
            target = float(cn)
            if abs(target - val) / max(abs(val), 1.0) <= 0.01:
                return chr(ord('A') + i)
    return None
```

---

## 6. `main.py` — Entry point

Nối module của Khải và Huy lại:

```python
# Pseudocode
def main():
    # 1. Load model (Khải)
    engine = load_model()
    
    # 2. Load data
    df = load_test_data()
    state = load_checkpoint() or init_state(df)
    
    # 3. Xử lý theo batch
    for batch in get_pending_batches(state, batch_size=10):
        # Router (Khải)
        routes = router_agent(engine, batch)
        
        # Phân nhánh
        for item, route in zip(batch, routes):
            if route == "FAST_QA":
                answer = fast_qa_agent(engine, item)
            elif route == "READING":
                answer = reading_agent(engine, item)
            elif route == "CODEABLE":
                answer = coder_sandbox_loop(engine, item)  # Huy
            
            # Lọc rác
            final = dynamic_mapper(answer, item["choices"], item["num_choices"])
            state["final_answers"][item["idx"]] = final
        
        # Lưu checkpoint
        save_checkpoint(state)
    
    # 4. Ghi kết quả
    save_predictions(state)
```

---

## 7. Dockerfile

*(Nguồn: [rule.txt](file:///d:/Competition/Hackathon/rule.txt))*

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
CMD ["python", "src/main.py"]
```

Mount khi chạy: `docker run -v ./data:/data -v ./output:/output image_name`

**requirements.txt:**
```
unsloth
pandas
sympy
pydantic
tqdm
```

---

## 8. Tóm tắt luồng của Cường

```
main.py khởi động
  → load model (gọi module Khải)
  → load data từ /data
  → load checkpoint (nếu có)
  → vòng lặp batch:
      → router (Khải) phân loại
      → fast_qa/reading (Khải) hoặc coder+sandbox (Huy)
      → dynamic_mapper lọc rác → chốt A/B/C/D
      → save checkpoint
  → ghi /output/pred.csv
```
