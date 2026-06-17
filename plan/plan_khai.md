# Plan — Khải (LLM Core & Agent Prompts)

Phụ trách load model, quản lý VRAM, và viết 3 Agent: Router, Fast-QA, Reading.

---

## 1. File cần tạo

| File | Chức năng |
|---|---|
| `src/core/llm_engine.py` | Load model Unsloth, hàm batch inference, dọn VRAM |
| `src/agents/router.py` | Phân loại câu hỏi → FAST_QA / READING / CODEABLE |
| `src/agents/fast_qa.py` | Trả lời zero-shot câu ngắn |
| `src/agents/reading.py` | Chain-of-Thought đọc hiểu văn bản dài |

---

## 2. `llm_engine.py` — Load Model & VRAM

*(Nguồn: [report_pipeline.md L272-276](file:///d:/Competition/Hackathon/report/report_pipeline.md#L272))*

- Load Gemma-4 hoặc Qwen3.5 (<=9B) qua Unsloth 4-bit.
- Viết hàm `batch_inference()` nhận list prompts, trả list kết quả.
- **Bắt buộc** gọi dọn rác VRAM sau mỗi batch.

```python
from unsloth import FastLanguageModel
import torch, gc

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-E4B-it",
    max_seq_length=2048, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def cleanup_vram():
    gc.collect()
    torch.cuda.empty_cache()

def batch_inference(prompts, max_new_tokens=256, temperature=0.2):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  temperature=temperature, do_sample=(temperature > 0))
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    del inputs, outputs
    cleanup_vram()
    return results
```

---

## 3. `router.py` — Phân loại câu hỏi

*(Nguồn: [report_pipeline.md L90-104](file:///d:/Competition/Hackathon/report/report_pipeline.md#L90))*

Nhận batch câu hỏi, gán nhãn `FAST_QA`, `READING`, hoặc `CODEABLE`. Ghi vào `GraphState["routes"]`.

**System Prompt:**
```
Bạn là bộ định tuyến dữ liệu siêu tốc của hệ thống LangGraph. Phân loại câu hỏi đầu vào vào một nhóm duy nhất.
Trả về JSON: {"route": "FAST_QA" | "READING" | "CODEABLE"}

Quy tắc:
1. 'READING': Đọc hiểu văn bản dài, đoạn trích lịch sử/pháp luật phức tạp.
2. 'CODEABLE': Giải toán, tính công thức, số liệu lý/hóa cụ thể.
3. 'FAST_QA': Định nghĩa ngắn, tri thức nền, sự thật đơn giản dưới 2000 ký tự.
Chỉ trả JSON sạch. Không viết thêm lời dẫn.
```
Cấu hình: `temperature=0.0`, `max_new_tokens=32`.

---

## 4. `fast_qa.py` — Zero-shot siêu tốc

*(Nguồn: [report_pipeline.md L106-136](file:///d:/Competition/Hackathon/report/report_pipeline.md#L106))*

Câu kiến thức nền. Cần nhanh nhất có thể (ảnh hưởng 10đ Inference Time).

**System Prompt:**
```
Bạn là chuyên gia trắc nghiệm tri thức. Chọn đáp án đúng duy nhất.
Trả về JSON: {"reasoning": "Tối đa 1 câu", "answer": "A/B/C/D"}
Phần 'reasoning' chỉ 1 câu. Không giải thích dài. Không viết ngoài JSON.
```

**User Prompt (có `/no_think` tắt CoT, tăng tốc):**
```
/no_think
Ví dụ: Câu hỏi: Thủ đô Việt Nam? → {"reasoning": "Hà Nội là thủ đô.", "answer": "B"}

Câu hỏi: {question}
Lựa chọn: {choices}
```

---

## 5. `reading.py` — Đọc hiểu văn bản dài

*(Nguồn: [report_pipeline.md L138-168](file:///d:/Competition/Hackathon/report/report_pipeline.md#L138))*

Rà soát văn bản dài, dò bẫy từ phủ định ("không phải", "ngoại trừ"). Dùng `/think` bật CoT.

**System Prompt:**
```
Bạn là chuyên gia rà soát bẫy văn bản dài. Đối chiếu chi tiết, rà soát từ phủ định (không phải, ngoại trừ, sai).
Trả về JSON: {"reasoning": "Tối đa 2 câu ngắn", "answer": "A/B/C/D"}
Không chép lại đoạn văn bản dài vào JSON.
```

**User Prompt (có `/think` bật CoT):**
```
/think
Câu hỏi: {question}
Lựa chọn: {choices}
```

---

## 6. Lưu ý

- Parse JSON thất bại → gán mặc định "A", để Dynamic Mapper của Cường xử lý sau.
- `temperature=0.0` cho Router. `0.2` cho Fast-QA/Reading.
- Hàm `batch_inference()` sẽ được Huy gọi lại cho Coder + Voting, viết cho dễ tái sử dụng.
