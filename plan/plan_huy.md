# Plan — Huy (Coder Agent, Sandbox & Voting)

Phụ trách sinh code Python giải toán, chạy code trong Sandbox, tự sửa lỗi, và bầu chọn đáp án.

---

## 1. File cần tạo

| File | Chức năng |
|---|---|
| `src/agents/coder.py` | Ép LLM viết code Python giải toán |
| `src/tools/python_sandbox.py` | Chạy code bằng subprocess, có timeout |
| `src/pipeline/majority_voting.py` | Gộp phiếu Sandbox + LLM text để chốt đáp án |

---

## 2. `coder.py` — Sinh code Python giải toán

*(Nguồn: [report_pipeline.md L170-225](file:///d:/Competition/Hackathon/report/report_pipeline.md#L170), [current_pipeline_architecture.md L112-126](file:///d:/Competition/Hackathon/report/current_pipeline_architecture.md#L112))*

LLM tính toán sai (ảo giác số học). Không cho LLM tự đưa đáp án. Ép nó viết code Python.

**System Prompt (copy nguyên):**
```
Bạn là chuyên gia lập trình Python tối giản, xử lý bài toán định lượng đa ngành.
Viết mã Python hoàn chỉnh để giải bài toán.

QUY TẮC:
1. CẤM dùng dấu thăng (#). Không comment.
2. CẤM in ra chữ cái nhãn đáp án (A, B, C, D).
3. Lệnh print() cuối cùng:
   - Đáp án là số → print(float(gia_tri)). Không thêm đơn vị.
   - Đáp án là biểu thức SymPy → print(str(ket_qua)).
4. Bọc toàn bộ trong try-except. Nhánh except: print('ERROR').
5. Chỉ xuất code trong thẻ ```python```. Không viết lời dẫn.
```

**User Prompt (có few-shot):**
```
/no_think
Ví dụ 1 (đáp án là số):
Câu hỏi: Tính khối lượng NaOH cần dùng để trung hòa 200ml HCl 1M.
Lựa chọn: A. 40g  B. 80g  C. 160g
Code mẫu:
try:
    n_hcl = 0.2 * 1.0
    m_naoh = n_hcl * 40.0
    print(float(m_naoh))
except:
    print('ERROR')

Ví dụ 2 (đáp án là biểu thức):
Câu hỏi: Tìm nghiệm B'(t) = -k*B(t), B(0)=B0.
Code mẫu:
try:
    import sympy
    t, B0, k = sympy.symbols('t B0 k')
    ket_qua = B0 * sympy.exp(-k * t)
    print(str(ket_qua))
except:
    print('ERROR')

BÂY GIỜ GIẢI:
Câu hỏi: {question}
Lựa chọn: {choices}
```

---

## 3. `python_sandbox.py` — Chạy code cách ly

*(Nguồn: [report_pipeline.md L227-236](file:///d:/Competition/Hackathon/report/report_pipeline.md#L227))*

Code do LLM sinh ra → lưu file tạm → chạy qua subprocess → bắt stdout/stderr.

```python
import subprocess, sys, os

def execute_code(code_string: str, timeout: int = 5) -> dict:
    temp_file = "temp_sandbox.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code_string)
    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True, text=True, timeout=timeout
        )
        return {"success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Timeout > 5s"}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
```

**Lưu ý Docker:** Container phải cho phép subprocess. Cường sẽ lo phần này.

---

## 4. Self-Correction & Fallback (nằm trong coder.py)

*(Nguồn: [report_pipeline.md L227-243](file:///d:/Competition/Hackathon/report/report_pipeline.md#L227), [current_pipeline_architecture.md L124-126](file:///d:/Competition/Hackathon/report/current_pipeline_architecture.md#L124))*

Nếu Sandbox trả `success=False` hoặc stdout không khớp phương án nào (sai số > 1%):

1. **Correction (tối đa 2 lần):** Đưa `stderr` + code cũ vào LLM, bảo sửa.
   - Prompt: *"Bạn là chuyên gia sửa lỗi Python. Viết lại script sửa logic/cú pháp. Đây là code cũ: {code}. Lỗi: {stderr}."*
2. **Fallback (sau 2 lần fail):** Bỏ code, dùng LLM text thuần.
   - Prompt: *"Thuật toán giải toán đã thất bại. Dựa vào log lỗi, hãy phân tích văn bản để chọn đáp án."*

**So khớp số học:** Dùng Regex `r'-?\b\d+\.?\d*'` hút số từ stdout, so với các lựa chọn bằng sai số tương đối `abs(target - val) / max(abs(val), 1.0) <= 0.01`. *(Nguồn: [report_pipeline.md L259-263](file:///d:/Competition/Hackathon/report/report_pipeline.md#L259))*

---

## 5. `majority_voting.py` — Bầu chọn đáp án

*(Nguồn: [report_pipeline.md L245-253](file:///d:/Competition/Hackathon/report/report_pipeline.md#L245), [current_pipeline_architecture.md L128-133](file:///d:/Competition/Hackathon/report/current_pipeline_architecture.md#L128))*

Sandbox ra kết quả đúng chưa phải chốt cuối. Cần bầu thêm.

**Cơ chế:**
- Sandbox khớp phương án A → A được **+2 phiếu**.
- Chạy LLM text 3 vòng (`temperature=0.3`). Mỗi vòng đưa Hint (stdout Sandbox) vào prompt. Mỗi vòng **+1 phiếu**.
- Đáp án nhiều phiếu nhất thắng.

**Voting Prompt:**
```
Bạn là chuyên gia toán học. Dựa trên câu hỏi, phương án và kết quả gợi ý định lượng, chọn đáp án đúng nhất.
Trả về JSON: {"reasoning": "...", "answer": "A/B/C/D"}

[Kết quả Python gợi ý]: {stdout_val}
[Khớp phương án]: {closest_letter}

Câu hỏi: {question}
Lựa chọn: {choices}
```

---

## 6. Tóm tắt luồng xử lý của Huy

```
Câu hỏi CODEABLE
  → coder.py sinh code Python
  → python_sandbox.py chạy code (timeout 5s)
    → Thành công + khớp số → SUCCESS_MATCH
    → Lỗi/lệch → Correction Agent (tối đa 2 lần)
    → Vẫn fail → Fallback Agent (text thuần)
  → majority_voting.py bầu chọn (Sandbox 2 phiếu + LLM 3 phiếu)
  → Trả đáp án cho Cường (Dynamic Mapper)
```
