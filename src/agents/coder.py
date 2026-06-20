"""
Coder Agent — Ép LLM sinh mã Python, chạy Sandbox, tự sửa lỗi (Correction) và Fallback.
"""

import re
import logging
from src.agents.state import GraphState
from src.core.config import settings
from src.core.llm_engine import batch_inference
from src.tools.python_sandbox import execute_code, find_closest_choice_info
from src.pipeline.checkpointing import save_checkpoint

logger = logging.getLogger("HackAIthon_Agent")

# ===========================================================================
# PROMPTS CẤU HÌNH THEO ĐÚNG PLAN_HUY.MD
# ===========================================================================

CODER_SYSTEM_PROMPT = (
    "Bạn là chuyên gia lập trình Python tối giản, xử lý bài toán định lượng đa ngành.\n"
    "Viết mã Python hoàn chỉnh để giải bài toán.\n\n"
    "QUY TẮC:\n"
    "1. CẤM dùng dấu thăng (#). Không comment.\n"
    "2. CẤM in ra chữ cái nhãn đáp án (A, B, C, D).\n"
    "3. Lệnh print() cuối cùng:\n"
    "   - Đáp án là số → print(float(gia_tri)). Không thêm đơn vị.\n"
    "   - Đáp án là biểu thức SymPy → print(str(ket_qua)).\n"
    "4. Bọc toàn bộ trong try-except. Nhánh except: print('ERROR').\n"
    "5. Chỉ xuất code trong thẻ ```python và ```. Không viết lời dẫn."
)

CODER_USER_TEMPLATE = (
    "/no_think\n"
    "Ví dụ 1 (đáp án là số):\n"
    "Câu hỏi: Tính khối lượng NaOH cần dùng để trung hòa 200ml HCl 1M.\n"
    "Lựa chọn: A. 40g  B. 80g  C. 160g\n"
    "Code mẫu:\n"
    "```python\n"
    "try:\n"
    "    n_hcl = 0.2 * 1.0\n"
    "    m_naoh = n_hcl * 40.0\n"
    "    print(float(m_naoh))\n"
    "except:\n"
    "    print('ERROR')\n"
    "```\n\n"
    "Ví dụ 2 (đáp án là biểu thức):\n"
    "Câu hỏi: Tìm nghiệm B'(t) = -k*B(t), B(0)=B0.\n"
    "Lựa chọn: A. B0 * exp(-k * t)  B. B0 / (1 + k * t)\n"
    "Code mẫu:\n"
    "```python\n"
    "try:\n"
    "    import sympy\n"
    "    t, B0, k = sympy.symbols('t B0 k')\n"
    "    ket_qua = B0 * sympy.exp(-k * t)\n"
    "    print(str(ket_qua))\n"
    "except:\n"
    "    print('ERROR')\n"
    "```\n\n"
    "BÂY GIỜ GIẢI:\n"
    "Câu hỏi: {question}\n"
    "Lựa chọn: {choices}\n"
)

CORRECTION_SYSTEM_PROMPT = (
    "Bạn là chuyên gia sửa lỗi Python. Viết lại script sửa logic/cú pháp.\n"
    "Hãy viết lại một script Python hoàn chỉnh mới đáp ứng các yêu cầu sau:\n"
    "1. CẤM dùng dấu thăng (#). Không comment.\n"
    "2. CẤM in ra chữ cái nhãn đáp án (A, B, C, D).\n"
    "3. Bọc toàn bộ trong try-except. Nhánh except: print('ERROR').\n"
    "4. Chỉ xuất code trong thẻ ```python và ```."
)

FALLBACK_SYSTEM_PROMPT = (
    "Bạn là chuyên gia giải đề trắc nghiệm có tư duy phản biện cao.\n"
    "Thuật toán giải toán đã thất bại. Dựa vào log lỗi, hãy phân tích văn bản để chọn đáp án đúng nhất.\n"
    "Bắt buộc trả về định dạng JSON sạch:\n"
    '{"reasoning": "Phân tích và suy luận ngắn gọn", "answer": "Chữ cái viết hoa đáp án đúng nhất"}\n'
    "Không viết thêm lời dẫn nào khác ngoài JSON."
)


def _extract_code(raw_output: str) -> str:
    """Trích xuất mã nguồn Python từ khối markdown ```python."""
    match = re.search(r'```python\s*(.*?)\s*```', raw_output, re.DOTALL)
    return match.group(1) if match else raw_output


def format_choices(choices: list) -> str:
    """Định dạng list choices thành chuỗi nhãn A, B, C, D."""
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    return "  ".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


def coder_node(state: GraphState, model, tokenizer) -> GraphState:
    """Node Coder Agent: Sinh mã, thực thi, sửa lỗi và fallback cho luồng CODEABLE."""
    indices = [i for i, r in enumerate(state["routes"]) if r == "CODEABLE"]
    if not indices:
        return state

    max_retries = settings.sandbox.max_retries  # = 2
    timeout = settings.sandbox.timeout_sec      # = 5
    qa_cfg = settings.agents.qa

    logger.info(f"[Huy-Coder] Bắt đầu xử lý {len(indices)} câu hỏi CODEABLE...")

    for idx in indices:
        # Bỏ qua nếu câu đã có đáp án chốt cuối
        if state["final_answers"][idx] != "":
            continue

        q = state["questions"][idx]
        choices_str = format_choices(q["choices"])

        logger.info(f"   [Coder] Đang xử lý câu Index {idx}...")

        # -------------------------------------------------------------------
        # BƯỚC 1: SINH MÃ PYTHON BAN ĐẦU
        # -------------------------------------------------------------------
        user_content = CODER_USER_TEMPLATE.format(question=q["question"], choices=choices_str)
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ], tokenize=False, add_generation_prompt=True)

        raw_out = batch_inference([prompt], max_new_tokens=qa_cfg.max_new_tokens, temperature=qa_cfg.temperature)[0]
        code = _extract_code(raw_out)
        state["execution_codes"][idx] = code

        # -------------------------------------------------------------------
        # BƯỚC 2: VÒNG LẶP CHẠY SANDBOX & TỰ SỬA LỖI (MAX RETRIES = 2)
        # -------------------------------------------------------------------
        success = False
        run_res = {}
        closest_letter = None

        for attempt in range(max_retries + 1):
            logger.info(f"      [Sandbox] Thử nghiệm chạy mã (Lần {attempt + 1})...")
            run_res = execute_code(code, timeout=timeout)
            state["sandbox_results"][idx] = run_res

            if run_res["success"]:
                # So khớp số học/chuỗi với các đáp án lựa chọn
                closest_letter, min_diff, _ = find_closest_choice_info(run_res["stdout"], q["choices"])
                # Chấp nhận nếu sai số tương đối <= 0.01
                if closest_letter and min_diff <= 0.01:
                    success = True
                    state["final_answers"][idx] = f"SUCCESS_MATCH:{closest_letter}:{run_res['stdout']}"
                    logger.info(f"      [Sandbox] Thành công khớp đáp án {closest_letter} (stdout: {run_res['stdout']})")
                    break
                else:
                    err_msg = f"Stdout '{run_res['stdout']}' không khớp phương án nào có sai số <= 1%."
            else:
                err_msg = run_res["stderr"]

            # Nếu chưa thành công và còn lượt sửa lỗi
            if attempt < max_retries:
                logger.warning(f"      [Correction] Lỗi thực thi hoặc lệch kết quả: {err_msg}")
                corr_user_content = (
                    f"Mã nguồn trước đó bị lỗi hoặc cho ra kết quả lệch.\n"
                    f"Câu hỏi: {q['question']}\n"
                    f"Lựa chọn: {choices_str}\n"
                    f"Mã nguồn cũ:\n```python\n{code}\n```\n"
                    f"Thông báo lỗi:\n{err_msg}\n\n"
                    f"Hãy sửa đổi logic/cú pháp và viết lại mã nguồn mới hoàn chỉnh."
                )
                corr_prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": corr_user_content}
                ], tokenize=False, add_generation_prompt=True)

                raw_corr = batch_inference([corr_prompt], max_new_tokens=qa_cfg.max_new_tokens, temperature=qa_cfg.temperature)[0]
                code = _extract_code(raw_corr)
                state["execution_codes"][idx] = code
            else:
                logger.error(f"      [Sandbox] Đã thử {max_retries + 1} lần đều thất bại.")

        # -------------------------------------------------------------------
        # BƯỚC 3: FALLBACK AGENT (NẾU HẾT LƯỢT VẪN SAI/LỖI)
        # -------------------------------------------------------------------
        if not success:
            logger.info("      [Fallback] Kích hoạt Fallback Agent dùng text reasoning...")
            fb_user_content = (
                f"Câu hỏi: {q['question']}\n"
                f"Lựa chọn: {choices_str}\n\n"
                f"Nhật ký lỗi từ Sandbox:\n"
                f"Code đã chạy:\n{code}\n"
                f"Stdout: {run_res.get('stdout', '')}\n"
                f"Stderr: {run_res.get('stderr', '')}\n"
            )
            fb_prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": fb_user_content}
            ], tokenize=False, add_generation_prompt=True)

            raw_fb = batch_inference([fb_prompt], max_new_tokens=qa_cfg.max_new_tokens, temperature=qa_cfg.temperature)[0]
            # Lưu tạm kết quả Fallback vào final_answers (sẽ được dùng làm cơ sở Voting sau này)
            state["final_answers"][idx] = raw_fb

        save_checkpoint(state)

    return state
