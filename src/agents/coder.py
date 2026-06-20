"""Coder Agent — Sinh code Python, chạy Sandbox, Self-Correction, Fallback."""

import re

from src.agents.state import GraphState, format_choices
from src.core.config import settings
from src.core.llm_engine import batch_inference
from src.tools.python_sandbox import execute_code, find_closest_choice_info
from src.pipeline.checkpointing import save_checkpoint

import logging
logger = logging.getLogger("HackAIthon_Agent")


# ===========================================================================
# PROMPTS
# ===========================================================================

CODER_SYSTEM_PROMPT = (
    "Bạn là một chuyên gia lập trình Python tối giản, phụ trách xử lý các bài toán "
    "định lượng và logic đa ngành.\n"
    "Nhiệm vụ: Hãy viết mã nguồn Python hoàn chỉnh để giải bài toán. "
    "Bạn được cung cấp danh sách phương án trắc nghiệm thực tế CHỈ ĐỂ tham khảo "
    "dạng kết quả số học hoặc kí tự đại số.\n\n"
    "QUY TẮC ÉP KHUÔN ĐỊNH DẠNG TUYỆT ĐỐI KHÔNG GÂY LỖI CÚ PHÁP:\n"
    "1. CẤM TUYỆT ĐỐI SỬ DỤNG KÝ TỰ DẤU THĂNG (#). Không viết comment, "
    "không tạo chuỗi hậu tố lặp lại dông dài.\n"
    "2. CẤM TUYỆT ĐỐI IN RA CÁC KÝ TỰ NHÃN ĐÁP ÁN NHƯ 'A', 'B', 'C', 'D'...\n"
    "3. QUY TẮC LỆNH PRINT() CUỐI CÙNG:\n"
    "   - Nếu các phương án lựa chọn chứa số → print(float(gia_tri)). "
    "Tuyệt đối không thêm chuỗi đơn vị.\n"
    "   - Nếu các phương án lựa chọn là biểu thức kí tự/đại số → "
    "print(str(ket_qua)) từ SymPy.\n"
    "4. Toàn bộ mã nguồn phải bọc gọn trong try-except toàn cục. "
    "Nhánh except chỉ viết: print('ERROR').\n"
    "5. Chỉ xuất duy nhất khối mã trong thẻ ```python và ```. Không viết lời dẫn."
)

CODER_FEW_SHOTS = (
    "Ví dụ 1 (Đề bài chứa số hoặc đơn vị kèm theo như %, g, kJ/mol, đô la):\n"
    "Câu hỏi: Tính khối lượng NaOH cần dùng để trung hòa 200ml dung dịch HCl 1M.\n"
    "Lựa chọn đáp án:\nA. 40g.\nB. 80g.\nC. 160g.\n"
    "Mã nguồn mẫu:\n"
    "```python\n"
    "try:\n"
    "    n_hcl = 0.2 * 1.0\n"
    "    m_naoh = n_hcl * 40.0\n"
    "    print(float(m_naoh))\n"
    "except:\n"
    "    print('ERROR')\n"
    "```\n\n"
    "Ví dụ 2 (Đề bài chứa công thức kí tự, phương trình đại số):\n"
    "Câu hỏi: Tìm nghiệm của phương trình vi phân B'(t) = -k*B(t) với B(0)=B0.\n"
    "Lựa chọn đáp án:\nA. B0 * exp(-k * t)\nB. B0 / (1 + k * t)\n"
    "Mã nguồn mẫu:\n"
    "```python\n"
    "try:\n"
    "    import sympy\n"
    "    t, B0, k = sympy.symbols('t B0 k')\n"
    "    ket_qua = B0 * sympy.exp(-k * t)\n"
    "    print(str(ket_qua))\n"
    "except:\n"
    "    print('ERROR')\n"
    "```\n\n"
)

CORRECTION_SYSTEM_PROMPT = (
    "Bạn là một chuyên gia sửa lỗi và tối ưu hóa mã nguồn Python đa ngành.\n"
    "Nhiệm vụ: Viết lại một script Python hoàn chỉnh mới, sửa đổi logic tính toán "
    "hoặc sửa triệt để lỗi gọi tên biến/cú pháp trong hàm print.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. CẤM TUYỆT ĐỐI SỬ DỤNG KÝ TỰ DẤU THĂNG (#).\n"
    "2. CẤM in ra các chữ cái nhãn đáp án A, B, C, D...\n"
    "3. Kiểm tra kĩ lưỡng tên biến bên trong print() có trùng khớp với biến đã khai báo không.\n"
    "4. Chỉ xuất duy nhất khối mã trong thẻ ```python và ```."
)

FALLBACK_SYSTEM_PROMPT = (
    "Bạn là chuyên gia giải đề trắc nghiệm có tư duy phản biện cao.\n"
    "Thuật toán viết mã giải toán trước đó của hệ thống đã bị tính toán lệch số "
    "hoặc lỗi runtime sau nhiều lần thử.\n"
    "Nhiệm vụ: Dựa vào câu hỏi, các phương án lựa chọn và chi tiết log lỗi, "
    "hãy phân tích lập luận văn bản để tìm ra đáp án đúng cuối cùng.\n"
    "Bắt buộc phải trả về JSON sạch:\n"
    '{"reasoning": "Phân tích logic và suy luận", '
    '"answer": "Chữ cái viết hoa đáp án chuẩn xác nhất"}\n'
    "Tuyệt đối không giải thích thêm ngoài khối JSON."
)


def _extract_code(raw_output: str) -> str:
    """Trích xuất code Python từ output LLM."""
    match = re.search(r'```python\s*(.*?)\s*```', raw_output, re.DOTALL)
    return match.group(1) if match else raw_output


def coder_sandbox_node(state: GraphState, model, tokenizer) -> GraphState:
    """Node 4 & 5: Sinh code → Sandbox → Self-Correction → Fallback → Voting."""
    indices = [i for i, r in enumerate(state["routes"]) if r == "CODEABLE"]
    if not indices:
        return state

    cfg_coder = settings.agents.coder
    cfg_correction = settings.agents.correction
    cfg_fallback = settings.agents.fallback
    sandbox_cfg = settings.sandbox
    total = len(indices)

    logger.info("\n[Node 4&5] Coder & Sandbox Loop: %d câu toán học", total)

    # ===========================================================================
    # CHU TRÌNH 1: SINH MÃ PYTHON
    # ===========================================================================
    for i in range(0, total, cfg_coder.batch_size):
        end_idx = min(i + cfg_coder.batch_size, total)
        batch_indices = indices[i:end_idx]

        # Skip câu đã sinh code
        if all(state["execution_codes"][idx] != "" for idx in batch_indices):
            continue

        logger.info("   [Coder] Sinh code: Câu %d đến %d / %d...", i + 1, end_idx, total)

        prompts = []
        for idx in batch_indices:
            q = state["questions"][idx]
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"/no_think\n{CODER_FEW_SHOTS}"
                    f"BÂY GIỜ HÃY GIẢI CÂU HỎI SAU, CHÚ Ý PRINT KẾT QUẢ SỐ THUẦN TUÝ "
                    f"HOẶC CHUỖI BIỂU THỨC SẠCH:\n"
                    f"Câu hỏi: {q['question']}\n"
                    f"Lựa chọn đáp án thực tế:\n{format_choices(q['choices'])}"
                )},
            ], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        raw_codes = batch_inference(
            model, tokenizer, prompts,
            max_new_tokens=cfg_coder.max_new_tokens,
            temperature=cfg_coder.temperature,
            micro_batch_size=cfg_coder.batch_size,
        )

        for j, raw_code in enumerate(raw_codes):
            state["execution_codes"][batch_indices[j]] = _extract_code(raw_code)

        save_checkpoint(state)

    # ===========================================================================
    # CHU TRÌNH 2: THỰC THI SANDBOX & SELF-CORRECTION
    # ===========================================================================
    max_retries = sandbox_cfg.max_retries  # = 2

    for attempt in range(max_retries):
        # Tìm câu chưa thành công
        active_indices = []
        for idx in indices:
            ans = state["final_answers"][idx]
            if (ans.startswith("SUCCESS_MATCH")
                    or (len(ans) == 1 and ans in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    or "reasoning" in ans):
                continue
            active_indices.append(idx)

        if not active_indices:
            break

        logger.info("   [Sandbox] Thực thi lượt %d/%d: %d câu...",
                    attempt + 1, max_retries, len(active_indices))

        correction_prompts_map = {}
        next_active = []

        for prog_idx, idx in enumerate(active_indices):
            code = state["execution_codes"][idx]
            logger.info("      [Sandbox Run] (%d/%d) Index %d...",
                        prog_idx + 1, len(active_indices), idx)

            # Chạy code
            run_res = execute_code(code, timeout=sandbox_cfg.timeout_sec)
            state["sandbox_results"][idx] = run_res

            # So khớp kết quả
            closest_letter, min_diff, target_val = None, float("inf"), None
            if run_res["success"]:
                closest_letter, min_diff, target_val = find_closest_choice_info(
                    run_res["stdout"], state["questions"][idx]["choices"]
                )

            # Thành công: sai số < 1%
            if run_res["success"] and closest_letter and min_diff < 0.01:
                state["final_answers"][idx] = (
                    f"SUCCESS_MATCH:{closest_letter}:{run_res['stdout']}"
                )
            else:
                # Tạo thông báo lỗi chi tiết
                if not run_res["success"]:
                    err_msg = f"Mã nguồn bị lỗi Runtime.\nLog: {run_res['stderr']}"
                elif target_val is None and run_res["stdout"] == "ERROR":
                    err_msg = "Mã nguồn crash nhảy vào except do lỗi cú pháp hoặc biến."
                elif target_val is None:
                    err_msg = (f"Mã chạy thành công nhưng stdout không khớp dải đáp án "
                               f"(stdout: '{run_res['stdout']}').")
                else:
                    err_msg = (f"Mã chạy ra '{target_val}' nhưng lệch quá xa "
                               f"(sai số nhỏ nhất: {min_diff * 100:.2f}% tại {closest_letter}).")

                # Còn lượt: tạo prompt sửa code
                if attempt < max_retries - 1:
                    cor_prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"/no_think\n"
                            f"Hãy sửa đổi và viết lại đoạn mã nguồn mới hoàn chỉnh.\n\n"
                            f"Câu hỏi: {state['questions'][idx]['question']}\n"
                            f"Lựa chọn:\n{format_choices(state['questions'][idx]['choices'])}\n\n"
                            f"Thông báo lỗi:\n{err_msg}\n\n"
                            f"Code lỗi:\n{code}"
                        )},
                    ], tokenize=False, add_generation_prompt=True)
                    correction_prompts_map[idx] = cor_prompt
                    next_active.append(idx)
                else:
                    # Hết lượt → Fallback Agent
                    logger.info("      [Fallback] Index %d đã thử sai %d lần. Suy luận văn bản...",
                                idx, max_retries)
                    fb_prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Câu hỏi: {state['questions'][idx]['question']}\n"
                            f"Lựa chọn:\n{format_choices(state['questions'][idx]['choices'])}\n\n"
                            f"Mã nguồn lỗi:\n{code}\n\n"
                            f"Log lỗi:\n{err_msg}"
                        )},
                    ], tokenize=False, add_generation_prompt=True)

                    fb_out = batch_inference(
                        model, tokenizer, [fb_prompt],
                        max_new_tokens=cfg_fallback.max_new_tokens,
                        temperature=cfg_fallback.temperature,
                        micro_batch_size=1,
                    )
                    state["final_answers"][idx] = fb_out[0]

        save_checkpoint(state)

        # Chạy LLM Correction cho các câu cần sửa
        if next_active and attempt < max_retries - 1:
            logger.info("   [Correction] Vá lỗi %d câu...", len(next_active))
            cor_batch = cfg_correction.batch_size
            for c_i in range(0, len(next_active), cor_batch):
                c_end = min(c_i + cor_batch, len(next_active))
                sub_batch = next_active[c_i:c_end]

                logger.info("      [Correction] Câu %d đến %d / %d...",
                            c_i + 1, c_end, len(next_active))

                prompts_batch = [correction_prompts_map[idx] for idx in sub_batch]
                raw_cor_codes = batch_inference(
                    model, tokenizer, prompts_batch,
                    max_new_tokens=cfg_correction.max_new_tokens,
                    temperature=cfg_correction.temperature,
                    micro_batch_size=cor_batch,
                )

                for j, raw_code in enumerate(raw_cor_codes):
                    state["execution_codes"][sub_batch[j]] = _extract_code(raw_code)

                save_checkpoint(state)

    return state
