"""
Majority Voting — Thực hiện bầu chọn đáp án kết hợp kết quả từ Python Sandbox và LLM.
"""

import json
import re
import logging
from collections import Counter
from src.agents.state import GraphState
from src.core.config import settings
from src.core.llm_engine import batch_inference
from src.pipeline.checkpointing import save_checkpoint

logger = logging.getLogger("HackAIthon_Agent")

# ===========================================================================
# PROMPTS CẤU HÌNH THEO ĐÚNG PLAN_HUY.MD
# ===========================================================================

VOTING_SYSTEM_PROMPT = (
    "Bạn là chuyên gia toán học. Dựa trên câu hỏi, phương án và kết quả gợi ý định lượng, "
    "chọn đáp án đúng nhất.\n"
    "Bắt buộc trả về định dạng JSON sạch khớp chính xác mẫu sau:\n"
    '{"reasoning": "Lập luận ngắn gọn đối chiếu gợi ý (tối đa 1 đến 2 câu)", '
    '"answer": "Chữ cái viết hoa đáp án đúng nhất"}\n'
    "Không viết thêm bất kỳ lời dẫn nào ngoài JSON."
)

VOTING_USER_TEMPLATE = (
    "[Kết quả Python gợi ý]: {stdout_val}\n"
    "[Khớp phương án]: {closest_letter}\n\n"
    "Câu hỏi: {question}\n"
    "Lựa chọn: {choices}\n"
)


def _parse_voting_answer(raw_output: str) -> str:
    """Parse đáp án chữ cái từ kết quả trả về của LLM Voting."""
    # Thử parse JSON trước
    try:
        # Tìm cụm {} đầu tiên
        json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            ans = data.get("answer", "").strip().upper()
            if ans in ["A", "B", "C", "D", "E", "F", "G"]:
                return ans
    except Exception:
        pass

    # Regex trích xuất chữ cái viết hoa đứng sau từ khóa
    patterns = [
        r'"answer"\s*:\s*"([A-G])"',
        r'\b([A-G])\b',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw_output, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
            
    return "A"  # Mặc định an toàn


def format_choices(choices: list) -> str:
    """Định dạng list choices thành chuỗi nhãn A, B, C, D."""
    labels = ["A", "B", "C", "D", "E", "F", "G"]
    return "  ".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


def run_majority_voting(state: GraphState, model, tokenizer, indices: list) -> GraphState:
    """
    Thực hiện bỏ phiếu bầu chọn đáp án cho các câu hỏi CODEABLE.
    Cơ chế:
      - Sandbox thành công khớp phương án A → A được +2 phiếu.
      - Chạy LLM voting 3 vòng (temperature=0.3). Mỗi vòng +1 phiếu.
    """
    if not indices:
        return state

    vote_cfg = settings.agents.voting
    logger.info(f"[Huy-Voting] Bắt đầu Majority Voting cho {len(indices)} câu hỏi...")

    for idx in indices:
        q = state["questions"][idx]
        choices_str = format_choices(q["choices"])
        votes = Counter()

        ans_state = state["final_answers"][idx]
        stdout_val = "N/A (Sandbox Lỗi/Không chạy)"
        closest_letter = "N/A"

        # 1. Kiểm tra kết quả từ Sandbox của Huy
        if ans_state.startswith("SUCCESS_MATCH:"):
            parts = ans_state.split(":", 2)
            closest_letter = parts[1]
            stdout_val = parts[2]
            
            # Gợi ý số học khớp đáp án được +2 phiếu
            votes[closest_letter] += 2
            logger.info(f"   [Voting] Index {idx}: Sandbox gợi ý '{closest_letter}' (+2 phiếu)")
        
        elif "reasoning" in ans_state:
            # Nếu Sandbox lỗi và đã chạy qua Fallback Agent
            fallback_ans = _parse_voting_answer(ans_state)
            votes[fallback_ans] += 1
            logger.info(f"   [Voting] Index {idx}: Nhận gợi ý từ Fallback Agent '{fallback_ans}' (+1 phiếu)")

        # 2. Chạy LLM bỏ phiếu 3 vòng độc lập
        user_content = VOTING_USER_TEMPLATE.format(
            stdout_val=stdout_val,
            closest_letter=closest_letter,
            question=q["question"],
            choices=choices_str
        )
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": VOTING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ], tokenize=False, add_generation_prompt=True)

        logger.info(f"   [Voting] Đang lấy ý kiến LLM ({vote_cfg.num_runs} vòng)...")
        
        # Gọi batch_inference cho các vòng (hoặc gọi từng vòng để tăng tính độc lập)
        prompts = [prompt] * vote_cfg.num_runs
        raw_outputs = batch_inference(
            model,
            tokenizer,
            prompts,
            max_new_tokens=256,
            temperature=vote_cfg.temperature
        )

        for round_idx, raw_out in enumerate(raw_outputs):
            ans = _parse_voting_answer(raw_out)
            votes[ans] += 1
            logger.info(f"      - Vòng {round_idx + 1}: Chọn đáp án {ans} (+1 phiếu)")

        # 3. Chốt đáp án chiến thắng có số phiếu cao nhất
        winner = votes.most_common(1)[0][0]
        logger.info(f"   [Voting] Kết quả bình chọn Index {idx}: {dict(votes)} -> Thắng cuộc: {winner}")
        
        # Ghi đè kết quả chốt cuối cùng
        state["final_answers"][idx] = winner

        save_checkpoint(state)

    return state
