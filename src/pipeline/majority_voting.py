"""Majority Voting — Bầu chọn đáp án cho câu hỏi toán học."""

from collections import Counter
from typing import List

from src.agents.state import GraphState, format_choices
from src.core.config import settings
from src.core.llm_engine import batch_inference
from src.pipeline.dynamic_mapper import parse_answer_to_letter
from src.pipeline.checkpointing import save_checkpoint

import logging
logger = logging.getLogger("HackAIthon_Agent")


def run_majority_voting(
    state: GraphState, model, tokenizer, indices: List[int], checkpoint_path: str = None
) -> GraphState:
    """
    Majority Voting cho các câu CODEABLE.
    Sandbox khớp → +2 phiếu. LLM 3 vòng → +3 phiếu.
    """
    if not indices:
        return state

    cfg = settings.agents.voting
    vote_batch_size = cfg.batch_size  # = 2 (chống OOM)
    n_votes = cfg.num_runs            # = 3

    logger.info("[Voting] Bắt đầu Majority Voting cho %d câu (batch_size=%d, n_votes=%d)...",
                len(indices), vote_batch_size, n_votes)

    for v_i in range(0, len(indices), vote_batch_size):
        v_end = min(v_i + vote_batch_size, len(indices))
        batch_indices = indices[v_i:v_end]

        # Skip nếu tất cả đã chốt chữ cái đơn
        if all(
            len(state["final_answers"][idx]) == 1
            and state["final_answers"][idx] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for idx in batch_indices
        ):
            continue

        # Tạo voting prompts
        voting_prompts = []
        for idx in batch_indices:
            q = state["questions"][idx]
            current_status = state["final_answers"][idx]

            if current_status.startswith("SUCCESS_MATCH"):
                _, closest_letter, stdout_val = current_status.split(":", 2)
                python_hint = (
                    f"\n[Kết quả chạy chương trình Python gợi ý]: {stdout_val}"
                    f"\n[Khớp phương án số học tương ứng]: {closest_letter}"
                )
            elif "reasoning" in current_status:
                python_hint = (
                    f"\n[Lưu ý hệ thống]: Sandbox lỗi toán học. "
                    f"Kết quả phân tích suy luận văn bản dự phòng: {current_status}"
                )
            else:
                python_hint = (
                    "\n[Lưu ý hệ thống]: Không thể thực thi mã Python "
                    "hoặc trích xuất suy luận văn bản chuẩn xác."
                )

            prompt = tokenizer.apply_chat_template([
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia toán học xuất sắc. "
                        "Dựa trên câu hỏi, phương án trắc nghiệm và kết quả gợi ý định lượng, "
                        "hãy chọn đáp án đúng nhất.\n"
                        "Bắt buộc phải trả về định dạng JSON sạch:\n"
                        '{"reasoning": "Suy luận logic ngắn gọn", '
                        '"answer": "Chữ cái lựa chọn đúng duy nhất"}\n'
                        "Tuyệt đối không giải thích thêm ngoài khối JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"/think\nCâu hỏi: {q['question']}\n"
                        f"Lựa chọn:\n{format_choices(q['choices'])}\n"
                        f"{python_hint}"
                    ),
                },
            ], tokenize=False, add_generation_prompt=True)
            voting_prompts.append(prompt)

        # Chạy n_votes vòng bỏ phiếu
        all_votes_matrix = []
        for vote_round in range(n_votes):
            logger.info("   [Voting Round %d/%d] Câu %d đến %d...",
                        vote_round + 1, n_votes, v_i + 1, v_end)
            round_outputs = batch_inference(
                model, tokenizer, voting_prompts,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                micro_batch_size=vote_batch_size,
            )
            all_votes_matrix.append(round_outputs)

        # Đếm phiếu
        for list_pos, idx in enumerate(batch_indices):
            votes = Counter()

            # Phiếu từ LLM (mỗi vòng +1)
            for vote_round in range(n_votes):
                raw_text = all_votes_matrix[vote_round][list_pos]
                parsed_letter = parse_answer_to_letter(raw_text)
                votes[parsed_letter] += 1.0

            # Ưu tiên Sandbox thành công (+2 phiếu)
            if state["final_answers"][idx].startswith("SUCCESS_MATCH"):
                _, closest_letter, _ = state["final_answers"][idx].split(":", 2)
                if closest_letter:
                    votes[closest_letter] += 2.0

            # Chốt đáp án nhiều phiếu nhất
            winner = votes.most_common(1)[0][0]
            state["final_answers"][idx] = winner

        save_checkpoint(state)

    return state
