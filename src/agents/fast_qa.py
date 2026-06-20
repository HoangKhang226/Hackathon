"""Fast-QA Agent — Zero-shot trả lời nhanh câu hỏi tri thức nền."""

from src.agents.state import GraphState, format_choices
from src.core.config import settings
from src.core.llm_engine import batch_inference
from src.pipeline.checkpointing import save_checkpoint

import logging
logger = logging.getLogger("HackAIthon_Agent")


FAST_QA_SYSTEM_PROMPT = (
    "Bạn là chuyên gia trắc nghiệm tri thức. Hãy phân tích và chọn đáp án đúng duy nhất.\n"
    "Bắt buộc phải trả về định dạng cấu trúc JSON sạch khớp chính xác với mẫu sau:\n"
    '{"reasoning": "Suy luận ngắn gọn (độ dài tối đa 1 câu)", '
    '"answer": "Chữ cái viết hoa duy nhất (Ví dụ: A hoặc B hoặc C...)"}\n\n'
    "QUY TẮC PHÒNG THỦ TRÀN TOKEN:\n"
    "Phần 'reasoning' chỉ được viết đúng 1 câu duy nhất, đi thẳng vào bản chất tri thức nền. "
    "Tuyệt đối không giải thích dông dài, không viết ngoài khối JSON."
)

FAST_QA_FEW_SHOTS = (
    "Ví dụ mẫu:\n"
    "Câu hỏi: Thủ đô của Việt Nam là gì?\n"
    "Lựa chọn:\nA. TP. Hồ Chí Minh\nB. Hà Nội\n"
    "Đầu ra mẫu:\n"
    "```json\n"
    "{\n"
    '  "reasoning": "Hà Nội là thủ đô hành chính chính thức của Việt Nam.",\n'
    '  "answer": "B"\n'
    "}\n"
    "```\n\n"
)


def fast_qa_node(state: GraphState, model, tokenizer) -> GraphState:
    """Node 2: Giải quyết câu hỏi tri thức nền bằng Zero-shot."""
    indices = [i for i, r in enumerate(state["routes"]) if r == "FAST_QA"]
    if not indices:
        return state

    cfg = settings.agents.fast_qa
    total = len(indices)
    logger.info("\n[Node 2] Fast-QA Agent: %d câu tri thức ngắn", total)

    for i in range(0, total, cfg.batch_size):
        end_idx = min(i + cfg.batch_size, total)
        batch_indices = indices[i:end_idx]

        # Skip câu đã trả lời
        if all(state["final_answers"][idx] != "" for idx in batch_indices):
            continue

        logger.info("   [Fast-QA] Câu %d đến %d / %d...", i + 1, end_idx, total)

        prompts = []
        for idx in batch_indices:
            q = state["questions"][idx]
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": FAST_QA_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"/no_think\n{FAST_QA_FEW_SHOTS}"
                    f"BÂY GIỜ HÃY GIẢI CÂU HỎI SAU VÀ TUÂN THỦ JSON THUẬN CÔ ĐỌNG:\n"
                    f"Câu hỏi: {q['question']}\n"
                    f"Lựa chọn:\n{format_choices(q['choices'])}"
                )},
            ], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        raw_outputs = batch_inference(
            model, tokenizer, prompts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            micro_batch_size=cfg.batch_size,
        )

        for j, raw_out in enumerate(raw_outputs):
            state["final_answers"][batch_indices[j]] = raw_out

        save_checkpoint(state)

    return state
