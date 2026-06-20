"""Reading Agent — Chain-of-Thought đọc hiểu văn bản dài, rà bẫy phủ định."""

from src.agents.state import GraphState, format_choices
from src.core.config import settings
from src.core.llm_engine import batch_inference
from src.pipeline.checkpointing import save_checkpoint

import logging
logger = logging.getLogger("HackAIthon_Agent")


READING_SYSTEM_PROMPT = (
    "Bạn là chuyên gia rà soát bẫy văn bản dài. "
    "Hãy đối chiếu chi tiết các tình huống, rà soát kỹ các từ phủ định "
    "(không phải, ngoại trừ, sai) để trích xuất thông tin.\n"
    "Bắt buộc phải trả về định dạng cấu trúc JSON sạch khớp chính xác với mẫu sau:\n"
    '{"reasoning": "Đối chiếu bối cảnh tài liệu (tối đa 1 đến 2 câu ngắn)", '
    '"answer": "Chữ cái đáp án viết hoa đúng nhất"}\n\n'
    "QUY TẮC CHỐNG CẮT CỤT CHUỖI DO HẾT TOKEN:\n"
    "Phần 'reasoning' phải viết cực kỳ cô đọng, đi thẳng vào việc chỉ rõ "
    "Đoạn/Tài liệu nào chứa từ khóa để chốt đáp án. "
    "Tuyệt đối không chép lại cả đoạn văn bản dài vào JSON."
)

READING_FEW_SHOTS = (
    "Ví dụ mẫu:\n"
    "Câu hỏi: Giai đoạn Mạt Pháp bắt đầu khi nào theo tài liệu?\n"
    "Lựa chọn:\nA. 1000 năm\nB. 1500 năm\n"
    "Đầu ra mẫu:\n"
    "```json\n"
    "{\n"
    '  "reasoning": "Tài liệu tại Đoạn 1 ghi nhận thời điểm Mạt Pháp bắt đầu '
    'là 1500 năm sau khi Thích Ca nhập niết bàn, khớp với phương án B.",\n'
    '  "answer": "B"\n'
    "}\n"
    "```\n\n"
)


def reading_node(state: GraphState, model, tokenizer) -> GraphState:
    """Node 3: Đọc hiểu văn bản dài bằng Chain-of-Thought."""
    indices = [i for i, r in enumerate(state["routes"]) if r == "READING"]
    if not indices:
        return state

    cfg = settings.agents.reading
    total = len(indices)
    logger.info("\n[Node 3] Reading Agent: %d câu đọc hiểu văn bản dài", total)

    for i in range(0, total, cfg.batch_size):
        end_idx = min(i + cfg.batch_size, total)
        batch_indices = indices[i:end_idx]

        # Skip câu đã trả lời
        if all(state["final_answers"][idx] != "" for idx in batch_indices):
            continue

        logger.info("   [Reading] Câu %d đến %d / %d...", i + 1, end_idx, total)

        prompts = []
        for idx in batch_indices:
            q = state["questions"][idx]
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": READING_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"/think\n{READING_FEW_SHOTS}"
                    f"BÂY GIỜ HÃY ĐỐI CHIẾU VĂN BẢN VÀ GIẢI CÂU HỎI SAU:\n"
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
