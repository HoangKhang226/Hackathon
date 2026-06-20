from typing import Callable
from src.agents.state import GraphState
from src.core.llm_engine import batch_inference
from src.core.config import settings
from src.utils.logger import logger
from src.pipeline.checkpointing import save_checkpoint

LETTER_MAP = {i: chr(65 + i) for i in range(26)}

def format_choices(choices: list) -> str:
    return "\n".join(f"{LETTER_MAP[i]}. {c}" for i, c in enumerate(choices))

def fast_qa_agent_node(state: GraphState, model, tokenizer, checkpoint_callback: Callable = None) -> GraphState:
    """Agent giải quyết nhanh các câu hỏi tri thức nền tảng."""
    indices = [i for i, r in enumerate(state["routes"]) if r == "FAST_QA"]
    if not indices:
        return state

    logger.info(f"\n[Node 2]: Khởi chạy Fast-QA Agent giải quyết {len(indices)} câu tri thức ngắn (Đường cao tốc)...")
    cfg = settings.agents.qa
    node_batch_size = cfg.batch_size
    total_prompts = len(indices)

    for i in range(0, total_prompts, node_batch_size):
        end_idx = min(i + node_batch_size, total_prompts)
        batch_indices = indices[i:end_idx]

        if all(state["final_answers"][idx] != "" for idx in batch_indices):
            continue

        logger.info(f"      [LLM Batch Progress] Đang xử lý câu thứ {i + 1} đến {end_idx} / Tổng {total_prompts} câu...")
        prompts = []
        for idx in batch_indices:
            q = state["questions"][idx]

            fast_qa_few_shots = (
                "Ví dụ mẫu:\n"
                "Câu hỏi: Thủ đô của Việt Nam là gì?\n"
                "Lựa chọn:\nA. TP. Hồ Chí Minh\nB. Hà Nội\n"
                "Đầu ra mẫu:\n"
                "```json\n"
                "{\n"
                "  \"reasoning\": \"Hà Nội là thủ đô hành chính chính thức của Việt Nam.\",\n"
                "  \"answer\": \"B\"\n"
                "}\n"
                "```\n\n"
            )

            prompt = tokenizer.apply_chat_template([
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia trắc nghiệm tri thức. Hãy phân tích và chọn đáp án đúng duy nhất.\n"
                        "Bắt buộc phải trả về định dạng cấu trúc JSON sạch khớp chính xác với mẫu sau:\n"
                        '{"reasoning": "Suy luận ngắn gọn (độ dài tối đa 1 câu)", "answer": "Chữ cái viết hoa duy nhất (Ví dụ: A hoặc B hoặc C...)"}\n\n'
                        "QUY TẮC PHÒNG THỦ TRÀN TOKEN:\n"
                        "Phần 'reasoning' chỉ được viết đúng 1 câu duy nhất, đi thẳng vào bản chất tri thức nền. Tuyệt đối không giải thích dông dài, không viết ngoài khối JSON."
                    )
                },
                {
                    "role": "user",
                    "content": f"/no_think\n{fast_qa_few_shots}BÂY GIỜ HÃY GIẢI CÂU HỎI SAU VÀ TUÂN THỦ JSON THUẬN CÔ ĐỌNG:\nCâu hỏi: {q['question']}\nLựa chọn:\n{format_choices(q['choices'])}"
                }
            ], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        raw_outputs = batch_inference(
            model, tokenizer, prompts,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            micro_batch_size=cfg.batch_size
        )
        for j, raw_out in enumerate(raw_outputs):
            state["final_answers"][batch_indices[j]] = raw_out

        if checkpoint_callback:
            checkpoint_callback(state)
        save_checkpoint(state)

    return state
