from typing import Callable
from src.agents.state import GraphState
from src.core.llm_engine import unsloth_json_batch_inference

LETTER_MAP = {i: chr(65 + i) for i in range(26)}

def format_choices(choices: list) -> str:
    return "\n".join(f"{LETTER_MAP[i]}. {c}" for i, c in enumerate(choices))

def reading_comprehension_agent_node(state: GraphState, model, tokenizer, checkpoint_callback: Callable = None) -> GraphState:
    """Agent rà soát văn bản dài, có khả năng phát hiện bẫy phủ định."""
    indices = [i for i, r in enumerate(state["routes"]) if r == "READING"]
    if not indices:
        return state

    print(f"\n[Node 3]: Khởi chạy Reading Comprehension Agent bóc tách {len(indices)} câu văn bản ngữ cảnh dài...")
    node_batch_size = 5
    total_prompts = len(indices)

    for i in range(0, total_prompts, node_batch_size):
        end_idx = min(i + node_batch_size, total_prompts)
        batch_indices = indices[i:end_idx]

        if all(state["final_answers"][idx] != "" for idx in batch_indices):
            continue

        print(f"      [LLM Batch Progress] Đang xử lý câu thứ {i + 1} đến {end_idx} / Tổng {total_prompts} câu...")
        prompts = []
        for idx in batch_indices:
            q = state["questions"][idx]

            reading_few_shots = (
                "Ví dụ mẫu:\n"
                "Câu hỏi: Giai đoạn Mạt Pháp bắt đầu khi nào theo tài liệu?\n"
                "Lựa chọn:\nA. 1000 năm\nB. 1500 năm\n"
                "Đầu ra mẫu:\n"
                "```json\n"
                "{\n"
                "  \"reasoning\": \"Tài liệu tại Đoạn 1 ghi nhận thời điểm Mạt Pháp bắt đầu là 1500 năm sau khi Thích Ca nhập niết bàn, khớp với phương án B.\",\n"
                "  \"answer\": \"B\"\n"
                "}\n"
                "```\n\n"
            )

            prompt = tokenizer.apply_chat_template([
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia rà soát bẫy văn bản dài. Hãy đối chiếu chi tiết các tình huống, rà soát kỹ các từ phủ định (không phải, ngoại trừ, sai) để trích xuất thông tin.\n"
                        "Bắt buộc phải trả về định dạng cấu trúc JSON sạch khớp chính xác với mẫu sau:\n"
                        '{"reasoning": "Đối chiếu bối cảnh tài liệu (tối đa 1 đến 2 câu ngắn)", "answer": "Chữ cái đáp án viết hoa đúng nhất"}\n\n'
                        "QUY TẮC CHỐNG CẮT CỤT CHUỖI DO HẾT TOKEN:\n"
                        "Phần 'reasoning' phải viết cực kỳ cô đọng, đi thẳng vào việc chỉ rõ Đoạn/Tài liệu nào chứa từ khóa để chốt đáp án. Tuyệt đối không chép lại cả đoạn văn bản dài vào JSON."
                    )
                },
                {
                    "role": "user",
                    "content": f"/think\n{reading_few_shots}BÂY GIỜ HÃY ĐỐI CHIẾU VĂN BẢN VÀ GIẢI CÂU HỎI SAU:\nCâu hỏi: {q['question']}\nLựa chọn:\n{format_choices(q['choices'])}"
                }
            ], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Suy luận batch (Temperature = 0.2, max_tokens tăng lên 384 do cần tư duy CoT dài hơn chút)
        raw_outputs = unsloth_json_batch_inference(model, tokenizer, prompts, max_new_tokens=384, temperature=0.2, node_batch_size=node_batch_size)
        
        for j, raw_out in enumerate(raw_outputs):
            state["final_answers"][batch_indices[j]] = raw_out

        if checkpoint_callback:
            checkpoint_callback(state)

    return state
