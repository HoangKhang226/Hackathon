import re
from typing import Callable
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from src.agents.state import GraphState
from src.core.llm_engine import unsloth_json_batch_inference

# Bảng chữ cái ánh xạ đáp án
LETTER_MAP = {i: chr(65 + i) for i in range(26)}

def format_choices(choices: list) -> str:
    return "\n".join(f"{LETTER_MAP[i]}. {c}" for i, c in enumerate(choices))

class RouterSchema(BaseModel):
    route: Literal["FAST_QA", "READING", "CODEABLE"] = Field(
        description="Phân luồng xử lý chính xác cho câu hỏi."
    )

def llm_router_node(state: GraphState, model, tokenizer, checkpoint_callback: Callable = None) -> GraphState:
    """Agent phân loại câu hỏi vào 3 luồng: FAST_QA, READING, CODEABLE."""
    print("\n" + "="*75 + "\n[Node 1]: Khởi chạy LLM Router Agent\n" + "="*75)

    # Đếm số phương án nếu chưa có
    if not state.get("choice_counts"):
        state["choice_counts"] = [len(q["choices"]) for q in state["questions"]]

    node_batch_size = 10
    total_questions = len(state["questions"])

    for i in range(0, total_questions, node_batch_size):
        end_idx = min(i + node_batch_size, total_questions)

        # Bỏ qua nếu batch này đã route xong (phục vụ checkpoint)
        if all(state["routes"][k] != "" for k in range(i, end_idx)):
            continue

        print(f"      [Router Progress] Đang xử lý câu thứ {i + 1} đến {end_idx} / Tổng {total_questions} câu...")
        batch_questions = state["questions"][i:end_idx]
        prompts = []
        for q in batch_questions:
            prompt = tokenizer.apply_chat_template([
                {
                    "role": "system",
                    "content": (
                        "Bạn là bộ định tuyến dữ liệu siêu tốc của hệ thống LangGraph. Nhiệm vụ của bạn là phân loại câu hỏi đầu vào vào một nhóm duy nhất.\n"
                        "Bắt buộc phải trả về dữ liệu cấu trúc dưới dạng JSON khớp với schema sau:\n"
                        '{"route": "FAST_QA" | "READING" | "CODEABLE"}\n\n'
                        "Quy tắc phân loại:\n"
                        "1. 'CODEABLE': Dành cho câu hỏi định lượng. Yêu cầu lập công thức, giải phương trình hoặc tính toán con số toán, lý , hóa(Không bao gồm hỏi năm lịch sử).\n"
                        "2. 'READING': Dành cho câu hỏi có đính kèm ĐOẠN VĂN/BÀI ĐỌC DÀI. Yêu cầu đọc hiểu sâu và phân tích ngữ liệu để tránh bẫy.\n"
                        "3. 'FAST_QA': Dành cho câu hỏi kiến thức độc lập HOẶC chỉ đính kèm TRÍCH DẪN RẤT NGẮN (1-2 câu). Có thể trả lời ngay bằng trí nhớ.\n"
                        "Chỉ trả ra chuỗi JSON sạch. Tuyệt đối không viết thêm lời dẫn."
                    )
                },
                {"role": "user", "content": f"Câu hỏi: {q['question']}\nLựa chọn:\n{format_choices(q['choices'])}"}
            ], tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Suy luận batch (Temperature = 0.0 để đảm bảo tính nhất quán)
        raw_outputs = unsloth_json_batch_inference(model, tokenizer, prompts, max_new_tokens=32, temperature=0.0, node_batch_size=node_batch_size)

        for j, raw_out in enumerate(raw_outputs):
            try:
                json_str = re.search(r'\{.*\}', raw_out, re.DOTALL).group(0)
                validated_data = RouterSchema.model_validate_json(json_str)
                state["routes"][i + j] = validated_data.route
            except Exception:
                # Fallback an toàn về FAST_QA
                state["routes"][i + j] = "FAST_QA"

        if checkpoint_callback:
            checkpoint_callback(state)

    print(f"[Router Kết Quả] Hoàn tất định tuyến cho {total_questions} câu.")
    return state
