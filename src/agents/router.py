"""Router Agent — Phân loại câu hỏi thành FAST_QA / READING / CODEABLE."""

import re

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from src.agents.state import GraphState, format_choices
from src.core.config import settings
from src.core.llm_engine import batch_inference
from src.pipeline.checkpointing import save_checkpoint

import logging
logger = logging.getLogger("HackAIthon_Agent")


class RouterSchema(BaseModel):
    route: Literal["FAST_QA", "READING", "CODEABLE"] = Field(
        description="Phân luồng xử lý chính xác cho câu hỏi."
    )


ROUTER_SYSTEM_PROMPT = (
    "Bạn là bộ định tuyến dữ liệu siêu tốc của hệ thống LangGraph. "
    "Nhiệm vụ của bạn là phân loại câu hỏi đầu vào vào một nhóm duy nhất.\n"
    "Bắt buộc phải trả về dữ liệu cấu trúc dưới dạng JSON khớp với schema sau:\n"
    '{"route": "FAST_QA" | "READING" | "CODEABLE"}\n\n'
    "Quy tắc phân loại:\n"
    "1. 'READING': Nếu câu hỏi là đọc hiểu văn bản dài, có các đoạn thông tin "
    "hoặc đoạn trích lịch sử/pháp luật dài phức tạp.\n"
    "2. 'CODEABLE': Nếu câu hỏi yêu cầu giải toán, tính công thức tài chính/kinh tế, "
    "tính số liệu lý/hóa cụ thể.\n"
    "3. 'FAST_QA': Định nghĩa ngắn, tri thức nền hoặc kiểm tra sự thật đơn giản "
    "dưới 2000 ký tự.\n"
    "Chỉ trả ra chuỗi JSON sạch. Tuyệt đối không viết thêm lời dẫn."
)


def router_node(state: GraphState, model, tokenizer) -> GraphState:
    """Node 1: Phân loại batch câu hỏi."""
    cfg = settings.agents.router
    total = len(state["questions"])

    logger.info("\n" + "=" * 75)
    logger.info("[Node 1] Khởi chạy LLM Router Agent")
    logger.info("=" * 75)

    for i in range(0, total, cfg.batch_size):
        end_idx = min(i + cfg.batch_size, total)

        # Skip câu đã route
        if all(state["routes"][k] != "" for k in range(i, end_idx)):
            continue

        logger.info("   [Router] Đang xử lý câu %d đến %d / %d...", i + 1, end_idx, total)

        prompts = []
        for q in state["questions"][i:end_idx]:
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": (
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
            try:
                json_str = re.search(r'\{.*\}', raw_out, re.DOTALL).group(0)
                validated = RouterSchema.model_validate_json(json_str)
                state["routes"][i + j] = validated.route
            except Exception:
                # Fallback về FAST_QA (KHÔNG PHẢI "A")
                state["routes"][i + j] = "FAST_QA"

        save_checkpoint(state)

    from collections import Counter
    counts = Counter(state["routes"])
    logger.info("[Router] Phân luồng hoàn tất: %s", dict(counts))

    return state
