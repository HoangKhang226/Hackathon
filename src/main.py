"""Entry Point — Khởi tạo mô hình, nạp dữ liệu và điều phối Multi-Agent Graph."""

import os
import time

from src.core.config import settings
from src.core.llm_engine import load_model
from src.utils.logger import logger

from src.agents.state import init_state
from src.agents.router import router_node
from src.agents.fast_qa import fast_qa_node
from src.agents.reading import reading_node
from src.agents.coder import coder_sandbox_node

from src.pipeline.io_handler import load_test_data, save_predictions
from src.pipeline.checkpointing import load_checkpoint, save_checkpoint
from src.pipeline.majority_voting import run_majority_voting
from src.pipeline.dynamic_mapper import map_final_answers


def main():
    start_time = time.time()
    logger.info("===========================================================================")
    logger.info("BẮT ĐẦU CHẠY PIPELINE MULTI-AGENT HACKAITHON 2026")
    logger.info("===========================================================================")

    # 1. Khởi tạo môi trường
    os.makedirs(settings.paths.output_dir, exist_ok=True)

    # 2. Nạp mô hình LLM Unsloth
    model, tokenizer = load_model()

    # 3. Nạp dữ liệu
    questions = load_test_data()
    state = init_state(questions)

    # 4. Nạp checkpoint cũ (nếu có)
    state = load_checkpoint(state)

    # 5. Node 1: Router Agent (Phân luồng)
    state = router_node(state, model, tokenizer)

    # 6. Node 2: Fast-QA Agent (Tri thức nền)
    state = fast_qa_node(state, model, tokenizer)

    # 7. Node 3: Reading Comprehension Agent (Đọc hiểu)
    state = reading_node(state, model, tokenizer)

    # 8. Node 4 & 5: Coder & Sandbox Loop (Toán học & Code)
    state = coder_sandbox_node(state, model, tokenizer)

    # 9. Node 5 (Voting): Majority Voting cho các câu CODEABLE
    codeable_indices = [i for i, r in enumerate(state["routes"]) if r == "CODEABLE"]
    if codeable_indices:
        state = run_majority_voting(state, model, tokenizer, codeable_indices)

    # 10. Node 6: Answer Voter & Dynamic Mapper (Chuẩn hoá đáp án)
    state = map_final_answers(state)

    # 11. Ghi file kết quả pred.csv và lưu checkpoint cuối cùng
    save_predictions(state)
    save_checkpoint(state)

    total_time = time.time() - start_time
    logger.info("===========================================================================")
    logger.info("HOÀN THÀNH PIPELINE TRONG %.2f PHÚT (%.2f GIÂY)", total_time / 60, total_time)
    logger.info("===========================================================================")


if __name__ == "__main__":
    main()
