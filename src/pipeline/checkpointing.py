"""Checkpointing — Lưu/nạp trạng thái pipeline chống sập."""

import json
import os

from src.agents.state import GraphState
from src.core.config import settings

import logging
logger = logging.getLogger("HackAIthon_Agent")


def _get_checkpoint_path() -> str:
    output_dir = settings.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, settings.paths.checkpoint_file)


def save_checkpoint(state: GraphState):
    """Ghi đè toàn bộ state hiện tại xuống ổ đĩa."""
    path = _get_checkpoint_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "routes": state["routes"],
            "execution_codes": state["execution_codes"],
            "sandbox_results": state["sandbox_results"],
            "final_answers": state["final_answers"],
        }, f, ensure_ascii=False, indent=2)


def load_checkpoint(state: GraphState) -> GraphState:
    """
    Nạp checkpoint nếu có. Ghi đè lên state hiện tại
    (giữ nguyên questions và choice_counts từ dữ liệu mới).
    """
    path = _get_checkpoint_path()
    if not os.path.exists(path):
        logger.info("[Checkpoint] Không có checkpoint cũ. Bắt đầu mới.")
        return state

    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    n = len(state["questions"])

    # Chỉ khôi phục nếu kích thước khớp
    if len(saved.get("routes", [])) == n:
        state["routes"] = saved["routes"]
        state["execution_codes"] = saved["execution_codes"]
        state["sandbox_results"] = saved["sandbox_results"]
        state["final_answers"] = saved["final_answers"]

        done_count = sum(1 for a in state["final_answers"] if a != "")
        logger.info("[Checkpoint] Đã khôi phục tiến trình: %d/%d câu đã xử lý.", done_count, n)
    else:
        logger.warning("[Checkpoint] Kích thước checkpoint (%d) không khớp dữ liệu (%d). Bỏ qua.",
                       len(saved.get("routes", [])), n)

    return state
