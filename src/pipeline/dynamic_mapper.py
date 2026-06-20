"""Dynamic Mapper — Trích xuất đáp án từ output LLM thô, chuẩn hoá về A/B/C/D."""

import json
import re
from typing import List

from src.agents.state import LETTER_MAP

import logging
logger = logging.getLogger("HackAIthon_Agent")


def parse_answer_to_letter(text: str) -> str:
    """
    Trích xuất chữ cái đáp án từ output LLM thô (7 tầng fallback).
    Xử lý được: JSON, SUCCESS_MATCH, markdown bold, tiếng Việt, tiếng Anh, chữ cái đơn.
    """
    if not text:
        return "A"

    # Tầng 0: Trạng thái SUCCESS_MATCH từ Sandbox
    if text.startswith("SUCCESS_MATCH:"):
        parts = text.split(":")
        if len(parts) >= 2:
            letter = parts[1].strip().upper()
            if len(letter) == 1 and letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return letter

    # Tầng 1: Nếu text là chữ cái đơn lẻ
    text_stripped = text.strip().upper()
    if len(text_stripped) == 1 and text_stripped in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return text_stripped

    # Tầng 2: Parse JSON (cả trong và ngoài khối ```json```)
    text_clean = text.strip()
    json_match = re.search(r'```json\s*(.*?)\s*```', text_clean, re.DOTALL)
    if json_match:
        text_clean = json_match.group(1).strip()
    else:
        start_idx = text_clean.find("{")
        end_idx = text_clean.rfind("}")
        if start_idx != -1 and end_idx != -1:
            text_clean = text_clean[start_idx: end_idx + 1]

    try:
        data = json.loads(text_clean)
        if isinstance(data, dict):
            for key in ["answer", "choice", "result", "prediction"]:
                ans = data.get(key, "").strip().upper()
                if ans and len(ans) == 1 and ans in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    return ans
    except Exception:
        pass

    # Tầng 3: Markdown bold **A**
    match = re.search(r'\*\*([A-Z])\*\*', text)
    if match:
        return match.group(1)

    # Tầng 4: Tiếng Việt — "Đáp án là A" / "đáp án: B"
    match = re.search(r'[đĐ]áp\s*án\s*(?:là|:)?\s*([A-Z])\b', text)
    if match:
        return match.group(1)

    # Tầng 5: Tiếng Anh — "answer is A" / "Answer: B"
    match = re.search(r'[aA]nswer\s*(?:is|:)?\s*([A-Z])\b', text)
    if match:
        return match.group(1)

    # Tầng 6: Chữ cái viết hoa đứng riêng cuối cùng
    matches = re.findall(r'\b([A-Z])\b', text)
    if matches:
        return matches[-1]

    # Tầng 7: Bất kỳ chữ cái viết hoa nào cuối cùng
    matches = re.findall(r'([A-Z])', text)
    if matches:
        return matches[-1]

    return "A"


def map_final_answers(state: dict) -> dict:
    """
    Duyệt qua tất cả final_answers, chuẩn hoá về chữ cái hợp lệ.
    Đảm bảo không vượt quá số phương án thực tế của câu hỏi.
    """
    for i in range(len(state["questions"])):
        raw_answer = state["final_answers"][i]
        num_choices = state["choice_counts"][i]

        # Bước 1: Trích xuất chữ cái
        letter = parse_answer_to_letter(raw_answer)

        # Bước 2: Kiểm tra giới hạn
        idx = ord(letter) - ord("A")
        max_idx = num_choices - 1
        if idx > max_idx:
            letter = "A"  # Fallback an toàn

        state["final_answers"][i] = letter

    return state
