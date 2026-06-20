"""GraphState — Trạng thái toàn cục luân chuyển qua đồ thị đa tác tử."""

from typing import List, TypedDict


class GraphState(TypedDict):
    questions: List[dict]        # [{qid, question, choices}, ...]
    choice_counts: List[int]     # Số phương án mỗi câu (4, 5, ...)
    routes: List[str]            # FAST_QA, READING, CODEABLE
    execution_codes: List[str]   # Code Python sinh ra (luồng CODEABLE)
    sandbox_results: List[dict]  # {success, stdout, stderr}
    final_answers: List[str]     # Đáp án thô cuối cùng của từng Node


LETTER_MAP = {i: chr(65 + i) for i in range(26)}


def format_choices(choices: list) -> str:
    """Format danh sách lựa chọn thành chuỗi A. xxx\\nB. yyy..."""
    return "\n".join(f"{LETTER_MAP[i]}. {c}" for i, c in enumerate(choices))


def init_state(questions: list) -> GraphState:
    """Khởi tạo GraphState mới từ danh sách câu hỏi."""
    n = len(questions)
    return GraphState(
        questions=questions,
        choice_counts=[len(q["choices"]) for q in questions],
        routes=[""] * n,
        execution_codes=[""] * n,
        sandbox_results=[{}] * n,
        final_answers=[""] * n,
    )
