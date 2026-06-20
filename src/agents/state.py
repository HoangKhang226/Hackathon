from typing import List, TypedDict

class GraphState(TypedDict):
    questions: List[dict]        # {qid, question, choices}
    choice_counts: List[int]     # số phương án mỗi câu (4, 5,...)
    routes: List[str]            # FAST_QA, READING, CODEABLE
    execution_codes: List[str]   # code Python sinh ra (luồng CODEABLE)
    sandbox_results: List[dict]  # {success, stdout, stderr}
    final_answers: List[str]     # đáp án chốt cuối: A/B/C/D


def init_graph_state(questions: List[dict]) -> GraphState:
    """
    Khởi tạo trạng thái đồ thị ban đầu với các câu hỏi đã cho.
    """
    choice_counts = [len(q.get("choices", [])) for q in questions]
    routes = [""] * len(questions)
    execution_codes = [""] * len(questions)
    sandbox_results = [{}] * len(questions)
    final_answers = [""] * len(questions)

    return GraphState(
        questions=questions,
        choice_counts=choice_counts,
        routes=routes,
        execution_codes=execution_codes,
        sandbox_results=sandbox_results,
        final_answers=final_answers
    )