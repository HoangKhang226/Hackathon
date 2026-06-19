from typing import List, TypedDict

class GraphState(TypedDict):
    questions: List[dict]        # {qid, question, choices}
    choice_counts: List[int]     # số phương án mỗi câu (4, 5,...)
    routes: List[str]            # FAST_QA, READING, CODEABLE
    execution_codes: List[str]   # code Python sinh ra (luồng CODEABLE)
    sandbox_results: List[dict]  # {success, stdout, stderr}
    final_answers: List[str]     # đáp án chốt cuối: A/B/C/D