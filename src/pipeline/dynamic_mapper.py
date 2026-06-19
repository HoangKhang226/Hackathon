import re
def clean_text(text: str) -> str:
    return re.sub(r'[\$\{\}\\\^\_\(\)\*\/\+\-\=\s\[\]\,\.]', '', text.lower().strip())

def map_answer(llm_output: str, choices: list, num_choices: int) -> str:
    clean_out = clean_text(llm_output)
    best_idx, best_score = 0, -1
    for i, choice in enumerate(choices):
        score = len(set(clean_text(choice)) & set(clean_out))
        if score > best_score:
            best_score = score
            best_idx = i
    # Kiểm tra không vượt quá số đáp án thực tế
    best_idx = min(best_idx, num_choices - 1)
    return chr(ord('A') + best_idx)

def match_number(stdout_val: str, choices: list) -> str | None:
    nums = re.findall(r'-?\b\d+\.?\d*', stdout_val)
    if not nums:
        return None
    val = float(nums[-1])
    for i, choice in enumerate(choices):
        choice_nums = re.findall(r'-?\b\d+\.?\d*', choice)
        for cn in choice_nums:
            target = float(cn)
            if abs(target - val) / max(abs(val), 1.0) <= 0.01:
                return chr(ord('A') + i)
    return None