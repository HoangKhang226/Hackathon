"""Python Sandbox — Chạy code cách ly, trích xuất số, so khớp đáp án."""

import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple

from src.agents.state import LETTER_MAP


def execute_code(code_string: str, timeout: int = 5) -> dict:
    """Chạy code Python trong subprocess cách ly với timeout."""
    temp_file = "temp_sandbox.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code_string)
    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Lỗi: Quá thời gian thực thi (>{timeout} giây).",
        }
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


def extract_numbers_from_text(text: str) -> List[float]:
    """Trích xuất tất cả số (kể cả phân số) từ chuỗi văn bản."""
    text = text.replace(",", "")

    # Bắt phân số: 1/2, -3/4, ...
    fraction_pattern = r'(-?\b\d+)/(\d+\b)'
    fractions = re.findall(fraction_pattern, text)
    nums = []
    for num, denom in fractions:
        try:
            nums.append(float(num) / float(denom))
        except ZeroDivisionError:
            pass

    # Xoá phân số khỏi text trước khi bắt số thập phân
    text_clean = re.sub(fraction_pattern, "", text)

    # Bắt số thập phân: 3.14, -0.5, 100, ...
    decimal_pattern = r'-?\b\d+\.?\d*'
    for n in re.findall(decimal_pattern, text_clean):
        try:
            nums.append(float(n))
        except ValueError:
            pass

    return nums


def is_numeric_choices(choices: list) -> bool:
    """Kiểm tra tất cả các phương án có chứa đúng 1 số không."""
    for choice in choices:
        nums = extract_numbers_from_text(choice)
        if len(nums) != 1:
            return False
    return True


def clean_text_for_match(text: str) -> str:
    """Xoá ký tự đặc biệt để so khớp chuỗi ký tự."""
    text = text.lower().strip()
    return re.sub(r'[\$\{\}\\\^\_\(\)\*\/\+\-\=\s\[\]\,\.]', '', text)


def find_closest_choice_info(
    python_output: str, choices: list
) -> Tuple[Optional[str], float, Optional[float]]:
    """
    So khớp kết quả Python Sandbox với các phương án trắc nghiệm.
    Trả về: (chữ_cái_đáp_án, sai_số_tương_đối, giá_trị_số_thô)
    """
    # Bước 1: Thử so khớp chuỗi ký tự (cho đáp án biểu thức SymPy)
    py_clean = clean_text_for_match(python_output)
    if py_clean and bool(re.search(r'[a-z]', py_clean)):
        for i, choice in enumerate(choices):
            choice_clean = clean_text_for_match(choice)
            if (py_clean == choice_clean
                    or py_clean in choice_clean
                    or choice_clean in py_clean):
                if len(py_clean) >= 2 or py_clean == choice_clean:
                    return LETTER_MAP[i], 0.0, None

    # Bước 2: So khớp số
    if not is_numeric_choices(choices):
        return None, float("inf"), None

    code_nums = extract_numbers_from_text(python_output)
    if not code_nums:
        return None, float("inf"), None

    target_val = code_nums[-1]
    best_letter, min_diff = None, float("inf")

    for i, choice in enumerate(choices):
        for val in extract_numbers_from_text(choice):
            denom = max(abs(val), 1.0)
            diff = abs(target_val - val) / denom
            if diff < min_diff:
                min_diff = diff
                best_letter = LETTER_MAP[i]

    return best_letter, min_diff, target_val
