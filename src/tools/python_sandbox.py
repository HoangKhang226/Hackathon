"""Python Sandbox — Thực thi code Python cách ly và so khớp số học/biểu thức đại số."""

import os
import re
import sys
import subprocess
from typing import List, Tuple, Optional

# Regex để hút số thực
NUM_REGEX = re.compile(r'-?\b\d+\.?\d*')


def execute_code(code_string: str, timeout: int = 5) -> dict:
    """Ghi mã nguồn Python ra file tạm và thực thi cách ly qua subprocess."""
    temp_file = "temp_sandbox.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code_string)
    try:
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True, text=True, timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Timeout Expired > {timeout}s"
        }
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


def extract_numbers(text: str) -> List[float]:
    """Hút toàn bộ số thực từ chuỗi văn bản."""
    # Thay thế phân số dạng a/b bằng giá trị số học tương đương
    fraction_pattern = r'\b(\d+)/(\d+)\b'
    def replace_fraction(match):
        num, denom = float(match.group(1)), float(match.group(2))
        return str(num / denom) if denom != 0 else match.group(0)
    
    processed_text = re.sub(fraction_pattern, replace_fraction, text)
    nums = NUM_REGEX.findall(processed_text)
    
    extracted = []
    for n in nums:
        try:
            extracted.append(float(n))
        except ValueError:
            continue
    return extracted


def find_closest_choice_info(stdout_val: str, choices: List[str]) -> Tuple[Optional[str], float, Optional[float]]:
    """
    So sánh giá trị stdout của sandbox với các phương án trắc nghiệm.
    Trả về: (Chữ cái đáp án, sai số nhỏ nhất, giá trị so khớp).
    """
    stdout_nums = extract_numbers(stdout_val)
    if not stdout_nums:
        # Nếu không có số, so khớp chuỗi đại số (ví dụ: SymPy expressions)
        clean_stdout = stdout_val.strip().lower()
        # Loại bỏ khoảng trắng và dấu thừa để so khớp chuỗi
        clean_stdout_sub = re.sub(r'[\s\*\(\)\-\+\/\_\^]', '', clean_stdout)
        
        best_letter = None
        best_len_diff = float('inf')
        
        for i, choice in enumerate(choices):
            clean_choice = choice.strip().lower()
            # Bỏ ký tự nhãn đầu tiên (A., B., C., D.) nếu có
            if clean_choice.startswith(chr(ord('a') + i).lower()):
                clean_choice = clean_choice[1:].lstrip('.').strip()
            
            clean_choice_sub = re.sub(r'[\s\*\(\)\-\+\/\_\^]', '', clean_choice)
            if clean_stdout_sub == clean_choice_sub and clean_choice_sub != "":
                return chr(ord('A') + i), 0.0, None
                
        return None, float('inf'), None

    # Lấy số cuối cùng in ra từ stdout
    val = stdout_nums[-1]
    
    min_diff = float("inf")
    closest_letter = None
    
    for i, choice in enumerate(choices):
        choice_nums = extract_numbers(choice)
        for cn in choice_nums:
            # Tính sai số tương đối
            diff = abs(cn - val) / max(abs(val), 1.0)
            if diff < min_diff:
                min_diff = diff
                closest_letter = chr(ord('A') + i)
                
    return closest_letter, min_diff, val
