"""
Compiler Script — Tự động đóng gói toàn bộ code modular trong src/
thành một Jupyter Notebook duy nhất để nộp bài trên Kaggle / Colab.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_NOTEBOOK = PROJECT_ROOT / "submission_kaggle.ipynb"

# Thứ tự ghép các module (đảm bảo định nghĩa trước khi sử dụng)
MODULE_ORDER = [
    "utils/logger.py",
    "core/config.py",
    "agents/state.py",
    "tools/python_sandbox.py",
    "pipeline/dynamic_mapper.py",
    "pipeline/checkpointing.py",
    "pipeline/majority_voting.py",
    "core/llm_engine.py",
    "agents/router.py",
    "agents/fast_qa.py",
    "agents/reading.py",
    "agents/coder.py",
    "main.py"
]


def clean_imports_and_comments(code: str) -> str:
    """Xóa các dòng import nội bộ và cấu hình thừa để chạy trên 1 file."""
    lines = code.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # Bỏ import nội bộ của dự án
        if (stripped.startswith("from src.")
                or stripped.startswith("import src.")
                or stripped == "from src import agents, core, pipeline, tools, utils"):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def compile_to_notebook():
    cells = []

    def add_markdown_cell(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code_cell(code):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")]
        })

    # 1. Tiêu đề & Giới thiệu
    add_markdown_cell(
        "# AI MCQ Multi-Agent Pipeline — Kaggle Submission Edition\n\n"
        "Kiến trúc Đồ thị đa tác tử (Multi-Agent Graph) kết hợp Unsloth 4-bit, "
        "Python Sandbox thực thi code và Majority Voting.\n"
        "Được đóng gói tự động từ cấu trúc modular của dự án."
    )

    # 2. Cell cài đặt thư viện trên Kaggle
    add_markdown_cell("### 1. Cài đặt Unsloth & Các thư viện bổ trợ")
    install_code = (
        "# Cài đặt Unsloth tương thích tốt nhất với môi trường PyTorch trên Kaggle\n"
        '!pip install -q "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"\n'
        "!pip install -q pandas sympy pydantic pydantic-settings python-dotenv PyYAML tqdm"
    )
    add_code_cell(install_code)

    # 3. Đọc và nhúng trực tiếp settings.yaml làm cấu hình fallback
    add_markdown_cell("### 2. Cấu hình mặc định (settings.yaml)")
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            yaml_content = f.read()
    else:
        yaml_content = "llm:\n  model_name: 'google/gemma-4-E4B-it'\n  max_seq_length: 4096\n  load_in_4bit: true"

    # Tạo code python để tạo file settings.yaml động khi chạy trên Kaggle
    settings_creator = (
        "import os\n"
        "os.makedirs('config', exist_ok=True)\n"
        f"with open('config/settings.yaml', 'w', encoding='utf-8') as f:\n"
        f"    f.write('''{yaml_content}''')\n"
        "print('Đã khởi tạo config/settings.yaml thành công.')"
    )
    add_code_cell(settings_creator)

    # 4. Gộp toàn bộ source code
    add_markdown_cell("### 3. Mã nguồn Hệ thống Multi-Agent")
    combined_code_parts = []

    for mod in MODULE_ORDER:
        file_path = SRC_DIR / mod
        if not file_path.exists():
            print(f"⚠️ Cảnh báo: Không tìm thấy file {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        cleaned_code = clean_imports_and_comments(code)
        combined_code_parts.append(
            f"# ===========================================================================\n"
            f"# MODULE: {mod}\n"
            f"# ===========================================================================\n"
            f"{cleaned_code}\n"
        )

    add_code_cell("\n".join(combined_code_parts))

    # 5. Cell khởi chạy tự động trên Kaggle (quét tự động input data)
    add_markdown_cell("### 4. Khởi chạy Pipeline trên Kaggle")
    run_code = (
        "import os\n\n"
        "# Cấu hình đường dẫn đầu ra làm việc\n"
        "os.makedirs('/kaggle/working/output', exist_ok=True)\n\n"
        "# Tự động tìm file dữ liệu (.csv hoặc .json) trong các datasets đã add vào Kaggle\n"
        "input_dir = '/kaggle/input'\n"
        "found_file = None\n\n"
        "for root, dirs, files in os.walk(input_dir):\n"
        "    for file in files:\n"
        "        if file.endswith('.json') or file.endswith('.csv'):\n"
        "            found_file = os.path.join(root, file)\n"
        "            break\n"
        "    if found_file:\n"
        "        break\n\n"
        "if found_file:\n"
        "    print(f'[Kaggle] Tìm thấy dữ liệu tại: {found_file}')\n"
        "    # Ghi đè cấu hình Settings để trỏ tới thư mục chứa dữ liệu\n"
        "    settings.paths.data_dir = os.path.dirname(found_file)\n"
        "    settings.paths.output_dir = '/kaggle/working/output'\n"
        "    # Gọi hàm main() chạy toàn bộ pipeline\n"
        "    main()\n"
        "else:\n"
        "    print('⚠️ Cảnh báo: Không tìm thấy file dữ liệu nào trong /kaggle/input.')\n"
        "    print('Hãy chắc chắn rằng bạn đã bấm \"Add Input\" ở góc phải và thêm dataset thi đấu.')"
    )
    add_code_cell(run_code)

    # Đóng gói Notebook JSON
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"Compiled successfully! Output path: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    compile_to_notebook()
