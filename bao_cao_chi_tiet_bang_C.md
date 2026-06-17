# Báo Cáo Phân Tích & Lựa Chọn Giải Pháp AI Agent (Bảng C - INNOVATOR)

Dựa trên yêu cầu từ **Thông báo số 1 (BTC HackAIthon 2026)** và hệ thống mã nguồn hiện tại trong thư mục `report/`, tôi đã tiến hành rà soát chi tiết để tìm ra giải pháp tối ưu nhất cho bạn khi thi đấu tại Bảng C.

---

## 1. Phân Tích Yêu Cầu Bảng C (Từ BTC)

Theo quy định của BTC, Bảng C - INNOVATOR có các tiêu chuẩn và định hướng vô cùng rõ ràng:
- **Mục tiêu cốt lõi:** Thiết kế **AI Agent xử lý đa tác vụ**.
- **Giới hạn Mô hình:** Chỉ được phép sử dụng **Qwen3.5 Series (<=9B)** hoặc **Gemma-4 Series**. Có thể kết hợp mô hình Embedding/Rerank (BGE-m3, Qwen-Rerank).
- **Tiêu chí đánh giá Vòng 2 (Bộ test 2000 câu):**
  - **Độ chính xác (Accuracy):** Chiếm trọng số cực cao **80 điểm**.
  - **Thời gian suy luận (Inference Time):** Chiếm **10 điểm**.
  - **Ý tưởng tối ưu/Sáng tạo:** Chiếm **10 điểm**.
- **Yêu cầu đầu ra:** Chạy trong Docker Container, đọc file test tại `/data` và ghi kết quả (qid, answer A/B/C/D) vào `/output/pred.csv`.

---

## 2. Đánh Giá Các File Trong Thư Mục `report/`

Trong thư mục hiện tại của bạn, mã nguồn được chia làm hai hướng tiếp cận chính:

### Hướng tiếp cận 1: Pipeline Cơ Bản (Single-Pass / Prompt Engineering)
- **Các file:** `gemma4-2.ipynb`, `Advanced_Gemma4_Pipeline.ipynb`
- **Đặc điểm:** Sử dụng các hàm đơn giản để phân loại câu hỏi (Classifier) và xây dựng prompt (Prompt Builder) để gọi LLM trả lời trực tiếp.
- **Hạn chế:** Không có khả năng giải quyết triệt để vấn đề "ảo giác số học" (LLM tính toán toán/lý/hóa sai). Dễ mất điểm ở phần Accuracy và không thể hiện rõ yếu tố "AI Agent" theo yêu cầu của Bảng C.

### Hướng tiếp cận 2: Đồ thị Đa Tác Tử - Multi-Agent Graph
- **Các file mã nguồn:** `hackathon.ipynb`, `kaggle_multiagent.ipynb`
- **Các file tài liệu kiến trúc:** `report_pipeline.md`, `current_pipeline_architecture.md`
- **Đặc điểm:** Xây dựng một luồng Agent hoàn chỉnh (Router -> FastQA / Reading / Coder + Sandbox -> Voting -> Mapper).

---

## 3. Lựa Chọn Tốt Nhất: Pipeline Multi-Agent (`hackathon.ipynb` / `kaggle_multiagent.ipynb`)

**Kết luận:** File **`hackathon.ipynb`** (và bản rút gọn `kaggle_multiagent.ipynb` cùng bộ tài liệu `report_pipeline.md`) là **BẢN ỔN NHẤT VÀ CÓ KHẢ NĂNG TRẢ LỜI TỐT NHẤT**. Đây là giải pháp "Perfect Fit" (hoàn toàn vừa vặn) cho Bảng C với những lý do sau:

### Điểm mạnh tối đa hóa Tiêu Chí Đánh Giá của BTC:
1. **Thể hiện rõ nét tư duy "AI Agent xử lý đa tác vụ" (Ý tưởng: 10 điểm):**
   - Hệ thống dùng **Router Agent** để định tuyến luồng xử lý: Câu ngắn đi vào bộ nhớ (`Fast-QA`), câu dài dùng `Reading`, câu tính toán dùng `Coder`. Việc chia để trị này ghi điểm tuyệt đối về mặt sáng tạo kỹ thuật.

2. **Bảo vệ Độ Chính Xác Cao Nhất (Accuracy: 80 điểm):**
   - Đối phó với điểm yếu chí tử của các LLM nhỏ (Gemma-4) là giải Toán/Lý/Hóa kém. Pipeline này tích hợp **Coder Agent và Python Sandbox**. Thay vì tự tính nhẩm, LLM sinh mã Python, hệ thống chạy code lấy kết quả số học thô.
   - **Vòng lặp tự sửa sai (Self-Correction):** Nếu Python chạy lỗi, Agent tự đọc log lỗi (stderr) và sửa lại code.
   - **Bầu chọn đa số (Majority Voting):** Kết hợp kết quả chạy code và 3 lần suy luận của LLM để chốt đáp án chắc chắn nhất.
   - **Dynamic Mapper:** Chuẩn hóa đầu ra, ép bộ so khớp thuật toán xử lý để luôn trả về `A, B, C, D` (không bao giờ in ra lỗi định dạng như "Đáp án là A").

3. **Tối ưu Thời gian suy luận (Inference Time: 10 điểm):**
   - Được tích hợp bộ khung **Unsloth (4-bit)** và các hàm dọn dẹp bộ nhớ VRAM tự động.
   - Thay vì dùng Chain-of-Thought (CoT - suy luận dài dòng) cho mọi câu hỏi làm chậm thời gian, luồng `Fast-QA` giải quyết câu hỏi ngắn siêu tốc với zero-shot, giúp tăng tốc độ "Reg/s" đáng kể.

---

## 4. Kế Hoạch Hành Động (Next Steps) Để Chinh Phục Vòng 2

Để hoàn thiện sản phẩm theo chuẩn đầu ra của BTC Bảng C, bạn cần thực hiện các bước sau:

1. **Chuyển đổi Notebook sang Script Python:**
   - Đưa toàn bộ logic từ `hackathon.ipynb` / `kaggle_multiagent.ipynb` vào một file `main.py` (hoặc `predict.py`).
2. **Chuẩn hóa Entry-point & I/O:**
   - Thiết lập hàm đọc file bằng `pandas` từ đường dẫn cứng `/data/public_test.csv` (hoặc `private_test.csv`).
   - Ghi kết quả sau khi qua `Mapper Node` thành file `pred.csv` tại folder `/output/`. Cột bắt buộc: `qid, answer`.
3. **Đóng gói Docker:**
   - Viết `Dockerfile` base từ image của Unsloth hoặc vLLM/PyTorch.
   - Đảm bảo cài đặt các thư viện `pandas`, `unsloth`, `sympy` (cho toán).
   - *Lưu ý:* Luồng chạy code Sandbox yêu cầu OS trong Docker phải cho phép chạy tiến trình phụ (`subprocess`), cần cấp đủ quyền thực thi.
4. **Hoàn thiện tài liệu:**
   - Dùng chính nội dung cực kỳ chi tiết trong file `report_pipeline.md` (do bạn đã viết rất tốt) để làm tài liệu thuyết minh phương pháp, nộp kèm mã nguồn lên Github. Tối ưu lại một chút để nhấn mạnh việc pipeline này bám sát tiêu chí của Ban giám khảo.

Chúc bạn đạt điểm tuyệt đối với kiến trúc ấn tượng này!
