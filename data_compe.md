# TÓM TẮT YÊU CẦU & THỂ LỆ - BẢNG C (INNOVATOR)

Tài liệu này tóm tắt toàn bộ yêu cầu cốt lõi từ Thông báo số 1 của BTC để phục vụ cho việc nộp bài, giúp bạn có thể an tâm xóa các file PDF và Text chứa luật gốc cho dự án gọn gàng.

## 1. Yêu cầu về Giải pháp Kỹ thuật (Những gì được phép làm)
- **Mục tiêu:** Xây dựng một AI Agent đa tác vụ có khả năng giải quyết chính xác các câu hỏi trắc nghiệm đa lĩnh vực.
- **Giới hạn mô hình ngôn ngữ (LLM):** Chỉ được phép sử dụng các dòng mô hình cỡ nhỏ/vừa:
  - `Qwen3.5 Series` (<= 9B)
  - `Gemma-4 Series` (Đây là model đội đang dùng).
- **Mô hình hỗ trợ (Embedding / Rerank):** Được phép dùng `BGE-m3` hoặc `Qwen-Rerank` (tùy chọn nếu dùng RAG).

## 2. Tiêu chí Đánh giá Vòng 2
Hệ thống của đội sẽ được chấm chéo bằng file `private_test.csv` (2000 câu) chạy tự động trên server BTC với thang điểm như sau:
- **Độ chính xác (Accuracy):** 80 điểm (Quan trọng nhất).
- **Tốc độ xử lý (Inference Time - Reg/s):** 10 điểm.
- **Tính sáng tạo (Innovation):** 10 điểm.

## 3. Kiến trúc Nộp bài (Docker)
Để hệ thống tự động của BTC chấm được điểm, mã nguồn phải tuân thủ chuẩn I/O sau:
- **Input:** Tự động đọc file đề bài định dạng CSV nằm tại thư mục `/data` (Ví dụ: `/data/public_test.csv` hoặc `/data/private_test.csv`).
- **Output:** Tự động ghi kết quả ra file CSV tại thư mục `/output/pred.csv`.
- **Định dạng Output:** Bắt buộc chỉ có đúng 2 cột là `qid` (ID câu hỏi) và `answer` (Chỉ chứa một ký tự duy nhất: `A`, `B`, `C` hoặc `D`).

## 4. Danh sách các Mục Cần Nộp (Checklist)
Hồ sơ dự thi được nộp trực tuyến và bắt buộc phải có đủ 3 thành phần sau:

- [ ] **1. Docker Container:** Phải được đóng gói (Build) và đẩy (Push) lên registry của **Docker Hub**. Bạn sẽ nộp đường dẫn Docker Image này cho BTC.
- [ ] **2. Mã nguồn Github:** Một đường dẫn Repository chứa toàn bộ source code của dự án, kèm hướng dẫn chi tiết cách chạy để Ban Giám Khảo có thể reproduce lại kết quả y hệt như trong container. (Chính là file `README.md` đang có).
- [ ] **3. Tài liệu thuyết minh phương pháp:** Bản báo cáo mô tả cách thức tối ưu mô hình, chiến lược Multi-Agent và tính sáng tạo của giải pháp. (Chính là file `bao_cao_bang_C.pdf` mà chúng ta vừa xuất).

---
*Lưu ý: Mọi thứ trong dự án hiện tại (I/O, Dockerfile, Report) đã được tuân thủ 100% theo đúng các luật lệ này.*
