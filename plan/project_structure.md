# Cấu Trúc Thư Mục Dự Án — HackAIthon 2026

Hệ thống được tổ chức theo chuẩn MLOps và Microservices. Tách biệt rõ ràng: Tầng Core, Tầng Agents, Tầng Tools, Tầng Pipeline và Cấu hình Deployment.

```text
HackAIthon_Agent/
│
├── config/                         # Cấu hình môi trường và tham số pipeline
│   ├── logging.yaml                # Cấu hình log hệ thống (format, level)
│   └── settings.yaml               # Tham số: MAX_SEQ_LENGTH, node_batch_size, timeout sandbox
│
├── data/                           # Dữ liệu từ BTC (Mount qua Docker)
│   ├── public_test.csv             
│   └── private_test.csv            
│
├── output/                         # Thư mục xuất kết quả (Mount qua Docker)
│   ├── pred.csv                    # Kết quả cuối (qid, answer)
│   └── pipeline_checkpoint.json    # File state backup chống sập
│
├── deployment/                     # Đóng gói và triển khai
│   ├── Dockerfile                  # Container chạy hệ thống (Base: Unsloth)
│   └── entrypoint.sh               # Kích hoạt src/main.py
│
├── plan/                           # Kế hoạch chi tiết của từng thành viên
│   ├── plan_khang.md               # Leader: Review, QA, Master Plan
│   ├── plan_khai.md                # AI Engineer: Core LLM, Router, Prompts
│   ├── plan_huy.md                 # Backend/AI: Sandbox, Coder, Voting
│   └── plan_cuong.md               # MLOps: I/O, Checkpoint, Mapper, Docker
│
├── scripts/                        # Các script hỗ trợ vòng ngoài
│   ├── run_test_local.py           # Chạy thử 50 câu E2E trên máy local
│   ├── run_docker_build.sh         # Lệnh build docker image nhanh
│   └── run_docker_test.sh          # Chạy thử container mô phỏng BTC
│
├── src/                            # MÃ NGUỒN CHÍNH
│   │
│   ├── core/                       # CÁC KẾT NỐI VÀ CẤU HÌNH DÙNG CHUNG
│   │   ├── __init__.py
│   │   ├── config.py               # Load biến môi trường (Pydantic Settings)
│   │   └── llm_engine.py           # Quản lý Unsloth, batch inference, VRAM cleanup
│   │
│   ├── agents/                     # TẦNG ĐIỀU PHỐI (MULTI-AGENT GRAPH)
│   │   ├── __init__.py
│   │   ├── state.py                # GraphState (TypedDict) luân chuyển qua graph
│   │   ├── router.py               # Node 1: Phân loại FAST_QA / READING / CODEABLE
│   │   ├── fast_qa.py              # Node 2: Zero-shot answer
│   │   ├── reading.py              # Node 3: Chain-of-Thought rà bẫy
│   │   └── coder.py                # Node 4: Sinh code Python
│   │
│   ├── tools/                      # CÔNG CỤ NGOẠI VI
│   │   ├── __init__.py
│   │   └── python_sandbox.py       # Môi trường chạy code cách ly (timeout 5s)
│   │
│   ├── pipeline/                   # XỬ LÝ I/O, HỢP NHẤT VÀ KIỂM SOÁT
│   │   ├── __init__.py
│   │   ├── io_handler.py           # Đọc/Ghi file CSV
│   │   ├── checkpointing.py        # Lưu và nạp state.json
│   │   ├── majority_voting.py      # Bầu chọn kết quả Toán học
│   │   └── dynamic_mapper.py       # Ánh xạ kết quả về A/B/C/D
│   │
│   ├── utils/                      # CÁC HÀM TIỆN ÍCH DÙNG CHUNG
│   │   ├── __init__.py
│   │   └── logger.py               # Setup structured loggers
│   │
│   └── main.py                     # Entry point khởi tạo và chạy pipeline
│
├── requirements.txt                # Các thư viện: unsloth, pandas, sympy, pydantic...
└── README.md                       # Tài liệu nộp BTC
```

---

## Giải Thích Luồng Xử Lý (Data Flow)

### Luồng Startup (Chạy 1 lần khi Docker khởi động)
```text
deployment/entrypoint.sh 
    → src/main.py 
        → src/core/logger.py (Init Log)
        → src/core/llm_engine.py (Load Unsloth Model)
        → src/pipeline/checkpointing.py (Load resume state)
```

### Luồng Inference (Chạy lặp qua từng Batch)
1. **Data:** `io_handler.py` đẩy batch câu hỏi vào `GraphState`.
2. **Routing:** `router.py` đọc batch, gán nhãn cho từng câu.
3. **Execution:** Tùy nhãn mà đẩy vào `fast_qa.py`, `reading.py` hoặc `coder.py`.
4. **Sandbox (Nếu là Code):** `coder.py` gọi `python_sandbox.py`. Lỗi thì gọi self-correction. Chốt thì gọi `majority_voting.py`.
5. **Mapping:** Mọi output cuối cùng đưa qua `dynamic_mapper.py` chốt đáp án.
6. **Save:** Gọi `checkpointing.py` lưu State và lặp lại batch tiếp theo.
