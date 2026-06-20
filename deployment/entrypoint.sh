#!/bin/bash
# entrypoint.sh — Kích hoạt pipeline Multi-Agent chạy trong container

echo "[Container] Khởi động HackAIthon Multi-Agent Pipeline..."
python -m src.main
