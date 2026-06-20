import gc
import time
# pyrefly: ignore [missing-import]
import torch
from typing import List

# Khởi tạo Unsloth
# pyrefly: ignore [missing-import]
from unsloth import FastLanguageModel

def load_llm_engine():
    """Tải mô hình Gemma-4-E4B-it chuẩn Bảng C qua Unsloth 4-bit."""
    print("[Hệ thống] Đang khởi tạo LLM Engine (google/gemma-4-E4B-it) qua Unsloth 4-bit...")
    
    # Ép fix lỗi float8 trên một số bản PyTorch
    if not hasattr(torch, "float8_e8m0fnu"):
        setattr(torch, "float8_e8m0fnu", torch.float32)

    max_seq_length = 2048 # Đủ cho Reading
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-4-E4B-it",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def cleanup_vram():
    """Dọn dẹp rác VRAM GPU sau mỗi batch để chống OOM."""
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

def unsloth_json_batch_inference(model, tokenizer, prompts: List[str], max_new_tokens: int = 256, temperature: float = 0.2, node_batch_size: int = 10) -> List[str]:
    """Hàm lõi chạy batch inference và tự dọn VRAM."""
    results = []
    total_prompts = len(prompts)

    for i in range(0, total_prompts, node_batch_size):
        end_idx = min(i + node_batch_size, total_prompts)
        batch = prompts[i : end_idx]
        
        # Tokenize
        inputs = tokenizer(text=batch, return_tensors="pt", padding=True).to("cuda")

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True
        }

        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        # Generate text
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            generated_tokens = output[input_len:]
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(decoded_text.strip())

        # Cleanup VRAM strictly
        del inputs
        del outputs
        cleanup_vram()

    return results
