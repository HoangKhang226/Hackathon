"""LLM Engine — Load model Unsloth, batch inference, VRAM cleanup."""

import gc
import sys
import time
from types import ModuleType
from typing import List

import torch

# ===========================================================================
# COMPATIBILITY PATCHES (phải chạy TRƯỚC import unsloth)
# ===========================================================================

# Patch 1: torch.float8_e8m0fnu (thiếu trên một số phiên bản PyTorch)
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

# Patch 2: all_special_tokens_extended (thiếu trên transformers mới)
import transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

# Patch 3: pyairports dummy module (thiếu trên Kaggle/Docker)
if "pyairports" not in sys.modules:
    dummy = ModuleType("pyairports")
    dummy_airports = ModuleType("pyairports.airports")
    dummy_airports.AIRPORT_LIST = []
    dummy.airports = dummy_airports
    sys.modules["pyairports"] = dummy
    sys.modules["pyairports.airports"] = dummy_airports

# Patch 4: nest_asyncio (cần cho một số môi trường notebook)
import nest_asyncio
nest_asyncio.apply()

# ===========================================================================
# BÂY GIỜ MỚI import Unsloth
# ===========================================================================
from unsloth import FastLanguageModel

from src.core.config import settings

import logging
logger = logging.getLogger("HackAIthon_Agent")


def cleanup_vram():
    """Dọn rác VRAM sau mỗi batch inference."""
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)


def load_model():
    """Nạp model Unsloth 4-bit và chuẩn bị cho inference."""
    logger.info("Đang nạp mô hình: %s (4-bit, seq_len=%d)...",
                settings.llm.model_name, settings.llm.max_seq_length)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings.llm.model_name,
        max_seq_length=settings.llm.max_seq_length,
        load_in_4bit=settings.llm.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    logger.info("Nạp mô hình thành công!")
    return model, tokenizer


def batch_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    micro_batch_size: int = 10,
) -> List[str]:
    """
    Batch inference phòng thủ: chia prompts thành micro-batches,
    decode chỉ phần generated tokens, dọn VRAM sau mỗi micro-batch.
    """
    results = []

    for i in range(0, len(prompts), micro_batch_size):
        batch = prompts[i: i + micro_batch_size]

        inputs = tokenizer(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.llm.max_seq_length,
        ).to("cuda")

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode CHỈ phần generated tokens (bỏ phần prompt)
        for j, output in enumerate(outputs):
            input_len = inputs.input_ids[j].shape[0]
            generated_tokens = output[input_len:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(decoded.strip())

        del inputs, outputs
        cleanup_vram()

    return results
