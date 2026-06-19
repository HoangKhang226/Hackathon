from unsloth import FastLanguageModel
import torch, gc

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-E4B-it",
    max_seq_length=2048, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def cleanup_vram():
    gc.collect()
    torch.cuda.empty_cache()

def batch_inference(prompts, max_new_tokens=256, temperature=0.2):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  temperature=temperature, do_sample=(temperature > 0))
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    del inputs, outputs
    cleanup_vram()
    return results