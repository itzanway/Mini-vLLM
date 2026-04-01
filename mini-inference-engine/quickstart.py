#!/usr/bin/env python3
"""
Phase 1 Quick-start: Run this FIRST to verify your GPU / model setup.
No FastAPI, no frontend — just the raw custom generation loop.

Usage:
  pip install torch transformers
  python quickstart.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def top_p_sample(logits: torch.Tensor, temperature: float = 0.8, top_p: float = 0.9) -> int:
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
    remove[0] = False
    logits[sorted_indices[remove]] = float("-inf")
    return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 80):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = []

    print(f"\nPrompt: {prompt}")
    print("Output: ", end="", flush=True)

    with torch.no_grad():
        while len(generated) < max_new_tokens:
            all_ids = torch.cat([input_ids.squeeze(0),
                                 torch.tensor(generated, device=device)]).unsqueeze(0)
            logits = model(input_ids=all_ids).logits[0, -1, :]
            next_id = top_p_sample(logits)

            if next_id == tokenizer.eos_token_id:
                break

            generated.append(next_id)
            token_text = tokenizer.decode([next_id], skip_special_tokens=True)
            print(token_text, end="", flush=True)

    print(f"\n\nGenerated {len(generated)} tokens.")


if __name__ == "__main__":
    model_name = "gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    generate(model, tokenizer, "The future of AI inference is", max_new_tokens=60)
    generate(model, tokenizer, "Continuous batching works by", max_new_tokens=60)