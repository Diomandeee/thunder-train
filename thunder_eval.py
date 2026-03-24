#!/usr/bin/env python3
"""
thunder_eval.py — Evaluate a Thunder-Train adapter.

Loads the base model + LoRA adapter, runs sample completions,
and computes perplexity on the eval set.

Usage:
  python3 thunder_eval.py \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --adapter-path ~/projects/karl/thunder-adapter-v1 \
    --eval-data ~/projects/karl/autocontinue-data/eval_merged.jsonl \
    --num-samples 5
"""

import argparse
import json
import math
import sys
import time

import mlx.core as mx
import mlx.nn as nn


def load_model_with_adapter(model_path, adapter_path):
    """Load base model and apply the saved LoRA adapter."""
    from mlx_lm import load

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    return model, tokenizer


def generate_sample(model, tokenizer, prompt, max_tokens=256):
    """Generate a completion for a prompt."""
    from mlx_lm import generate

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response


def compute_perplexity(model, tokenizer, eval_path, max_samples=100, max_seq_len=2048):
    """Compute perplexity over the eval set."""
    total_loss = 0.0
    total_tokens = 0
    count = 0

    with open(eval_path) as f:
        for line in f:
            if count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])
            if not messages:
                continue

            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                text = "\n".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                    for m in messages
                )

            tokens = tokenizer.encode(text)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            if len(tokens) < 2:
                continue

            input_ids = mx.array([tokens[:-1]])
            targets = mx.array([tokens[1:]])

            logits = model(input_ids)
            vocab_size = logits.shape[-1]
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)

            ce = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
            mx.eval(ce)

            total_loss += ce.sum().item()
            total_tokens += len(tokens) - 1
            count += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, count


def main():
    parser = argparse.ArgumentParser(
        description="Thunder-Train: Evaluate a trained LoRA adapter"
    )
    parser.add_argument("--model", required=True,
                        help="Base model path or HuggingFace ID")
    parser.add_argument("--adapter-path", required=True,
                        help="Path to Thunder-Train adapter directory")
    parser.add_argument("--eval-data", default=None,
                        help="Path to evaluation data (ChatML JSONL) for perplexity")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of sample completions to generate")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per generation")

    args = parser.parse_args()

    # Load checkpoint metadata
    ckpt_path = f"{args.adapter_path}/checkpoint.json"
    try:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        print(f"Adapter checkpoint: step={ckpt.get('step')}, "
              f"model={ckpt.get('model')}, "
              f"lora_rank={ckpt.get('lora_rank')}, "
              f"trained={ckpt.get('timestamp')}")
    except FileNotFoundError:
        print(f"No checkpoint.json found at {ckpt_path}, loading adapter weights only")

    # Load model + adapter
    print(f"Loading model: {args.model}")
    print(f"Loading adapter: {args.adapter_path}")
    model, tokenizer = load_model_with_adapter(args.model, args.adapter_path)
    print("Model loaded.\n")

    # Sample completions
    test_prompts = [
        "What should we prioritize for the next sprint?",
        "The build is failing on Mac4, what should I check?",
        "How should we handle the Supabase migration?",
        "The exo cluster keeps dropping Mac5, what's going on?",
        "Should we use data parallel or tensor parallel for 14B?",
    ]

    print("=" * 60)
    print("Sample Completions")
    print("=" * 60)
    for i, prompt in enumerate(test_prompts[:args.num_samples]):
        print(f"\n--- Prompt {i+1} ---")
        print(f"User: {prompt}")
        t0 = time.time()
        response = generate_sample(model, tokenizer, prompt, args.max_tokens)
        t1 = time.time()
        print(f"Assistant: {response}")
        print(f"({t1-t0:.1f}s)")

    # Perplexity
    if args.eval_data:
        print(f"\n{'='*60}")
        print(f"Perplexity Evaluation: {args.eval_data}")
        print(f"{'='*60}")
        t0 = time.time()
        avg_loss, ppl, n_samples = compute_perplexity(model, tokenizer, args.eval_data)
        t1 = time.time()
        print(f"Samples evaluated: {n_samples}")
        print(f"Average loss:      {avg_loss:.4f}")
        print(f"Perplexity:        {ppl:.2f}")
        print(f"Eval time:         {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
