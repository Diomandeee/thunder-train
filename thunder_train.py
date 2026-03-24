#!/usr/bin/env python3
"""
thunder_train.py — Distributed LoRA training across Mac4 + Mac5 via Thunderbolt.

Uses MLX native distributed primitives (mx.distributed, nn.average_gradients)
with the ring backend over the TB link (10.0.5.x subnet).

Supports two parallelism strategies:
  - data: Full model on each machine, gradients averaged (2x throughput for 7B-14B)
  - tensor: Model sharded across machines via AllToSharded/ShardedToAll layers (14B-27B)

Data format: ChatML JSONL ({"messages": [{"role": ..., "content": ...}, ...]})
Launch: mlx.launch --hostfile hostfile.json --backend ring -- python3 thunder_train.py [args]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.nn.layers.distributed import shard_inplace
from mlx.utils import tree_flatten


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rank():
    return mx.distributed.init().rank()

def world_size():
    return mx.distributed.init().size()

def is_main():
    return rank() == 0

def log(msg, force=False):
    if is_main() or force:
        print(msg, flush=True)

def log_step(step, loss_val, tokens_per_sec, lr, elapsed):
    log(f"[step {step:>5d}]  loss={loss_val:.4f}  tok/s={tokens_per_sec:.0f}  lr={lr:.2e}  elapsed={elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Data loading — ChatML JSONL
# ---------------------------------------------------------------------------

class ChatMLDataset:
    """Loads ChatML JSONL, tokenizes with the model's tokenizer, and shards by rank."""

    def __init__(self, path, tokenizer, max_seq_len=2048):
        self.samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                messages = obj.get("messages", [])
                if not messages:
                    continue
                # Apply chat template if available, otherwise concatenate
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
                if len(tokens) >= 2:
                    self.samples.append(tokens)

        # Sort by length for efficient batching
        self.samples.sort(key=len)
        log(f"Loaded {len(self.samples)} samples from {path}")

    def shard_for_rank(self):
        """Return only this rank's portion of the data (for data parallelism)."""
        r, ws = rank(), world_size()
        self.samples = self.samples[r::ws]
        log(f"Rank {r}: {len(self.samples)} samples after sharding", force=True)

    def iterate(self, batch_size):
        """Yield batches of (input_ids, targets) as padded MLX arrays."""
        # Shuffle deterministically per epoch (all ranks see different data if sharded)
        import random
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        batch_tokens = []
        for idx in indices:
            batch_tokens.append(self.samples[idx])
            if len(batch_tokens) == batch_size:
                yield self._collate(batch_tokens)
                batch_tokens = []
        if batch_tokens:
            yield self._collate(batch_tokens)

    def _collate(self, batch_tokens):
        """Pad and create input/target pairs."""
        max_len = max(len(t) for t in batch_tokens)
        input_ids = []
        targets = []
        for tokens in batch_tokens:
            padded = tokens + [0] * (max_len - len(tokens))
            input_ids.append(padded[:-1])
            targets.append(padded[1:])
        return mx.array(input_ids), mx.array(targets)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer via mlx_lm."""
    from mlx_lm import load
    model, tokenizer = load(model_path)
    return model, tokenizer


def apply_lora(model, num_layers, lora_rank=16):
    """Apply LoRA adapters using mlx_lm's built-in conversion."""
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    lora_config = {
        "rank": lora_rank,
        "alpha": lora_rank,
        "dropout": 0.0,
        "scale": 1.0,
    }

    linear_to_lora_layers(model, num_layers, lora_config)
    # linear_to_lora_layers handles freeze/unfreeze internally

    # Count trainable params
    from mlx_lm.tuner.utils import get_total_parameters
    total = get_total_parameters(model)
    from mlx.utils import tree_flatten
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    log(f"LoRA applied: {trainable:,} trainable / {total:,} total ({trainable/total*100:.3f}%)")


def apply_tensor_parallel(model):
    """Shard model layers across the distributed group for tensor parallelism."""
    group = mx.distributed.init()
    if group.size() <= 1:
        return

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Cannot find transformer layers in model")

    for layer in layers:
        attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attention

        # Shard Q, K, V projections: all-to-sharded (split output dim)
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            if hasattr(attn, proj_name):
                shard_inplace(getattr(attn, proj_name), "all-to-sharded", group=group)

        # O projection: sharded-to-all (gather input dim)
        if hasattr(attn, "o_proj"):
            shard_inplace(attn.o_proj, "sharded-to-all", group=group)

        # Scale head counts for sharded attention
        if hasattr(attn, "n_heads"):
            attn.n_heads //= group.size()
        if hasattr(attn, "n_kv_heads"):
            attn.n_kv_heads //= group.size()

        # FFN: gate + up are all-to-sharded, down is sharded-to-all
        mlp = layer.mlp if hasattr(layer, "mlp") else layer.feed_forward
        if hasattr(mlp, "gate_proj"):
            shard_inplace(mlp.gate_proj, "all-to-sharded", group=group)
        if hasattr(mlp, "up_proj"):
            shard_inplace(mlp.up_proj, "all-to-sharded", group=group)
        if hasattr(mlp, "down_proj"):
            shard_inplace(mlp.down_proj, "sharded-to-all", group=group)

    log(f"Tensor parallelism applied across {group.size()} devices")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(model, inputs, targets):
    """Cross-entropy loss over next-token predictions."""
    logits = model(inputs)
    # logits shape: (batch, seq_len, vocab_size)
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Mask padding (target == 0)
    mask = (targets_flat != 0).astype(mx.float32)
    n_tokens = mx.maximum(mask.sum(), mx.array(1.0))

    # Shift targets to valid class indices (avoid 0 as target in cross_entropy)
    # cross_entropy treats all targets including 0 as valid, so clip to be safe
    ce = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = (ce * mask).sum() / n_tokens
    return loss


def train(args):
    """Main training entrypoint."""
    group = mx.distributed.init()
    log(f"Distributed group initialized: rank={group.rank()}, size={group.size()}")

    # Load model + tokenizer
    log(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    log(f"Model loaded")

    # Apply strategy
    if args.strategy == "tensor" and group.size() > 1:
        apply_tensor_parallel(model)

    # Apply LoRA
    apply_lora(model, args.num_layers, args.lora_rank)

    # Load data
    log(f"Loading training data: {args.train_data}")
    train_dataset = ChatMLDataset(args.train_data, tokenizer, max_seq_len=args.max_seq_len)

    if args.strategy == "data" and group.size() > 1:
        train_dataset.shard_for_rank()

    valid_dataset = None
    if args.valid_data and os.path.exists(args.valid_data):
        log(f"Loading validation data: {args.valid_data}")
        valid_dataset = ChatMLDataset(args.valid_data, tokenizer, max_seq_len=args.max_seq_len)

    # Optimizer — AdamW with cosine schedule
    warmup_iters = max(1, min(100, args.num_iters // 10))
    warmup = opt.linear_schedule(1e-7, args.learning_rate, warmup_iters)
    cosine = opt.cosine_decay(args.learning_rate, args.num_iters - warmup_iters, 1e-7)
    lr_schedule = opt.join_schedules([warmup, cosine], [warmup_iters])
    optimizer = opt.AdamW(learning_rate=lr_schedule)

    # Compile loss + grad function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Resume from checkpoint if exists
    start_step = 0
    if args.resume and args.adapter_path and os.path.exists(args.adapter_path):
        ckpt_file = os.path.join(args.adapter_path, "checkpoint.json")
        if os.path.exists(ckpt_file):
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            start_step = ckpt.get("step", 0)
            log(f"Resuming from step {start_step}")
            # Load adapter weights
            adapter_weights = os.path.join(args.adapter_path, "adapters.safetensors")
            if os.path.exists(adapter_weights):
                model.load_weights(adapter_weights, strict=False)
                log(f"Loaded adapter weights from {adapter_weights}")

    # Training loop
    log(f"\n{'='*60}")
    log(f"Thunder-Train: Distributed LoRA Fine-Tuning")
    log(f"{'='*60}")
    log(f"Model:      {args.model}")
    log(f"Strategy:   {args.strategy} (world_size={group.size()})")
    log(f"LoRA:       rank={args.lora_rank}, layers={args.num_layers}")
    log(f"Iterations: {args.num_iters} (batch_size={args.batch_size})")
    log(f"LR:         {args.learning_rate} (warmup={warmup_iters})")
    log(f"Adapter:    {args.adapter_path}")
    log(f"{'='*60}\n")

    step = start_step
    epoch = 0
    best_val_loss = float("inf")
    start_time = time.time()

    while step < args.num_iters:
        epoch += 1
        for inputs, targets in train_dataset.iterate(args.batch_size):
            if step >= args.num_iters:
                break

            t0 = time.time()

            # Forward + backward
            loss, grads = loss_and_grad_fn(model, inputs, targets)

            # Sync gradients across machines (ring all-reduce)
            if group.size() > 1:
                grads = nn.average_gradients(grads, group=group)

            # Clip gradients to prevent NaN/explosion (esp. with 4-bit quant models)
            grads, _ = opt.clip_grad_norm(grads, max_norm=1.0)

            # Optimizer step
            optimizer.update(model, grads)

            # Evaluate to materialize the computation graph
            mx.eval(model.parameters(), optimizer.state, loss)

            t1 = time.time()
            step += 1

            # Logging
            if step % args.log_every == 0:
                tokens_in_batch = inputs.size
                tok_per_sec = tokens_in_batch / (t1 - t0)
                current_lr = lr_schedule(step) if callable(lr_schedule) else args.learning_rate
                log_step(step, loss.item(), tok_per_sec, current_lr, time.time() - start_time)

            # Validation
            if valid_dataset and step % args.eval_every == 0:
                val_loss = evaluate(model, valid_dataset, args.batch_size)
                log(f"[step {step:>5d}]  val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if is_main() and args.adapter_path:
                        save_checkpoint(model, optimizer, step, args, best=True)

            # Periodic checkpoint
            if is_main() and args.adapter_path and step % args.save_every == 0:
                save_checkpoint(model, optimizer, step, args)

    # Final save
    total_time = time.time() - start_time
    log(f"\nTraining complete: {step} steps in {total_time:.1f}s ({total_time/60:.1f}min)")

    if is_main() and args.adapter_path:
        save_checkpoint(model, optimizer, step, args, final=True)
        log(f"Final adapter saved to {args.adapter_path}")

    if valid_dataset:
        final_val = evaluate(model, valid_dataset, args.batch_size)
        log(f"Final validation loss: {final_val:.4f}")


def evaluate(model, dataset, batch_size):
    """Compute average loss over a dataset."""
    total_loss = 0.0
    total_batches = 0
    for inputs, targets in dataset.iterate(batch_size):
        loss = loss_fn(model, inputs, targets)
        mx.eval(loss)
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def save_checkpoint(model, optimizer, step, args, best=False, final=False):
    """Save adapter weights and training state (rank 0 only)."""
    out_dir = Path(args.adapter_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter weights (only LoRA params, not frozen weights)
    trainable = {}
    for k, v in dict(tree_flatten(model.trainable_parameters())).items():
        trainable[k] = v
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), trainable)

    # Save checkpoint metadata
    ckpt = {
        "step": step,
        "model": args.model,
        "strategy": args.strategy,
        "lora_rank": args.lora_rank,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "checkpoint.json", "w") as f:
        json.dump(ckpt, f, indent=2)

    # Save adapter config for mlx_lm compatibility
    adapter_config = {
        "lora_layers": args.num_layers,
        "lora_parameters": {"rank": args.lora_rank, "scale": 1.0, "dropout": 0.0},
        "model": args.model,
    }
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    label = "best" if best else ("final" if final else f"step-{step}")
    log(f"Checkpoint saved ({label}): {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Thunder-Train: Distributed LoRA fine-tuning over Thunderbolt"
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit)")
    parser.add_argument("--train-data", required=True,
                        help="Path to training data (ChatML JSONL)")
    parser.add_argument("--valid-data", default=None,
                        help="Path to validation data (ChatML JSONL)")
    parser.add_argument("--strategy", choices=["data", "tensor"], default="data",
                        help="Parallelism strategy: data (replicated) or tensor (sharded)")
    parser.add_argument("--num-iters", type=int, default=800,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Peak learning rate")
    parser.add_argument("--num-layers", type=int, default=8,
                        help="Number of transformer layers to apply LoRA to (from the end)")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Maximum sequence length (truncates longer samples)")
    parser.add_argument("--adapter-path", default=None,
                        help="Directory to save adapter weights")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint in adapter-path")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=200,
                        help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set seed
    mx.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # Default adapter path
    if args.adapter_path is None:
        model_name = args.model.split("/")[-1]
        args.adapter_path = f"./thunder-adapter-{model_name}"
        log(f"No adapter path specified, using: {args.adapter_path}")

    train(args)


if __name__ == "__main__":
    main()
