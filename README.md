# thunder-train

Distributed LoRA fine-tuning across two Apple Silicon machines over a Thunderbolt 5 link, using [MLX](https://github.com/ml-explore/mlx).

Two Mac M4s pool their memory and compute into a single 32GB training node. A 7B model that runs at ~500 tok/s on one machine runs at ~900 tok/s across both.

---

## Hardware Setup

```
Mac4 (M4 Pro, 24GB)  <--[TB5 40Gb/s]-->  Mac5 (M4, 16GB)
     10.0.5.1                                  10.0.5.2
```

- **Transport**: TCP over Thunderbolt 5 IP link (static IPs, 0.26ms RTT)
- **Backend**: MLX ring distributed (`mlx._distributed_utils`)
- **Memory**: Each machine holds its own full model copy (data parallelism) or one shard (tensor parallelism)

## Why Thunderbolt instead of Ethernet?

TB5 gives ~40 Gb/s bandwidth with sub-millisecond latency. For gradient sync on a 7B model, this is effectively free — the bottleneck is Metal compute, not the network. Even 1 GbE would work for 7B; TB5 matters at 27B+.

> **Note on Apple JACCL (RDMA)**: Apple has RDMA-over-Thunderbolt code inside MLX (`jaccl` backend) but the library isn't publicly distributed. The ring TCP backend is sufficient for all models up to ~27B 4-bit. When JACCL ships publicly, switching backends will be a one-line change.

---

## Requirements

- Two Apple Silicon Macs on the same Thunderbolt network
- Python 3.12+ on both machines (MLX 0.30+ drops Python 3.9)
- MLX >= 0.30.5 and mlx-lm on both machines
- SSH key-based auth between machines (no passphrase)

```bash
# Install on both machines
pip install 'mlx>=0.30.5' mlx-lm

# Verify distributed launcher is present
python3 -c "import mlx._distributed_utils.launch; print('OK')"
```

### SSH Setup

```bash
# On Mac4: generate a key for automation
ssh-keygen -t ed25519 -f ~/.ssh/id_mac4_auto -N ""
cat ~/.ssh/id_mac4_auto.pub >> ~/.ssh/authorized_keys

# On Mac5: same
ssh-keygen -t ed25519 -f ~/.ssh/id_mac5_auto -N ""
cat ~/.ssh/id_mac5_auto.pub >> ~/.ssh/authorized_keys

# On the launcher machine: add to ~/.ssh/config
Host mac4
    HostName <mac4-ip>
    User <username>
    IdentityFile ~/.ssh/id_mac4_auto
    IdentitiesOnly yes

Host mac5
    HostName <mac5-ip>
    User <username>
    IdentityFile ~/.ssh/id_mac5_auto
    IdentitiesOnly yes
```

> **Gotcha**: If you get "Too many authentication failures", check that the `IdentityFile` in `~/.ssh/config` matches the key whose public half is in `authorized_keys` on the remote. SSH tries every loaded agent key before hitting your specified one, triggering the failure limit.

---

## Configuration

Edit `hostfile.json` with your machines' SSH aliases and Thunderbolt IPs:

```json
[
  {"ssh": "mac4", "ips": ["10.0.5.1"]},
  {"ssh": "mac5", "ips": ["10.0.5.2"]}
]
```

Use Thunderbolt IPs (not Tailscale, not Wi-Fi) for gradient sync. The TB link handles this automatically once the static IPs are configured in System Settings → Network.

---

## Usage

### Quick start

```bash
./launch.sh \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --train-data ~/data/train.jsonl \
  --valid-data ~/data/valid.jsonl \
  --strategy data \
  --num-iters 800 \
  --batch-size 4 \
  --num-layers 8 \
  --adapter-path ~/adapters/my-adapter \
  --learning-rate 1e-6
```

Watch for `Distributed group initialized: rank=0, size=2` in the output. If you see `size=1`, SSH isn't reaching the remote (check verbose SSH output with `--verbose` in `launch.sh`).

### Data format

ChatML JSONL — each line is a JSON object with a `messages` array:

```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | HuggingFace model ID or local path |
| `--train-data` | required | Training JSONL path |
| `--valid-data` | required | Validation JSONL path |
| `--strategy` | `data` | `data` (full model on each node) or `tensor` (model sharded) |
| `--num-iters` | 500 | Training steps |
| `--batch-size` | 4 | Per-node batch size (effective batch = batch × world_size) |
| `--num-layers` | 16 | Number of transformer layers to apply LoRA to |
| `--lora-rank` | 16 | LoRA rank |
| `--learning-rate` | 1e-5 | Peak LR (linear warmup + cosine decay) |
| `--adapter-path` | required | Where to save the LoRA adapter |
| `--save-every` | 100 | Checkpoint interval |
| `--valid-every` | 50 | Validation interval |
| `--max-seq-len` | 2048 | Max token length (longer sequences are truncated) |

> **LR tip**: Use `1e-6` for 4-bit quantized models. The default `1e-5` causes gradient explosion through dequantization layers. Gradient clipping (max_norm=1.0) is applied automatically, but a lower LR is safer.

---

## How it works

### Data parallelism (`--strategy data`)

Each node loads the full model. Data is sharded round-robin by rank. After each backward pass, `nn.average_gradients(grads)` all-reduces gradients across the ring. Both nodes update their weights identically, staying in sync.

```
Node 0: samples 0, 2, 4, ...  →  forward/backward  →  average_gradients()  →  update
Node 1: samples 1, 3, 5, ...  →  forward/backward  →  average_gradients()  →  update
```

Effective batch size = `batch_size × world_size`. Throughput scales near-linearly with node count.

### Tensor parallelism (`--strategy tensor`)

The model is sharded across nodes using MLX's `AllToSharded`/`ShardedToAll` layers. Each node holds half the attention heads and MLP weights. Communication happens inside the forward pass. Use this for models that don't fit on a single machine (14B+ at full precision, 27B+ at 4-bit).

### Ring backend

MLX's ring backend establishes TCP connections directly between nodes using the IPs in `hostfile.json`. The launcher SSH's into each node, sets `MLX_RANK` and `MLX_HOSTFILE` environment variables, and starts the training script. No coordinator process — each node connects to the next in the ring.

---

## Verify the ring is working

```bash
# test_ring.py
python3 mlx_launch.py \
  --hostfile hostfile.json \
  --backend ring \
  -- test_ring.py
```

Expected output (order varies):
```
rank=0 size=2
rank=1 size=2
rank=0 all_sum OK: [2]
```

If `size=1` on both, SSH connectivity is broken. Run with `--verbose` to see SSH output.

---

## Rust rewrite?

Short answer: not yet, and probably not needed.

MLX is written in C++ with Python bindings. The Metal shader kernels (matrix multiply, quantized matmul, attention) are where ~95% of training time is spent — those are already as fast as they can be. The Python orchestration overhead (data loading, checkpoint saving, grad sync dispatch) is a few percent of total time. Rewriting the orchestration layer in Rust would save maybe 2-3%.

The one place where Rust would matter is the ring communication layer — if you're doing very frequent all-reduces at large batch sizes, a zero-copy Rust implementation of the ring protocol could help. That's effectively what JACCL (when it ships) will provide, but at the RDMA level rather than TCP.

If you want to contribute: the highest-leverage improvements are:
1. Async gradient compression (quantize grads to fp8 before all-reduce)
2. Pipeline parallelism for 3+ nodes
3. A proper JACCL backend once Apple publishes the library

---

## Project structure

```
thunder-train/
├── thunder_train.py      # Main training script (distributed LoRA)
├── thunder_eval.py       # Standalone eval on a trained adapter
├── thunder_status.py     # Check training status / adapter quality
├── launch.sh             # Wrapper: checks deps, launches distributed job
├── mlx_launch.py         # mlx._distributed_utils.launch wrapper (stdout fix)
├── hostfile.json         # Node topology (SSH aliases + Thunderbolt IPs)
└── distributed_launch.sh # Legacy — kept for reference, do not use
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `world_size=1` | SSH not reaching remote | Check `ssh mac4 "echo ok"` works |
| `Too many auth failures` | Wrong SSH key in `~/.ssh/config` | Verify `IdentityFile` matches remote `authorized_keys` |
| `loss=nan` | LR too high for 4-bit quant | Use `--learning-rate 1e-6` |
| `AttributeError: mx.utils.tree_flatten` | MLX < 0.31 | Upgrade to 0.31+ or use `from mlx.utils import tree_flatten` |
| `ValueError: warmup_iters=0` | Very few training iters | Use at least `--num-iters 10` |
| No output from launcher | stdout not primed | `mlx_launch.py` fixes this — don't use `-m mlx._distributed_utils.launch` directly |
| Mac5 stuck on MLX 0.29 | Python 3.9 system install | Install Python 3.12+ (e.g. via uv) and `pip install mlx` into that env |

---

## License

MIT
