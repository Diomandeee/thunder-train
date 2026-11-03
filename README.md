# thunder-train

Distributed LoRA fine-tuning across two Apple Silicon Macs over a Thunderbolt cable, using [MLX](https://github.com/ml-explore/mlx).

Two code paths:
- **Data parallel:** each Mac holds a full model copy and trains on half the data, with gradients averaged every step.
- **Tensor sharding:** each Mac holds half of each layer.

Data parallelism doesn't pool memory: the model still has to fit on one machine. Tensor sharding is how a model bigger than one Mac's memory could train, and how far that goes depends on the model and quantization.

---

## Setup it was built on

```
node-a (Apple M4, 16 GB)  <--[Thunderbolt bridge, 40 Gb/s ports]-->  node-b (Apple M4, 16 GB)
       10.0.5.1                                                              10.0.5.2
```

- **Transport:** TCP over a Thunderbolt bridge with static IPs
- **Backend:** MLX ring (`mlx._distributed_utils`)

## Why Thunderbolt?

Gradient sync is a burst of large transfers every step. A Thunderbolt bridge gives far more bandwidth and far lower latency than Wi-Fi, with nothing else on the link. `thunder_status.py` measures the latency on your own cable.

> **Apple JACCL (RDMA):** MLX has an RDMA-over-Thunderbolt backend (`jaccl`) that wasn't publicly available when this was written. The ring TCP backend is used instead. Switching backends should be a small change once it is.

---

## Requirements

- Two Apple Silicon Macs joined by a Thunderbolt cable, with static IPs on the Thunderbolt bridge
- Python 3.12+ on both (MLX 0.30+ dropped Python 3.9)
- MLX >= 0.30.5 and mlx-lm on both
- Key-based SSH between the machines, no passphrase

```bash
# On both machines
pip install 'mlx>=0.30.5' mlx-lm

# Check the distributed launcher is there
python3 -c "import mlx._distributed_utils.launch; print('OK')"
```

### SSH

```bash
# On each node: a key for automation
ssh-keygen -t ed25519 -f ~/.ssh/id_thunder -N ""
cat ~/.ssh/id_thunder.pub >> ~/.ssh/authorized_keys

# On the launcher machine, in ~/.ssh/config
Host node-a
    HostName <node-a-ip>
    User <username>
    IdentityFile ~/.ssh/id_thunder
    IdentitiesOnly yes

Host node-b
    HostName <node-b-ip>
    User <username>
    IdentityFile ~/.ssh/id_thunder
    IdentitiesOnly yes
```

> **"Too many authentication failures":** SSH tries every key loaded in the agent before the one you named. Set `IdentitiesOnly yes` and make sure the `IdentityFile` matches the key in the remote `authorized_keys`.

---

## Configuration

`hostfile.json` lists the nodes by SSH alias and Thunderbolt IP. `launch.sh` and `thunder_status.py` both read it.

```json
[
  {"ssh": "node-a", "ips": ["10.0.5.1"]},
  {"ssh": "node-b", "ips": ["10.0.5.2"]}
]
```

Use the Thunderbolt IPs, not Wi-Fi or a VPN address, so gradient sync goes over the cable.

---

## Usage

```bash
python3 thunder_status.py      # link, SSH, MLX and memory on every node

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

Look for `Distributed group initialized: rank=0, size=2`. If it says `size=1`, SSH isn't reaching the other node. `launch.sh` runs the MLX launcher with `--verbose`, so the SSH errors are in its output.

### Data format

ChatML JSONL, one object with a `messages` array per line:

```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Hugging Face model ID or local path |
| `--train-data` | required | Training JSONL |
| `--valid-data` | none | Validation JSONL |
| `--strategy` | `data` | `data` (full model on each node) or `tensor` (model sharded) |
| `--num-iters` | 800 | Training steps |
| `--batch-size` | 4 | Per-node batch (effective batch = batch × nodes) |
| `--num-layers` | 8 | Transformer layers that get LoRA, counted from the end |
| `--lora-rank` | 16 | LoRA rank |
| `--learning-rate` | 1e-5 | Peak LR |
| `--max-seq-len` | 2048 | Longer samples are truncated |
| `--adapter-path` | none | Where the adapter is saved |
| `--resume` | off | Resume from the checkpoint in `--adapter-path` |
| `--log-every` | 10 | Log interval |
| `--eval-every` | 100 | Validation interval |
| `--save-every` | 200 | Checkpoint interval |
| `--seed` | 42 | Random seed |

> **Learning rate:** use `1e-6` for 4-bit models. The default `1e-5` blew up gradients through the dequantization layers. Gradients are clipped at norm 1.0 either way.

---

## How it works

### Data parallel (`--strategy data`)

Each node loads the full model and takes every other sample by rank. After each backward pass, `nn.average_gradients(grads)` all-reduces the gradients around the ring, so both nodes apply the same update and stay in sync.

```
node 0: samples 0, 2, 4, ...  ->  forward/backward  ->  average_gradients()  ->  update
node 1: samples 1, 3, 5, ...  ->  forward/backward  ->  average_gradients()  ->  update
```

### Tensor sharding (`--strategy tensor`)

The model is split across nodes with MLX's `AllToSharded` / `ShardedToAll` layers. Each node holds half the attention heads and MLP weights, and the nodes talk inside the forward pass.

### Ring backend

The launcher SSHes into each node, sets `MLX_RANK` and `MLX_HOSTFILE`, and starts the training script. Each node opens a TCP connection to the next one in the ring. There's no coordinator process.

---

## Check the ring

`mlx_launch.py` wraps the MLX launcher and fixes its buffered stdout. Run any small script through it to check both ranks come up:

```bash
python3 mlx_launch.py --hostfile hostfile.json --backend ring -- your_check.py
```

Inside the script, `mx.distributed.init()` should report `size=2` on both ranks, and an `all_sum` of `1` should come back as `2`.

---

## Files

```
thunder_train.py      distributed LoRA training
thunder_eval.py       evaluate a trained adapter
thunder_status.py     check every node in hostfile.json
launch.sh             dependency checks, then the distributed launch
mlx_launch.py         MLX launcher wrapper (stdout fix)
run_mlx_lm_lora.py    run mlx_lm's own LoRA trainer through the same launcher
hostfile.json         nodes: SSH aliases and Thunderbolt IPs
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `size=1` | SSH not reaching the other node | Check `ssh node-b "echo ok"` |
| `Too many authentication failures` | Wrong key in `~/.ssh/config` | Match `IdentityFile` to the remote `authorized_keys` |
| `loss=nan` | LR too high for 4-bit | `--learning-rate 1e-6` |
| `AttributeError: mx.utils.tree_flatten` | MLX < 0.31 | Upgrade, or `from mlx.utils import tree_flatten` |
| `ValueError: warmup_iters=0` | Too few iterations | At least `--num-iters 10` |
| No launcher output | stdout buffering | Use `mlx_launch.py`, not `-m mlx._distributed_utils.launch` |
| A node stuck on MLX 0.29 | System Python 3.9 | Install Python 3.12+ (for example with uv) and MLX into that |

---

## License

MIT
