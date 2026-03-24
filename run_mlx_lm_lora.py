#!/usr/bin/env python3
"""
Thin launcher to run mlx_lm lora via mlx._distributed_utils.launch.
mlx_lm's trainer has @mx.compile with distributed support built in.
mlx.launch sets MLX_RANK + MLX_HOSTFILE before running this script.
"""
from mlx_lm import lora as _lora
_lora.main()
