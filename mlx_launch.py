#!/usr/bin/env python3
"""
Wrapper for mlx._distributed_utils.launch.main() that primes stdout
to avoid output buffering issues when invoked from non-interactive shells.
"""
import sys

# Prime stdout/stderr to ensure output appears in non-interactive shells
# (avoids buffering issue with -m invocation of mlx._distributed_utils.launch)
sys.stdout.write("")
sys.stdout.flush()
sys.stderr.write("")
sys.stderr.flush()

import mlx._distributed_utils.launch as _launch
_launch.main()
