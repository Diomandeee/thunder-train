#!/usr/bin/env python3
"""
thunder_status.py — Check readiness of the Thunder-Train cluster.

Pings Mac4 + Mac5, checks MLX versions, reports GPU memory,
and measures TB5 link latency.

Usage:
  python3 thunder_status.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HOSTS = [
    {"name": "Mac4", "ssh": "mac4", "ip": "10.0.5.1"},
    {"name": "Mac5", "ssh": "mac5", "ip": "10.0.5.2"},
]


def run_ssh(host_ssh, command, timeout=10):
    """Run a command over SSH and return (success, stdout)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
             host_ssh, command],
            capture_output=True, text=True, timeout=timeout + 5
        )
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception) as e:
        return False, str(e)


def ping_host(ip, count=3):
    """Ping a host and return (success, avg_latency_ms)."""
    try:
        result = subprocess.run(
            ["ping", "-c", str(count), "-W", "2000", ip],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return False, 0.0
        # Parse avg latency from "min/avg/max/stddev = ..."
        for line in result.stdout.splitlines():
            if "avg" in line and "=" in line:
                parts = line.split("=")[-1].strip().split("/")
                if len(parts) >= 2:
                    return True, float(parts[1])
        return True, 0.0
    except Exception:
        return False, 0.0


def check_mlx_version(host_ssh):
    """Check MLX version on remote host."""
    ok, output = run_ssh(
        host_ssh,
        "python3 -c \"import mlx.core as mx; print(mx.__version__)\""
    )
    return output if ok else "UNAVAILABLE"


def check_gpu_memory(host_ssh):
    """Check available GPU/unified memory on remote host."""
    # macOS reports unified memory via sysctl
    ok, output = run_ssh(
        host_ssh,
        "sysctl -n hw.memsize"
    )
    if ok:
        try:
            bytes_total = int(output)
            gb = bytes_total / (1024 ** 3)
            return f"{gb:.0f} GB"
        except ValueError:
            pass
    return "UNKNOWN"


def check_memory_pressure(host_ssh):
    """Check current memory pressure on remote host."""
    ok, output = run_ssh(
        host_ssh,
        "memory_pressure 2>/dev/null | head -1 || echo 'unavailable'"
    )
    return output if ok else "UNKNOWN"


def main():
    print("=" * 60)
    print("Thunder-Train Cluster Status")
    print("=" * 60)
    print()

    # Check each host
    all_ready = True
    for host in HOSTS:
        print(f"--- {host['name']} ({host['ip']}) ---")

        # Ping via TB interface
        reachable, latency = ping_host(host["ip"])
        if reachable:
            print(f"  TB Link:     OK ({latency:.1f}ms avg)")
        else:
            print(f"  TB Link:     UNREACHABLE")
            all_ready = False

        # SSH check
        ssh_ok, _ = run_ssh(host["ssh"], "echo ok")
        if ssh_ok:
            print(f"  SSH:         OK")
        else:
            print(f"  SSH:         FAILED")
            all_ready = False
            print()
            continue

        # MLX version
        mlx_ver = check_mlx_version(host["ssh"])
        ver_ok = mlx_ver != "UNAVAILABLE"
        if ver_ok:
            try:
                parts = mlx_ver.split(".")
                major, minor = int(parts[0]), int(parts[1])
                ver_ok = (major, minor) >= (0, 30)
            except (ValueError, IndexError):
                ver_ok = False
        status = "OK" if ver_ok else "NEEDS UPGRADE (require >= 0.30)"
        print(f"  MLX:         {mlx_ver} ({status})")
        if not ver_ok:
            all_ready = False

        # GPU memory
        mem = check_gpu_memory(host["ssh"])
        print(f"  Memory:      {mem} unified")

        # mlx_lm check
        ok, output = run_ssh(
            host["ssh"],
            "python3 -c \"import mlx_lm; print(mlx_lm.__version__)\" 2>/dev/null || echo 'NOT INSTALLED'"
        )
        mlx_lm_ver = output if ok else "NOT INSTALLED"
        print(f"  mlx_lm:      {mlx_lm_ver}")
        if "NOT INSTALLED" in mlx_lm_ver:
            all_ready = False

        print()

    # TB link quality (inter-node latency)
    print("--- Thunderbolt Link Quality ---")
    reachable_4, lat_4 = ping_host("10.0.5.1", count=10)
    reachable_5, lat_5 = ping_host("10.0.5.2", count=10)
    if reachable_4 and reachable_5:
        print(f"  Mac4 latency (10-ping avg): {lat_4:.2f}ms")
        print(f"  Mac5 latency (10-ping avg): {lat_5:.2f}ms")
        if lat_4 < 1.0 and lat_5 < 1.0:
            print(f"  Link quality: EXCELLENT (sub-millisecond)")
        elif lat_4 < 5.0 and lat_5 < 5.0:
            print(f"  Link quality: GOOD")
        else:
            print(f"  Link quality: DEGRADED (check TB cable)")
    else:
        print(f"  TB link: NOT AVAILABLE (machines offline or TB not connected)")
        all_ready = False

    # Hostfile check
    print()
    hostfile = Path(__file__).parent / "hostfile.json"
    if hostfile.exists():
        with open(hostfile) as f:
            hf = json.load(f)
        print(f"Hostfile:      {hostfile} ({len(hf)} hosts)")
    else:
        print(f"Hostfile:      MISSING (run from thunder-train directory)")
        all_ready = False

    # Summary
    print()
    print("=" * 60)
    if all_ready:
        print("CLUSTER READY for distributed training")
    else:
        print("CLUSTER NOT READY -- fix issues above before launching")
    print("=" * 60)


if __name__ == "__main__":
    main()
