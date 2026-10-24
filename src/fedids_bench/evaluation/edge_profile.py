"""
Analytical edge-cost model for FedIDS-Bench MLPs.

Everything reported in the paper's edge table is COMPUTED here from the model's layer sizes and a small
device table -- no value is typed in.  Two kinds of numbers are produced and kept apart:

  * ANALYTICAL (all devices):   latency  = FLOPs / (F_peak * eta)
                                energy   = batch_size * latency * TDP          (an upper bound: full TDP)
    eta (effective efficiency) is unknown for tiny MLPs, so a RANGE is reported
    (eta_best = 0.10 compute-bound, eta_worst = 0.01 memory/overhead-bound) instead of one invented point.

  * MEASURED (host machine only, optional):  wall-clock latency of a NumPy forward pass, and package
    energy from Intel RAPL when /sys/class/powercap is readable (Linux).  The implied host eta is
    reported so the analytical range can be sanity-checked against a real device.

Energy of the Pi 4 / Jetson is therefore *estimated*, never "measured"; say so in the paper.
"""
from __future__ import annotations

import csv
import os
import platform
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class EdgeDevice:
    key: str
    name: str
    tdp_w: float          # power used for the energy upper bound
    peak_gflops: float    # FP32 throughput used for the latency bound
    source: str           # provenance of the two numbers (must be citable in the paper)


DEFAULT_DEVICES: Sequence[EdgeDevice] = (
    EdgeDevice(
        "rpi4", "Raspberry Pi 4 (Cortex-A72, 4 cores @ 1.5 GHz)", tdp_w=5.0, peak_gflops=13.5,
        source="ASSUMPTION - Raspberry Pi Ltd publishes no TDP/FLOP rating; cite your own measurement or state as assumed",
    ),
    EdgeDevice(
        "jetson_orin_nano_8gb", "NVIDIA Jetson Orin Nano 8GB (1024 CUDA cores)", tdp_w=15.0, peak_gflops=1280.0,
        source="NVIDIA module datasheet: 1024 CUDA cores, GPU max 625 MHz -> 1024*2*0.625 GHz = 1.28 TFLOP/s FP32; 7 W / 15 W modes",
    ),
    EdgeDevice(
        "x86_laptop", "x86 laptop CPU (replace with your exact model)", tdp_w=45.0, peak_gflops=220.0,
        source="ASSUMPTION - fill in the datasheet TDP and cores*GHz*FLOPs/cycle of your exact CPU",
    ),
)


# ----------------------------------------------------------------------------- model arithmetic
def mlp_dims(n_features: int, hidden_dims: Sequence[int], n_classes: int) -> List[int]:
    return [int(n_features), *[int(h) for h in hidden_dims], int(n_classes)]


def count_params(dims: Sequence[int]) -> int:
    return int(sum(i * o + o for i, o in zip(dims[:-1], dims[1:])))


def count_flops(dims: Sequence[int]) -> int:
    """Forward FLOPs for ONE sample: 2*in*out (multiply+add) + out (bias) per Linear, + out per hidden ReLU."""
    total = 0
    n_layers = len(dims) - 1
    for li, (i, o) in enumerate(zip(dims[:-1], dims[1:])):
        total += 2 * i * o + o
        if li < n_layers - 1:
            total += o
    return int(total)


# ----------------------------------------------------------------------------- analytical estimate
def estimate(device: EdgeDevice, flops: int, batch_size: int, eta: float) -> Dict[str, float]:
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must be in (0, 1]")
    latency_s = flops / (device.peak_gflops * 1e9 * eta)
    return {
        "latency_ms_per_sample": latency_s * 1e3,
        "energy_mj_per_batch": batch_size * latency_s * device.tdp_w * 1e3,
    }


def build_profile(
    dims: Sequence[int],
    devices: Iterable[EdgeDevice] = DEFAULT_DEVICES,
    batch_size: int = 32,
    eta_best: float = 0.10,
    eta_worst: float = 0.01,
) -> List[Dict[str, object]]:
    flops = count_flops(dims)
    n_params = count_params(dims)
    param_bytes = 4 * n_params                        # float32
    rows: List[Dict[str, object]] = []
    for d in devices:
        best = estimate(d, flops, batch_size, eta_best)
        worst = estimate(d, flops, batch_size, eta_worst)
        rows.append({
            "device": d.name, "device_key": d.key, "tdp_w": d.tdp_w, "peak_gflops": d.peak_gflops,
            "spec_source": d.source, "model_dims": "-".join(map(str, dims)), "n_params": n_params,
            "param_bytes": param_bytes, "flops_per_sample": flops, "batch_size": batch_size,
            "eta_best": eta_best, "eta_worst": eta_worst,
            "latency_ms_per_sample_best": best["latency_ms_per_sample"],
            "latency_ms_per_sample_worst": worst["latency_ms_per_sample"],
            "energy_mj_per_batch_best": best["energy_mj_per_batch"],
            "energy_mj_per_batch_worst": worst["energy_mj_per_batch"],
            # FL traffic per CLIENT per ROUND: the model goes down once and comes back up once.
            "upload_bytes_per_client_round": param_bytes,
            "download_bytes_per_client_round": param_bytes,
            "traffic_bytes_per_client_round": 2 * param_bytes,
            "kind": "analytical",
        })
    return rows


# ----------------------------------------------------------------------------- host measurement
def _numpy_forward(x: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray:
    for li, (w, b) in enumerate(zip(weights, biases)):
        x = x @ w + b
        if li < len(weights) - 1:
            np.maximum(x, 0.0, out=x)
    return x


def read_rapl_energy_uj() -> Optional[int]:
    """Package energy counter (microjoules) from Intel RAPL, if readable (Linux, often needs root)."""
    for path in ("/sys/class/powercap/intel-rapl:0/energy_uj", "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return None


def measure_host(dims: Sequence[int], batch_size: int = 32, repeats: int = 3000, seed: int = 0) -> Dict[str, object]:
    """Wall-clock (and, if available, RAPL) cost of a NumPy forward pass on THIS machine."""
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal((i, o)).astype(np.float32) * 0.1 for i, o in zip(dims[:-1], dims[1:])]
    biases = [np.zeros(o, dtype=np.float32) for o in dims[1:]]
    x = rng.standard_normal((batch_size, dims[0])).astype(np.float32)
    for _ in range(200):                                   # warm-up
        _numpy_forward(x.copy(), weights, biases)
    e0 = read_rapl_energy_uj()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _numpy_forward(x.copy(), weights, biases)
    elapsed = time.perf_counter() - t0
    e1 = read_rapl_energy_uj()
    lat_ms_sample = elapsed / repeats / batch_size * 1e3
    out: Dict[str, object] = {
        "device": f"HOST: {platform.processor() or platform.machine()} ({platform.system()})",
        "device_key": "host_measured", "model_dims": "-".join(map(str, dims)),
        "n_params": count_params(dims), "flops_per_sample": count_flops(dims), "batch_size": batch_size,
        "latency_ms_per_sample_measured": lat_ms_sample, "kind": "measured",
        "backend": "numpy",
    }
    if e0 is not None and e1 is not None and e1 >= e0:
        out["energy_mj_per_batch_measured_package"] = (e1 - e0) / 1e3 / repeats     # uJ -> mJ per batch
    else:
        out["energy_mj_per_batch_measured_package"] = None
    return out


def implied_eta(flops: int, latency_ms_per_sample: float, peak_gflops: float) -> float:
    """eta that would reproduce a measured latency on a device with the given peak."""
    return flops / (peak_gflops * 1e9 * (latency_ms_per_sample / 1e3))


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
