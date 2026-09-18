from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DeviceProfile:
    name: str
    arch: str
    ram_mb: int
    tflops: float
    power_watts: float
    energy_per_flop_joules: float  # Estimated J/FLOP
    measurement_type: str = "ESTIMATED"

PROFILES: Dict[str, DeviceProfile] = {
    "laptop_cpu": DeviceProfile(
        name="Laptop CPU (x86_64)",
        arch="x86_64",
        ram_mb=16384,
        tflops=0.1, # ~100 GFLOPS CPU peak
        power_watts=28.0,
        energy_per_flop_joules=2.8e-10
    ),
    "raspberry_pi_4": DeviceProfile(
        name="Raspberry Pi 4B",
        arch="ARM Cortex-A72",
        ram_mb=4096,
        tflops=0.0135, # ~13.5 GFLOPS
        power_watts=5.0,
        energy_per_flop_joules=3.7e-10
    ),
    "jetson_orin_nano": DeviceProfile(
        name="NVIDIA Jetson Orin Nano",
        arch="ARM Cortex-A78AE + Ampere GPU",
        ram_mb=8192,
        tflops=0.4, # FP32 TFLOPS
        power_watts=15.0,
        energy_per_flop_joules=3.75e-11
    ),
    "esp32": DeviceProfile(
        name="ESP32 Microcontroller",
        arch="Xtensa LX6",
        ram_mb=1,
        tflops=0.0006, # ~600 MFLOPS
        power_watts=0.5,
        energy_per_flop_joules=8.3e-10
    )
}

def estimate_resource_costs(
    model_num_params: int,
    n_samples: int,
    n_epochs: int,
    profile_name: str = "laptop_cpu"
) -> Dict[str, Any]:
    """
    Estimate computational FLOPs and energy consumption for local training on a specific device profile.
    Tags all outputs clearly as ESTIMATED.
    """
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown device profile: '{profile_name}'. Available: {list(PROFILES.keys())}")

    profile = PROFILES[profile_name]

    # Approximate forward + backward FLOPs per sample per parameter = ~6 * num_params
    flops_per_sample_step = 6 * model_num_params
    total_flops = flops_per_sample_step * n_samples * n_epochs

    estimated_seconds = total_flops / (profile.tflops * 1e12 + 1e-8)
    estimated_energy_joules = total_flops * profile.energy_per_flop_joules

    return {
        "device_name": profile.name,
        "arch": profile.arch,
        "measurement_tag": "[ESTIMATED]",
        "total_flops": total_flops,
        "estimated_compute_time_sec": round(estimated_seconds, 6),
        "estimated_energy_joules": round(estimated_energy_joules, 6),
        "memory_fit_ok": (model_num_params * 4 / (1024 * 1024)) <= profile.ram_mb
    }
