#!/usr/bin/env python3
"""
Edge cost table (paper Table 6) COMPUTED from the model that was actually trained.

  python scripts/profile_edge.py --from-run results/runs/main/main__ton_iot__fedavg_iid__s0.json
  python scripts/profile_edge.py --n-features 31 --hidden 64,32 --measure-host

Latency = FLOPs / (peak FLOP/s x eta) and energy = batch x latency x TDP are ANALYTICAL bounds (eta in [0.01, 0.10]);
they are reported as ranges and must be called estimates.  --measure-host adds a wall-clock (and RAPL, if readable)
measurement on THIS machine plus the eta it implies.  Replace the Raspberry Pi / laptop assumptions with cited numbers
through --device-file (a JSON list of {key, name, tdp_w, peak_gflops, source}).
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys

from fedids_bench.evaluation.edge_profile import (
    DEFAULT_DEVICES, EdgeDevice, build_profile, implied_eta, measure_host, mlp_dims, write_csv,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-run", default=None, help="a run JSON from scripts/run_grid.py: takes n_features and hidden_dims from it")
    ap.add_argument("--n-features", type=int, default=31)
    ap.add_argument("--hidden", default="64,32")
    ap.add_argument("--n-classes", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eta-best", type=float, default=0.10)
    ap.add_argument("--eta-worst", type=float, default=0.01)
    ap.add_argument("--device-file", default=None)
    ap.add_argument("--measure-host", action="store_true")
    ap.add_argument("--out", default=os.path.join("results", "edge_profile.csv"))
    args = ap.parse_args()

    n_features, hidden = args.n_features, [int(h) for h in args.hidden.split(",") if h.strip()]
    if args.from_run:
        with open(args.from_run, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        hidden = d["overrides"]["model"]["hidden_dims"]
        n_features = int(d["result"]["dataset_summary"]["n_features"])
        print(f"model taken from {os.path.basename(args.from_run)}: {n_features} features, hidden {hidden}")
    dims = mlp_dims(n_features, hidden, args.n_classes)

    devices = DEFAULT_DEVICES
    if args.device_file:
        with open(args.device_file, "r", encoding="utf-8") as fh:
            devices = tuple(EdgeDevice(**x) for x in json.load(fh))

    rows = build_profile(dims, devices, args.batch_size, args.eta_best, args.eta_worst)
    if args.measure_host:
        m = measure_host(dims, args.batch_size)
        m["implied_eta_on_rpi4"] = implied_eta(m["flops_per_sample"], m["latency_ms_per_sample_measured"], devices[0].peak_gflops)
        rows.append(m)
    write_csv(args.out, rows)

    print(f"\nmodel {'-'.join(map(str, dims))}: {rows[0]['n_params']:,} parameters, {rows[0]['param_bytes']:,} bytes, "
          f"{rows[0]['flops_per_sample']:,} FLOPs/sample, {rows[0]['traffic_bytes_per_client_round']:,} bytes per client per round")
    for r in rows:
        if r["kind"] == "analytical":
            print(f"  {r['device_key']:22s} latency {r['latency_ms_per_sample_best']:.5f}..{r['latency_ms_per_sample_worst']:.5f} ms/sample"
                  f"   energy {r['energy_mj_per_batch_best']:.4f}..{r['energy_mj_per_batch_worst']:.4f} mJ/batch  [estimate]")
        else:
            e = r.get("energy_mj_per_batch_measured_package")
            print(f"  {r['device_key']:22s} latency {r['latency_ms_per_sample_measured']:.5f} ms/sample (measured, numpy)"
                  f"   package energy {'n/a' if e is None else f'{e:.4f} mJ/batch'}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
