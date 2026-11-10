"""Snapshot of every configuration value (configs/*.yaml + fixed constants in code) -> results/run_config.json.
Lets audit.py trace methodological numbers quoted in the manuscript."""
import os, sys, json
import yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from entropy import DEFAULTS  # noqa: E402

cfg = {n: yaml.safe_load(open(os.path.join(ROOT, "configs", f"{n}.yaml"))) for n in
       ["model", "reference", "benchmark", "sparse", "pinn"]}
cfg["entropy_defaults"] = DEFAULTS
cfg["val_block_offset_ms"] = 47.0          # sparse.val_blocks
cfg["spike_threshold_mV"] = -10.0          # metrics.crossings
cfg["peak_match_window_ms"] = 20.0         # metrics.metrics
cfg["odelsq"] = dict(w0_starts=[0.02, 0.1, 0.2, 0.35], n_starts=5, max_nfev=60, rtol_fit=1e-7, atol_fit=[1e-7, 1e-9],
                     bounds_V=[-90.0, 60.0], bounds_w=[0.0, 1.0])
cfg["pinn_derived"] = dict(n_col_per_segment=70, domain_ms=70.0, n_segments_per_current=20, lr_final=3e-4,
                           ema_alpha=0.9, lbfgs_history=50, adam_total_iter_dataonly=2400, w_scale_interface=0.1,
                           V_res_scale_div=10.0, n_samples_window=10001, n_samples_entropy=1001)
json.dump(cfg, open(os.path.join(ROOT, "results", "run_config.json"), "w"), indent=1)
print("run_config.json written")
