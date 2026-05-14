"""Quick smoke test: simulator -> SNPE -> NUTS -> report."""
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from snpe_enzax.simulator import make_observed_data, TRUE_LOG_KCAT, KCAT_REACTIONS
from snpe_enzax.snpe_pipeline import run_snpe
from snpe_enzax.hmc_baseline import run_nuts
import numpy as np

_, x_obs = make_observed_data()

# ── SNPE smoke (1 round, 50 sims, 200 posterior samples) ──────────────────────
print("=== SNPE smoke test (1 round x 50 sims) ===")
snpe = run_snpe(x_obs, n_rounds=1, n_simulations_per_round=50, n_posterior_samples=200)
assert snpe["posterior_samples"].shape == (200, 10), f"bad shape {snpe['posterior_samples'].shape}"
assert np.all(np.isfinite(snpe["posterior_samples"])), "NaN in SNPE samples"
print(f"  posterior shape : {snpe['posterior_samples'].shape}")
print(f"  n_simulations   : {snpe['n_simulations']}")
print(f"  train_time_s    : {snpe['train_time_s']:.1f}s")
print(f"  log_prob_true   : {snpe['log_prob_true']:.2f}")
print(f"  posterior mean  : {snpe['posterior_samples'].mean(0).round(3)}")
print("SNPE smoke test PASSED\n")

# ── NUTS smoke (50 warmup, 100 samples) ───────────────────────────────────────
print("=== NUTS smoke test (50 warmup + 100 samples) ===")
nuts = run_nuts(x_obs, num_warmup=50, num_samples=100)
assert nuts["posterior_samples"].shape == (100, 10), f"bad shape {nuts['posterior_samples'].shape}"
assert np.all(np.isfinite(nuts["posterior_samples"])), "NaN in NUTS samples"
print(f"  posterior shape : {nuts['posterior_samples'].shape}")
print(f"  n_simulations   : {nuts['n_simulations']}")
print(f"  train_time_s    : {nuts['train_time_s']:.1f}s")
print(f"  divergences     : {nuts['divergences']}")
print(f"  posterior mean  : {nuts['posterior_samples'].mean(0).round(3)}")
print("NUTS smoke test PASSED\n")

print("All smoke tests PASSED")
