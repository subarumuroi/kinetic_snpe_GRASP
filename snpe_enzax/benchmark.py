"""Benchmark SNPE vs NUTS on the enzax methionine cycle.

Comparison axes
---------------
1. Posterior quality  — posterior mean and std per parameter; MAE vs truth
2. Simulator calls    — how many times get_steady_state is called in total
3. Wall-clock time    — seconds from start to posterior samples

Usage
-----
    python -m snpe_enzax.benchmark                   # default settings
    python -m snpe_enzax.benchmark --snpe-rounds 2 --snpe-sims 500
    python -m snpe_enzax.benchmark --nuts-only       # skip SNPE
    python -m snpe_enzax.benchmark --snpe-only       # skip NUTS
"""

import argparse
import logging
import sys

import numpy as np

from .simulator import make_observed_data, TRUE_LOG_KCAT, KCAT_REACTIONS
from .snpe_pipeline import run_snpe
from .hmc_baseline import run_nuts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _posterior_stats(samples: np.ndarray) -> list[dict]:
    """Per-parameter mean, std, and 95 % CI from posterior samples."""
    stats = []
    for i in range(samples.shape[1]):
        col = samples[:, i]
        stats.append({
            "mean": float(col.mean()),
            "std":  float(col.std()),
            "q025": float(np.quantile(col, 0.025)),
            "q975": float(np.quantile(col, 0.975)),
        })
    return stats


def _print_result(title: str, result: dict, param_names: list[str]) -> list[dict]:
    stats = _posterior_stats(result["posterior_samples"])
    divs  = result.get("divergences", "n/a")

    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")
    print(f"  Simulator calls : {result['n_simulations']:,}")
    print(f"  Wall-clock time : {result['train_time_s']:.1f} s")
    if isinstance(divs, int):
        print(f"  Divergences     : {divs}")
    if "log_prob_true" in result:
        print(f"  log p(θ*|x_obs) : {result['log_prob_true']:.2f}")

    print(f"\n  {'Reaction':<12}  {'mean':>8}  {'std':>7}  {'95% CI':>18}  {'true':>8}  {'|err|':>7}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*18}  {'-'*8}  {'-'*7}")
    for i, name in enumerate(param_names):
        s = stats[i]
        true = float(TRUE_LOG_KCAT[i])
        print(
            f"  {name:<12}  {s['mean']:+8.3f}  {s['std']:7.3f}"
            f"  [{s['q025']:+7.3f}, {s['q975']:+7.3f}]"
            f"  {true:+8.3f}  {abs(s['mean'] - true):7.3f}"
        )
    return stats


def _mae(stats: list[dict], param_names: list[str]) -> float:
    errs = [abs(stats[i]["mean"] - float(TRUE_LOG_KCAT[i])) for i in range(len(param_names))]
    return float(np.mean(errs))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_benchmark(
    n_snpe_rounds: int = 3,
    n_snpe_sims_per_round: int = 1000,
    num_warmup: int = 500,
    num_samples: int = 2000,
    run_snpe_flag: bool = True,
    run_nuts_flag: bool = True,
) -> dict:
    """Run the full benchmark and print a formatted report.

    Returns the raw result dicts for further analysis.
    """
    rng = np.random.default_rng(42)
    true_theta, x_obs = make_observed_data(rng=rng)

    print("\n" + "═" * 65)
    print("  enzax methionine cycle — SNPE vs NUTS benchmark")
    print("═" * 65)
    print(f"\n  Free parameters : {len(KCAT_REACTIONS)} log_kcat values")
    print(f"  Observations    : {len(x_obs)} steady-state fluxes (3% noise)")
    print(f"\n  Observed fluxes (x_obs):")
    for i, r in enumerate(["the_drain"] + KCAT_REACTIONS):
        print(f"    {r:<12}  {x_obs[i]:+.6f}")

    results: dict[str, dict] = {}

    if run_snpe_flag:
        print(f"\n[SNPE]  {n_snpe_rounds} rounds × {n_snpe_sims_per_round} simulations each")
        results["snpe"] = run_snpe(
            x_obs,
            n_rounds=n_snpe_rounds,
            n_simulations_per_round=n_snpe_sims_per_round,
        )

    if run_nuts_flag:
        print(f"\n[NUTS]  {num_warmup} warmup + {num_samples} samples")
        results["nuts"] = run_nuts(
            x_obs,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )

    # ── Per-method report ────────────────────────────────────────────────────

    all_stats: dict[str, list[dict]] = {}
    if "snpe" in results:
        all_stats["SNPE"] = _print_result("SNPE posterior", results["snpe"], KCAT_REACTIONS)
    if "nuts" in results:
        all_stats["NUTS"] = _print_result("NUTS posterior", results["nuts"], KCAT_REACTIONS)

    # ── Head-to-head summary ─────────────────────────────────────────────────

    if len(results) == 2:
        snpe_r = results["snpe"]
        nuts_r  = results["nuts"]

        snpe_mae = _mae(all_stats["SNPE"], KCAT_REACTIONS)
        nuts_mae  = _mae(all_stats["NUTS"],  KCAT_REACTIONS)

        print(f"\n{'═' * 65}")
        print("  Head-to-head summary")
        print(f"{'═' * 65}")
        print(f"  {'Metric':<30}  {'SNPE':>12}  {'NUTS':>12}")
        print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
        print(f"  {'Simulator calls':<30}  {snpe_r['n_simulations']:>12,}  {nuts_r['n_simulations']:>12,}")
        print(f"  {'Wall-clock time (s)':<30}  {snpe_r['train_time_s']:>12.1f}  {nuts_r['train_time_s']:>12.1f}")
        print(f"  {'Posterior mean MAE vs truth':<30}  {snpe_mae:>12.4f}  {nuts_mae:>12.4f}")
        if "divergences" in nuts_r:
            print(f"  {'NUTS divergences':<30}  {'—':>12}  {nuts_r['divergences']:>12}")

        print(f"\n  Note: SNPE simulator calls are the training budget; NUTS")
        print(f"  calls are warmup + sampling steps (each requires one")
        print(f"  gradient evaluation of the ODE steady-state solver).")
        print(f"\n  SNPE amortisation: once trained, posterior inference for")
        print(f"  new x_obs is instant — no further simulator calls needed.")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SNPE vs NUTS benchmark on the enzax methionine cycle"
    )
    p.add_argument("--snpe-rounds", type=int, default=3,
                   help="Number of SNPE sequential rounds (default: 3)")
    p.add_argument("--snpe-sims",   type=int, default=1000,
                   help="Simulations per SNPE round (default: 1000)")
    p.add_argument("--nuts-warmup", type=int, default=500,
                   help="NUTS warmup steps (default: 500)")
    p.add_argument("--nuts-samples", type=int, default=2000,
                   help="NUTS post-warmup samples (default: 2000)")
    p.add_argument("--snpe-only", action="store_true",
                   help="Run SNPE only (skip NUTS)")
    p.add_argument("--nuts-only", action="store_true",
                   help="Run NUTS only (skip SNPE)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        n_snpe_rounds=args.snpe_rounds,
        n_snpe_sims_per_round=args.snpe_sims,
        num_warmup=args.nuts_warmup,
        num_samples=args.nuts_samples,
        run_snpe_flag=not args.nuts_only,
        run_nuts_flag=not args.snpe_only,
    )
