"""Enzax methionine cycle simulator wrapper.

Exposes a clean interface for both SNPE and HMC:
  theta : log_kcat values (10-dim), one per enzyme in KCAT_REACTIONS order
  x     : steady-state fluxes (11-dim), one per reaction in ALL_REACTIONS order

The 10 free parameters are the log catalytic-rate constants (log_kcat) for
MAT1, MAT3, METH-Gen, GNMT1, AHC1, MS1, BHMT1, CBS1, MTHFR1, PROT1.
All other model parameters (Km, Ki, enzyme concentrations, dgf, etc.) are
held fixed at the enzax default values.
"""

import numpy as np
import jax
import jax.numpy as jnp

from enzax.examples import methionine
from enzax.steady_state import get_steady_state

jax.config.update("jax_enable_x64", True)

# ── Parameter space definition ────────────────────────────────────────────────

# Ordered list of reactions that have a kcat parameter (excludes the_drain)
KCAT_REACTIONS: list[str] = list(methionine.parameters["log_kcat"].keys())
ALL_REACTIONS: list[str] = methionine.reactions

N_THETA: int = len(KCAT_REACTIONS)  # 10 free parameters
N_X: int = len(ALL_REACTIONS)       # 11 summary statistics (fluxes)

# Default parameter values in the inference space
TRUE_LOG_KCAT: np.ndarray = np.array(
    [float(methionine.parameters["log_kcat"][r]) for r in KCAT_REACTIONS]
)

# Everything that is NOT being inferred is fixed
_FIXED_PARAMS: dict = {
    k: v for k, v in methionine.parameters.items() if k != "log_kcat"
}

# Good initial guess for the ODE steady-state solver
_DEFAULT_GUESS: jnp.ndarray = jnp.array(methionine.steady_state)


# ── Core simulation ───────────────────────────────────────────────────────────

def _build_params(log_kcat_vec: jnp.ndarray) -> dict:
    """Merge fixed params with a new log_kcat vector into a full param dict."""
    return {
        **_FIXED_PARAMS,
        "log_kcat": {r: log_kcat_vec[i] for i, r in enumerate(KCAT_REACTIONS)},
    }


def simulate(theta: np.ndarray) -> np.ndarray:
    """Run the methionine cycle model and return steady-state fluxes.

    Parameters
    ----------
    theta : array_like of shape (10,)
        log_kcat values in KCAT_REACTIONS order.

    Returns
    -------
    x : ndarray of shape (11,)
        Steady-state reaction fluxes.  Returns a NaN-filled array when the
        ODE solver fails to find a steady state (e.g. for extreme parameters).
    """
    try:
        log_kcat_vec = jnp.asarray(theta, dtype=jnp.float64)
        params = _build_params(log_kcat_vec)
        steady = get_steady_state(methionine.model, _DEFAULT_GUESS, params)
        fluxes = methionine.model.flux(steady, params)
        result = np.asarray(fluxes)
        return result if np.all(np.isfinite(result)) else np.full(N_X, np.nan)
    except Exception:
        return np.full(N_X, np.nan)


# ── Synthetic data generation ─────────────────────────────────────────────────

def get_true_fluxes() -> np.ndarray:
    """Steady-state fluxes at the enzax default parameter values."""
    return simulate(TRUE_LOG_KCAT)


def make_observed_data(
    rng: np.random.Generator | None = None,
    noise_frac: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic observation by adding fractional Gaussian noise.

    Parameters
    ----------
    rng : numpy random generator (default: seed 42).
    noise_frac : fractional noise level, i.e. σ_i = noise_frac * |flux_i|.

    Returns
    -------
    true_theta : (10,) true log_kcat values.
    x_obs      : (11,) noisy observed fluxes.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    true_fluxes = get_true_fluxes()
    noise = rng.normal(0.0, noise_frac * np.abs(true_fluxes))
    return TRUE_LOG_KCAT.copy(), true_fluxes + noise
