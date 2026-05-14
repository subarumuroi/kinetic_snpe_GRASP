"""HMC / NUTS baseline using blackjax.

Infers the same 10 log_kcat parameters as SNPE, using a Gaussian likelihood
on the simulated steady-state fluxes.  The log-posterior is:

    log p(theta | x_obs) ∝ log p(theta) + log p(x_obs | theta)

where
    log p(theta)       = Σ_i  Normal(theta_i; mu_i, sigma²)   (independent)
    log p(x_obs|theta) = Σ_j  Normal(x_obs_j; flux_j(theta), flux_err_j²)

This deliberately mirrors the SNPE setup so the comparison is fair:
same prior, same likelihood structure, same observations.
"""

import functools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from jax.scipy.stats import norm

from enzax.examples import methionine
from enzax.steady_state import get_steady_state

from .simulator import KCAT_REACTIONS, TRUE_LOG_KCAT, _FIXED_PARAMS, _DEFAULT_GUESS

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


# ── Log-prior ─────────────────────────────────────────────────────────────────

def _log_prior(log_kcat_vec: jnp.ndarray, prior_scale: float) -> jnp.ndarray:
    loc = jnp.array(TRUE_LOG_KCAT)
    return norm.logpdf(log_kcat_vec, loc, prior_scale).sum()


# ── Log-likelihood ─────────────────────────────────────────────────────────────

def _log_likelihood(
    log_kcat_vec: jnp.ndarray,
    x_obs: jnp.ndarray,
    flux_err: jnp.ndarray,
) -> jnp.ndarray:
    """Normal likelihood: observed ~ N(simulated_flux, flux_err²)."""
    params = {
        **_FIXED_PARAMS,
        "log_kcat": {r: log_kcat_vec[i] for i, r in enumerate(KCAT_REACTIONS)},
    }
    steady = get_steady_state(methionine.model, _DEFAULT_GUESS, params)
    flux_hat = methionine.model.flux(steady, params)
    return norm.logpdf(x_obs, flux_hat, flux_err).sum()


# ── Posterior factory ─────────────────────────────────────────────────────────

def make_log_density(
    x_obs: np.ndarray,
    noise_frac: float = 0.03,
    prior_scale: float = 0.5,
):
    """Build a JIT-compiled log-posterior for blackjax.

    Parameters
    ----------
    x_obs       : (11,) observed flux vector.
    noise_frac  : fractional observation noise  σ_j = noise_frac * |x_obs_j|.
    prior_scale : std of the independent normal prior over log_kcat.

    Returns
    -------
    log_density : callable (log_kcat_vec,) → scalar
    flux_err    : (11,) heteroscedastic error vector used in the likelihood
    """
    x_obs_j = jnp.array(x_obs)
    # Heteroscedastic Gaussian noise; small floor avoids division by zero near
    # zero-flux reactions.
    flux_err = jnp.abs(x_obs_j) * noise_frac + 1e-8

    @jax.jit
    def log_density(log_kcat_vec: jnp.ndarray) -> jnp.ndarray:
        lp = _log_prior(log_kcat_vec, prior_scale)
        ll = _log_likelihood(log_kcat_vec, x_obs_j, flux_err)
        return lp + ll

    return log_density, flux_err


# ── JIT-compiled NUTS inference loop ─────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=["kernel", "num_samples"])
def _inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_state, keys)
    return states, info


# ── Public API ────────────────────────────────────────────────────────────────

def run_nuts(
    x_obs: np.ndarray,
    num_warmup: int = 500,
    num_samples: int = 2000,
    prior_scale: float = 0.5,
    noise_frac: float = 0.03,
    initial_step_size: float = 0.001,
    target_acceptance_rate: float = 0.8,
    seed: int = 0,
) -> dict:
    """Run NUTS on the methionine cycle model.

    Parameters
    ----------
    x_obs                 : (11,) observed flux vector.
    num_warmup            : warmup steps for window adaptation.
    num_samples           : post-warmup samples to collect.
    prior_scale           : std of the independent normal prior.
    noise_frac            : fractional observation noise level.
    initial_step_size     : starting leapfrog step size for adaptation.
    target_acceptance_rate: dual-averaging target acceptance rate.
    seed                  : JAX random seed.

    Returns
    -------
    dict
        posterior_samples : (num_samples, 10) array
        n_simulations     : warmup + sampling steps (one grad eval each)
        train_time_s      : wall-clock seconds
        divergences       : number of divergent post-warmup transitions
    """
    log_density, _ = make_log_density(x_obs, noise_frac=noise_frac, prior_scale=prior_scale)
    init_params = jnp.array(TRUE_LOG_KCAT)

    rng = jax.random.key(seed)
    rng, warmup_key = jax.random.split(rng)

    log.info(f"NUTS warmup: {num_warmup} steps")
    t0 = time.perf_counter()

    # Window adaptation tunes the step size and (optionally) the mass matrix
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_density,
        progress_bar=False,   # requires fastprogress; keep off to avoid dep issues
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
    )
    (initial_state, tuned_params), _ = warmup.run(
        warmup_key, init_params, num_steps=num_warmup
    )

    log.info(f"NUTS sampling: {num_samples} steps")
    rng, sample_key = jax.random.split(rng)
    nuts_kernel = blackjax.nuts(log_density, **tuned_params).step
    states, info = _inference_loop(
        sample_key, nuts_kernel, initial_state, num_samples
    )
    jax.block_until_ready(states.position)
    train_time = time.perf_counter() - t0

    samples = np.asarray(states.position)           # (num_samples, 10)
    n_divergent = int(jnp.sum(info.is_divergent))
    if n_divergent:
        log.warning(f"NUTS: {n_divergent} divergent post-warmup transitions")

    return {
        "posterior_samples": samples,
        "n_simulations": num_warmup + num_samples,
        "train_time_s": train_time,
        "divergences": n_divergent,
    }
