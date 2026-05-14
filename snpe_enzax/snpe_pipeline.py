"""SNPE pipeline using the sbi library (SNPE-C / APT).

Sequential Neural Posterior Estimation learns p(theta | x_obs) by training a
normalizing flow on (theta, x) pairs from the simulator.  In later rounds the
proposal distribution is focused toward the posterior, so each simulation is
more informative than blind prior sampling.

theta : log_kcat (10-dim)
x_obs : steady-state fluxes (11-dim)
"""

import time
import logging

import numpy as np
import torch
from torch.distributions import Independent, Normal

from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator

from .simulator import simulate, TRUE_LOG_KCAT, N_THETA

log = logging.getLogger(__name__)


# ── Prior ─────────────────────────────────────────────────────────────────────

def make_prior(
    loc: np.ndarray | None = None,
    scale: float = 0.5,
) -> Independent:
    """Independent normal prior over log_kcat.

    Parameters
    ----------
    loc   : prior means; defaults to the enzax true values.
    scale : standard deviation in log-kcat space (0.5 ≈ ±1.6× uncertainty).
    """
    if loc is None:
        loc = TRUE_LOG_KCAT
    return Independent(
        Normal(
            loc=torch.tensor(loc, dtype=torch.float32),
            scale=torch.full((N_THETA,), scale, dtype=torch.float32),
        ),
        reinterpreted_batch_ndims=1,
    )


# ── SNPE training loop ────────────────────────────────────────────────────────

def run_snpe(
    x_obs: np.ndarray,
    n_rounds: int = 3,
    n_simulations_per_round: int = 1000,
    prior_scale: float = 0.5,
    n_posterior_samples: int = 2000,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Train SNPE-C on the methionine cycle and sample the posterior.

    Parameters
    ----------
    x_obs                    : (11,) observed flux vector.
    n_rounds                 : sequential training rounds.
    n_simulations_per_round  : simulator calls per round.
    prior_scale              : std of the independent normal prior.
    n_posterior_samples      : samples drawn from the trained posterior.
    device                   : torch device ('cpu' or 'cuda').
    seed                     : random seed for reproducibility.

    Returns
    -------
    dict
        posterior_samples : (n_posterior_samples, 10) array
        n_simulations     : total number of simulator calls
        train_time_s      : wall-clock seconds (simulation + training)
        log_prob_true     : log p(theta_true | x_obs) under the trained flow
    """
    torch.manual_seed(seed)
    x_obs_t = torch.tensor(x_obs, dtype=torch.float32).to(device)

    # sbi wants the prior in a specific wrapper; process_prior handles that
    raw_prior = make_prior(scale=prior_scale)
    prior_sbi, _, _ = process_prior(raw_prior)

    # process_simulator wraps the numpy simulator for sbi compatibility
    sim_sbi = process_simulator(simulate, prior_sbi, is_numpy_simulator=True)

    inference = SNPE(prior=prior_sbi, device=device)
    proposal = prior_sbi   # first round: sample from the prior
    density_estimator = None
    total_sims = 0

    t0 = time.perf_counter()

    for rnd in range(n_rounds):
        log.info(f"SNPE round {rnd + 1}/{n_rounds}: drawing {n_simulations_per_round} simulations")
        theta, x = simulate_for_sbi(
            simulator=sim_sbi,
            proposal=proposal,
            num_simulations=n_simulations_per_round,
            seed=seed + rnd,
            show_progress_bar=True,
        )
        total_sims += n_simulations_per_round

        # For round > 0, pass the previous posterior as proposal so that
        # SNPE-C applies the correct importance-weight correction.
        inference.append_simulations(
            theta, x,
            proposal=None if rnd == 0 else proposal,
        )
        density_estimator = inference.train(show_train_summary=False)

        if rnd < n_rounds - 1:
            posterior = inference.build_posterior(density_estimator)
            posterior.set_default_x(x_obs_t)
            proposal = posterior   # next round: sample from refined posterior

    train_time = time.perf_counter() - t0

    posterior = inference.build_posterior(density_estimator)
    posterior.set_default_x(x_obs_t)

    samples = posterior.sample(
        (n_posterior_samples,), x=x_obs_t, show_progress_bars=False
    ).cpu().numpy()

    true_theta_t = torch.tensor(TRUE_LOG_KCAT, dtype=torch.float32).to(device)
    log_prob = posterior.log_prob(true_theta_t, x=x_obs_t).item()

    return {
        "posterior_samples": samples,     # (n_posterior_samples, 10)
        "n_simulations": total_sims,
        "train_time_s": train_time,
        "log_prob_true": log_prob,
    }
