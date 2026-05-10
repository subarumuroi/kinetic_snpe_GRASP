"""
GRASP + SNPE inference pipeline.
Replaces the ABC rejection sampler in Saa & Nielsen (2016) with
Sequential Neural Posterior Estimation (SNPE-C / APT).

Reference data taken directly from Saa & Nielsen (2016) Figure 2a/2c.
"""

import torch
import numpy as np
from sbi.inference import SNPE
from sbi import utils as sbi_utils
from sbi.utils import BoxUniform
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference data from Saa & Nielsen (2016)
# ---------------------------------------------------------------------------

# Reaction names in order - Figure 2a
REACTION_NAMES = [
    "v_INFLUX", "v_PROT", "v_MATI", "v_MATIII",
    "v_METH",   "v_GNMT", "v_AHC",  "v_MS",
    "v_BHMT",   "v_CBS",  "v_MTHFR"
]

N_REACTIONS = len(REACTION_NAMES)

# Reference flux distribution (mmol/L-cells/h) - Figure 2a
V_REF = torch.tensor([
    0.76,  # v_INFLUX
    0.14,  # v_PROT
    0.81,  # v_MATI
    0.25,  # v_MATIII
    1.02,  # v_METH
    0.05,  # v_GNMT
    1.06,  # v_AHC
    0.36,  # v_MS
    0.09,  # v_BHMT
    0.62,  # v_CBS
    0.09,  # v_MTHFR
], dtype=torch.float32)

# 12 training perturbations - Figure 2c
# Log-fold changes relative to v_ref
# Rows = perturbations, Cols = reactions (same order as above)
# These are read from the heatmap in Figure 2c
PERTURBATION_LABELS = [
    "50% up-regulation CBS",
    "50% increase v_INFLUX",
    "BHMT knockout",
    "80% down-regulation MTHFR",
    "2-fold up-regulation MS",
    "2-fold up-regulation AHC",
    "50% up-regulation v_PROT",
    "50% down-regulation MATI",
    "2-fold up-regulation GNMT",
    "2-fold up-regulation MATIII",
    "30% down-regulation METH",
    "30% down-regulation CBS",
]

# Enzyme level multipliers for each perturbation
# Shape: (12 perturbations, 11 reactions)
# These encode the perturbation magnitude applied to each enzyme
PERTURBATION_ENZYME_LEVELS = torch.tensor([
    # INFLUX PROT  MATI  MATIII METH  GNMT  AHC   MS    BHMT  CBS   MTHFR
    [1.0,   1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.5,  1.0  ],  # 1
    [1.5,   1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0  ],  # 2
    [1.0,   1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.0,  1.0,  1.0  ],  # 3
    [1.0,   1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.2  ],  # 4
    [1.0,   1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  2.0,  1.0,  1.0,  1.0  ],  # 5
    [1.0,   1.0,  1.0,  1.0,  1.0,  1.0,  2.0,  1.0,  1.0,  1.0,  1.0  ],  # 6
    [1.5,   1.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0  ],  # 7
    [1.0,   1.0,  0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0  ],  # 8
    [1.0,   1.0,  1.0,  1.0,  1.0,  2.0,  1.0,  1.0,  1.0,  1.0,  1.0  ],  # 9
    [1.0,   1.0,  1.0,  2.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0  ],  # 10
    [1.0,   1.0,  1.0,  1.0,  0.7,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0  ],  # 11
    [1.0,   1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.7,  1.0  ],  # 12
], dtype=torch.float32)

# ABC tolerance from the paper (equation 3)
ABC_TOLERANCE = 0.2


# ---------------------------------------------------------------------------
# MATLAB interface - plug in your GRASP code here
# ---------------------------------------------------------------------------

class GRASPSimulator:
    """
    Wraps the MATLAB GRASP implementation.

    When you get to work:
    1. Confirm fieldnames(ensemble) and fieldnames(ensemble.models(1))
    2. Fill in _extract_theta and _extract_fluxes accordingly
    3. Set use_mock=False
    """

    def __init__(self, grasp_path: str = None, use_mock: bool = True):
        self.use_mock = use_mock
        self.grasp_path = grasp_path
        self.eng = None

        if not use_mock:
            self._init_matlab()

    def _init_matlab(self):
        try:
            import matlab.engine
            logger.info("Starting MATLAB engine...")
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(self.grasp_path)
            logger.info("MATLAB engine ready.")
        except ImportError:
            raise RuntimeError(
                "matlab.engine not found. "
                "Install with: cd <matlabroot>/extern/engines/python && pip install ."
            )

    def sample_prior(self, n_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw samples from the GRASP prior.

        Returns
        -------
        theta : np.ndarray, shape (n_samples, n_params)
            Auxiliary parameters (e_tilde, R, r_elem) per sample.
            These are what SNPE learns to estimate.
        v_sim : np.ndarray, shape (n_samples, n_reactions)
            Simulated flux vector at reference state per sample.
        """
        if self.use_mock:
            return self._mock_sample_prior(n_samples)

        # --- Real MATLAB call ---
        # Replace with actual GRASP function call once you've confirmed
        # the output struct field names at work.
        #
        # Something like:
        #   ensemble = self.eng.sampleEnsemble(model, params, n_samples)
        #   theta = self._extract_theta(ensemble)
        #   v_sim = self._extract_fluxes(ensemble)
        #
        raise NotImplementedError("Fill in MATLAB interface at work.")

    def simulate_perturbed(
        self,
        theta: np.ndarray,
        enzyme_levels: np.ndarray
    ) -> np.ndarray:
        """
        Given a parameter set theta, simulate the perturbed steady state.

        Parameters
        ----------
        theta : np.ndarray, shape (n_params,)
        enzyme_levels : np.ndarray, shape (n_reactions,)
            Enzyme level multipliers (1.0 = reference level).

        Returns
        -------
        v_sim : np.ndarray, shape (n_reactions,)
        """
        if self.use_mock:
            return self._mock_simulate_perturbed(theta, enzyme_levels)

        # Real MATLAB call:
        # self.eng.simulatePerturbed(theta, enzyme_levels, ...)
        raise NotImplementedError("Fill in MATLAB interface at work.")

    # ------------------------------------------------------------------
    # Mock implementations - let you develop/test the Python side now
    # ------------------------------------------------------------------

    def _mock_sample_prior(self, n_samples: int):
        """
        Crude mock that returns plausible-looking samples.
        Real GRASP samples Dirichlet(1) for e_tilde and constrained
        reversibilities for R - this just gives you something to run with.
        """
        # Placeholder parameter dimensionality
        # Real dimension depends on number of enzyme intermediates
        # across all 9 methionine cycle enzymes - roughly ~72 params
        n_params = 72

        rng = np.random.default_rng(42)

        # Rough mock of Dirichlet-like samples (bounded 0-1)
        theta = rng.dirichlet(np.ones(n_params), size=n_samples)

        # Rough mock of flux output - perturb v_ref with noise
        v_ref = V_REF.numpy()
        noise = rng.normal(0, 0.1 * v_ref, size=(n_samples, N_REACTIONS))
        v_sim = np.clip(v_ref + noise, 0, None)

        return theta, v_sim

    def _mock_simulate_perturbed(
        self,
        theta: np.ndarray,
        enzyme_levels: np.ndarray
    ) -> np.ndarray:
        """Very crude mock - just scales v_ref by enzyme levels."""
        v_ref = V_REF.numpy()
        v_sim = v_ref * enzyme_levels
        noise = np.random.normal(0, 0.05 * v_ref)
        return np.clip(v_sim + noise, 0, None)

    # ------------------------------------------------------------------
    # Helpers to fill in once you know the GRASP struct field names
    # ------------------------------------------------------------------

    def _extract_theta(self, ensemble) -> np.ndarray:
        """
        Extract auxiliary parameters from GRASP ensemble struct.
        Fill this in at work once you've run fieldnames(ensemble).
        """
        # e.g. return np.array(ensemble.models[0].eAux)
        raise NotImplementedError

    def _extract_fluxes(self, ensemble) -> np.ndarray:
        """
        Extract simulated flux vectors from GRASP ensemble struct.
        Fill this in at work once you've run fieldnames(ensemble).
        """
        # e.g. return np.array(ensemble.simFluxes)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Distance function matching the paper (equation 3)
# ---------------------------------------------------------------------------

def abc_distance(v_sim: torch.Tensor, v_exp: torch.Tensor) -> torch.Tensor:
    """
    Weighted infinite-norm distance from Saa & Nielsen (2016) equation 3.

    max_i |v_sim_i - v_exp_i| / (max(v_ref) - min(v_ref))
    """
    v_ref_range = V_REF.max() - V_REF.min()
    return torch.max(torch.abs(v_sim - v_exp) / v_ref_range, dim=-1).values


# ---------------------------------------------------------------------------
# SNPE pipeline
# ---------------------------------------------------------------------------

class GRASPPosteriorEstimator:
    """
    Replaces the ABC rejection sampler with SNPE-C.

    Workflow:
    1. Draw (theta, v_sim) pairs from GRASP prior
    2. Train a normalizing flow to learn p(theta | v_sim)
    3. At test time, condition on experimental flux observations
    4. Sample from the learned posterior
    """

    def __init__(
        self,
        simulator: GRASPSimulator,
        n_params: int = 72,
        device: str = "cpu"
    ):
        self.simulator = simulator
        self.n_params = n_params
        self.device = device

        # Prior over auxiliary parameters
        # Real prior is Dirichlet(1) on simplices - BoxUniform is an
        # approximation that lets us get started. Refine once running.
        self.prior = BoxUniform(
            low=torch.zeros(n_params),
            high=torch.ones(n_params)
        )

        self.inference = SNPE(prior=self.prior, device=device)
        self.posterior = None

    def collect_simulations(self, n_simulations: int = 1000):
        """Draw samples from GRASP and store for training."""
        logger.info(f"Collecting {n_simulations} simulations from GRASP prior...")

        theta_np, v_sim_np = self.simulator.sample_prior(n_simulations)

        theta = torch.tensor(theta_np, dtype=torch.float32)
        v_sim = torch.tensor(v_sim_np, dtype=torch.float32)

        logger.info(f"theta shape: {theta.shape}, v_sim shape: {v_sim.shape}")
        return theta, v_sim

    def train(
        self,
        n_rounds: int = 3,
        n_simulations_per_round: int = 1000,
        x_obs: torch.Tensor = None
    ):
        """
        Sequential training rounds.

        After round 1 the proposal shifts toward the posterior,
        making each subsequent round more efficient - this is
        the core gain over rejection sampling.
        """
        if x_obs is None:
            x_obs = V_REF  # start with reference state

        x_obs = x_obs.to(self.device)

        for round_idx in range(n_rounds):
            logger.info(f"Round {round_idx + 1}/{n_rounds}")

            theta, x = self.collect_simulations(n_simulations_per_round)
            self.inference.append_simulations(theta, x)
            self.posterior = self.inference.train()

            logger.info(f"Round {round_idx + 1} training complete.")

        return self.posterior

    def sample_posterior(
        self,
        x_obs: torch.Tensor,
        n_samples: int = 1000
    ) -> torch.Tensor:
        """Draw samples from the posterior given observed fluxes."""
        if self.posterior is None:
            raise RuntimeError("Call train() first.")

        return self.posterior.sample(
            (n_samples,),
            x=x_obs.to(self.device)
        )

    def benchmark_vs_rejection(
        self,
        x_obs: torch.Tensor,
        n_prior_samples: int = 10000
    ) -> dict:
        """
        Compare SNPE posterior against naive ABC rejection sampler.
        Reproduces the acceptance rate analysis from Figure 3b.
        """
        logger.info("Running rejection sampler benchmark...")

        theta_np, v_sim_np = self.simulator.sample_prior(n_prior_samples)
        v_sim = torch.tensor(v_sim_np, dtype=torch.float32)

        distances = abc_distance(v_sim, x_obs.unsqueeze(0).expand_as(v_sim))
        accepted_mask = distances <= ABC_TOLERANCE
        acceptance_rate = accepted_mask.float().mean().item()

        n_accepted = accepted_mask.sum().item()
        logger.info(
            f"Rejection sampler: {n_accepted}/{n_prior_samples} accepted "
            f"({acceptance_rate:.3%})"
        )

        return {
            "n_prior_samples": n_prior_samples,
            "n_accepted": n_accepted,
            "acceptance_rate": acceptance_rate,
            "theta_accepted": theta_np[accepted_mask.numpy()],
            "v_sim_accepted": v_sim_np[accepted_mask.numpy()],
        }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_flux_posterior(
    posterior_samples: torch.Tensor,
    simulator: GRASPSimulator,
    x_obs: torch.Tensor,
    save_path: str = None
):
    """
    Reproduce Figure 4 style control structure comparison.
    """
    # Simulate fluxes for posterior samples
    fluxes = []
    for theta in posterior_samples[:200].numpy():
        v = simulator.simulate_perturbed(theta, np.ones(N_REACTIONS))
        fluxes.append(v)

    fluxes = np.array(fluxes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Posterior predictive flux distributions
    axes[0].boxplot(fluxes, labels=REACTION_NAMES, vert=True)
    axes[0].plot(
        range(1, N_REACTIONS + 1),
        x_obs.numpy(),
        'ro', label='Observed', zorder=5
    )
    axes[0].set_xticklabels(REACTION_NAMES, rotation=45, ha='right')
    axes[0].set_ylabel("Flux (mmol/L-cells/h)")
    axes[0].set_title("Posterior predictive fluxes")
    axes[0].legend()

    # Acceptance rate comparison placeholder
    axes[1].text(
        0.5, 0.5,
        "Acceptance rate comparison\n(fill in after MATLAB runs)",
        ha='center', va='center', transform=axes[1].transAxes
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Run with mock simulator for now
    # At work: set use_mock=False, grasp_path="/path/to/GRASP"
    simulator = GRASPSimulator(use_mock=True)

    estimator = GRASPPosteriorEstimator(simulator=simulator)

    # Train on reference state first (simplest case)
    posterior = estimator.train(
        n_rounds=3,
        n_simulations_per_round=500,
        x_obs=V_REF
    )

    # Sample from posterior
    posterior_samples = estimator.sample_posterior(V_REF, n_samples=1000)
    logger.info(f"Posterior samples shape: {posterior_samples.shape}")

    # Benchmark against rejection sampler
    rejection_results = estimator.benchmark_vs_rejection(V_REF, n_prior_samples=5000)

    # Plot
    fig = plot_flux_posterior(
        posterior_samples,
        simulator,
        V_REF,
        save_path="posterior_fluxes.png"
    )

    return posterior, posterior_samples, rejection_results


if __name__ == "__main__":
    main()