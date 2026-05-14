# SNPE vs NUTS on the enzax Methionine Cycle

## What this project does

This code compares two methods for **Bayesian parameter inference** on a
kinetic model of the mammalian methionine cycle:

| Method | Type | Library |
|--------|------|---------|
| **SNPE-C** | Sequential Neural Posterior Estimation | `sbi` + PyTorch |
| **NUTS** | No-U-Turn Sampler (a variant of HMC) | `blackjax` + JAX |

The **kinetic model** (provided by enzax, untouched) simulates how 10 enzymes
drive metabolite concentrations to a steady state.  We observe the 11
steady-state reaction fluxes and ask: *given these fluxes, what are the most
likely enzyme catalytic rates (log_kcat)?*

---

## Quick-start: how to run this yourself

### Step 1 — Open a terminal

In **VS Code** (which you already use): press `` Ctrl+` `` (the backtick key,
top-left of keyboard).  A terminal panel opens at the bottom.

### Step 2 — Navigate to the project

```bash
cd C:\Users\uqkmuroi\gitcode\kinetic_snpe_GRASP
```

### Step 3 — Create a virtual environment (Python 3.12 required)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` appear at the start of your prompt.

### Step 4 — Install everything

```bash
pip install -e enzax\          # installs enzax + JAX + blackjax from local clone
pip install -r requirements.txt
```

### Step 5 — Quick sanity check (~5 minutes)

```bash
python smoke_test.py
```

Expected output ends with:
```
SNPE smoke test PASSED
NUTS smoke test PASSED
All smoke tests PASSED
```

### Step 6 — Run the full benchmark

```bash
# Full run: ~10 min SNPE + ~60 min NUTS on CPU
python -m snpe_enzax.benchmark

# Faster version for exploration (less accurate posteriors)
python -m snpe_enzax.benchmark --snpe-rounds 2 --snpe-sims 500 --nuts-warmup 200 --nuts-samples 500

# Just one method at a time
python -m snpe_enzax.benchmark --snpe-only
python -m snpe_enzax.benchmark --nuts-only
```

---

## Project file map

```
kinetic_snpe_GRASP/
│
├── enzax/                      ← enzax library (DO NOT MODIFY)
│   └── src/enzax/
│       ├── examples/methionine.py    model + default parameters
│       ├── steady_state.py           ODE solver
│       ├── mcmc.py                   enzax's own NUTS helper
│       └── statistical_modelling.py  prior/likelihood functions
│
├── snpe_enzax/                 ← our inference layer (this project)
│   ├── simulator.py            wraps enzax → simulate(theta) → fluxes
│   ├── snpe_pipeline.py        SNPE-C training loop
│   ├── hmc_baseline.py         NUTS with window adaptation
│   └── benchmark.py            comparison script + CLI
│
├── smoke_test.py               fast sanity check (< 5 min)
├── requirements.txt            all Python dependencies
└── README_snpe.md              this file
```

---

## The scientific question

### What are we inferring?

**Free parameters (θ):** 10 log-catalytic-rate constants (`log_kcat`), one per
enzyme:

```
MAT1, MAT3, METH-Gen, GNMT1, AHC1, MS1, BHMT1, CBS1, MTHFR1, PROT1
```

In biological terms: *how fast can each enzyme process its substrate, per
enzyme molecule?*  We work in log-space because catalytic rates span several
orders of magnitude (~0.26 to ~234 s⁻¹ in this model).

**Observations (x):** 11 steady-state reaction fluxes (mmol / L·h), one per
reaction including the methionine drain.

**Everything else** (Michaelis constants, inhibition constants, enzyme
concentrations, Gibbs free energies, cofactor levels) is held fixed at the
enzax default values.  This is a deliberate simplification to make the
comparison tractable; extending to all parameters is straightforward.

---

## Why SNPE? — the honest justification

You are right to ask this.  SNPE is *not* more accurate than NUTS — it is an
approximation.  Here is when it is and is not worth using:

### Where SNPE genuinely helps

**1. Amortised inference across many conditions**

Once the normalizing flow is trained on (θ, x) pairs, answering
*"given these new fluxes, what is the posterior?"* takes a fraction of a
second — no new simulator calls needed.  NUTS must restart from scratch for
every new observation.

This matters in practice when you want to:
- Fit the same model to 50 patient samples or experimental conditions
- Explore how the posterior shifts as you perturb a perturbation
- Do online inference as new measurements arrive

With NUTS, 50 conditions × 2500 solver calls each = 125,000 calls.  With
SNPE, the same question costs the training budget (~3000 calls) plus essentially
zero per new condition.

**2. Simulators that cannot be differentiated**

NUTS needs the *gradient* of the log-posterior.  Enzax supports this via JAX
automatic differentiation through the ODE solver.  But most kinetic modelling
tools (MATLAB GRASP, COPASI, SBML-based models) are black boxes — you can run
them but you cannot differentiate through them.  SNPE only needs to *call* the
simulator, not differentiate it.  That is why the original `inference.py` in
this project used SNPE on the MATLAB GRASP model.

**3. Expensive simulators with many re-uses**

If a single simulator call takes minutes (e.g., a whole-cell model), paying
3000 calls upfront and then sampling for free can be more efficient than
paying 2500+ gradient evaluations each costing several calls for finite
differences.

### Where NUTS is better

- **Single inference problem, one observation**: NUTS is exact (asymptotically)
  and well-calibrated.  If you just need one good posterior, use NUTS.
- **Models with strong parameter correlations**: NUTS adapts its mass matrix
  and handles correlations naturally.  Normalizing flows need enough training
  data to learn the correlation structure.
- **Diagnosable quality**: NUTS gives R-hat and effective sample size (ESS) —
  hard numbers that tell you if it converged.  SNPE's quality is harder to
  assess without running additional checks.

### What the benchmark actually measures

| Metric | What it tells you |
|--------|-------------------|
| **Simulator calls** | Total number of times the ODE is solved |
| **Wall-clock time** | Practical time to get posterior samples |
| **Posterior mean MAE** | How close the posterior centre is to the known truth |
| **Posterior std** | Uncertainty quantification width |
| **NUTS divergences** | Whether NUTS found the geometry difficult (should be 0) |
| **SNPE log p(θ\*\|x)** | How much probability mass the flow puts at the true value |

### Known limitations of this benchmark

1. **We only infer log_kcat** — in reality you might want to infer Km values
   too.  The comparison may look different in higher dimensions.
2. **Synthetic data** — the "true" θ is known because we generated the data
   ourselves.  On real experimental data you cannot compute MAE vs truth.
3. **SNPE quality depends heavily on training budget** — with too few
   simulations the flow underfits.  The default (3000) is a reasonable start
   but should be validated with posterior predictive checks.
4. **NUTS time on CPU is long** — enzax solves a stiff ODE on every gradient
   step.  On a modern CPU this is ~0.1–0.3 s per step, so 2500 steps ≈ 10–45
   minutes.  This is the right comparison point against SNPE's training time.

---

## Interpreting the output

When you run the benchmark you will see something like:

```
═══════════════════════════════════════════════════════════════════
  SNPE posterior
═══════════════════════════════════════════════════════════════════
  Simulator calls : 3,000
  Wall-clock time : 420.3 s
  log p(θ*|x_obs) : 8.41

  Reaction       mean     std       95% CI               true   |err|
  ------------  -------  -------  ------------------  --------  -------
  MAT1          +2.063   0.121  [ +1.830,  +2.299]    +2.066    0.003
  ...
```

**How to read this:**
- `mean` is the posterior mean — your best single estimate of that parameter
- `std` is the posterior standard deviation — your uncertainty
- `95% CI` contains the true value if the inference is well-calibrated
- `|err|` is `|mean − true|` — only meaningful in this synthetic-data setting
- `log p(θ*|x_obs)` — higher means the flow assigns more probability to the
  known true parameter values; useful for comparing SNPE rounds

---

## Extending this work

To infer more parameters (e.g., Km values), edit `simulator.py`:

```python
# Currently only log_kcat is free. To also free log_enzyme:
FREE_PARAMS = ["log_kcat", "log_enzyme"]   # example extension
```

To use real experimental fluxes instead of synthetic ones, replace the
`x_obs` argument:

```python
import numpy as np
x_obs = np.array([...])   # your measured fluxes, 11 values
results = run_snpe(x_obs)
results = run_nuts(x_obs)
```
