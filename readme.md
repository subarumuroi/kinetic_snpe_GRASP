# GRASP + SNPE

Replaces the ABC rejection sampler in Saa & Nielsen (2016) with
Sequential Neural Posterior Estimation (SNPE-C).

## Setup

```bash
pip install -r requirements.txt
```

## Running now (mock mode)

```bash
python inference.py
```

This runs end to end with a mock simulator so you can verify the
SNPE pipeline works before touching MATLAB.

## At work - plugging in real GRASP

Three steps:

**1. Find the GRASP output struct fields**
```matlab
ensemble = % your existing code that runs GRASP
fieldnames(ensemble)
fieldnames(ensemble.models(1))  % or however it's indexed
```
Paste the output and fill in `_extract_theta` and `_extract_fluxes`
in `GRASPSimulator`.

**2. Switch to real simulator**
```python
simulator = GRASPSimulator(
    grasp_path="/path/to/GRASP",
    use_mock=False
)
```

**3. Run with real data**
The reference fluxes and perturbations are already hardcoded from
Figure 2a/2c of the paper so nothing else needs changing.

## What to expect

The paper reports ~13-22 hours on 16 CPUs to get 1000 accepted
samples from rejection sampling, with acceptance rates dropping
steeply as more experimental datasets are added (Figure 3b).

SNPE should get comparable posterior quality with roughly 10-100x
fewer simulator calls by learning the posterior density rather than
rejecting samples.

## Key files

- `inference.py` - main pipeline
  - `GRASPSimulator` - MATLAB interface (fill in at work)
  - `GRASPPosteriorEstimator` - SNPE training and sampling
  - `abc_distance` - equation 3 from the paper
  - `plot_flux_posterior` - reproduces Figure 4 style plots