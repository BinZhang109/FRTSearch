# IMPIC: Iterative Mask-based Parameter Inference and Calibration

> **Reference:** Zhang et al. (2025), *"FRTSearch: Unified Detection and Parameter Inference of Fast Radio Transients using Instance Segmentation"*, **Section 3.3, Algorithm 1**

## Overview

IMPIC is the parameter inference component of the FRTSearch framework. Given binary segmentation masks produced by Mask R-CNN, IMPIC extracts physical parameters — **Dispersion Measure (DM)** and **Time of Arrival (ToA)** — by fitting the cold plasma dispersion relation to the pixel-level trajectory coordinates.

**Key idea:** Rather than performing exhaustive DM-trial searches, IMPIC directly inverts the geometric relationship between a dispersive trajectory's shape and its underlying physical parameters.

## Mathematical Foundation

Radio signals traversing the interstellar medium experience frequency-dependent delays governed by the cold plasma dispersion relation (Lorimer & Kramer 2005):

```
Δt = k_DM × DM × (ν_low⁻² − ν_high⁻²)
```

where:
- `Δt` — time delay between two frequency channels (seconds)
- `k_DM = 4.148808 × 10³ MHz² pc⁻¹ cm³ s` — the dispersion constant
- `DM` — Dispersion Measure (pc cm⁻³), the integrated column density of free electrons
- `ν_low`, `ν_high` — observation frequencies (MHz)

This equation establishes a **deterministic mapping** between the pixel coordinates `{(tₖ, νₖ)}` of a segmented trajectory and the physical parameters (DM, ToA). IMPIC exploits this mapping via robust curve fitting.

## Algorithm

```
Algorithm 1: IMPIC

Input:  Masks M = {M₁, ..., Mₘ}, iterations N_iter, sample size N_sample, freq ν_high
Output: Parameter matrix T_out ∈ ℝ^{m×2} containing [ToA, DM] per instance

For each mask Mⱼ:
  1. Extract coordinates:  Pⱼ = {(tₖ, νₖ) | Mⱼ[νₖ, tₖ] = 1}
  2. Convert pixel indices → physical units (seconds, MHz)
  3. Construct pairwise differences:
       Δt = tᵢ − tⱼ,   Δ(ν⁻²) = νᵢ⁻² − νⱼ⁻²
     Filter pairs by temporal threshold (≥ half the trajectory time span)
  4. RANSAC loop (N_iter iterations):
       a. Randomly sample N_sample pairs
       b. Fit DM via NLS:  Δt = k_DM × DM × Δ(ν⁻²)
       c. Compute R² over all pairs
       d. Keep fit with highest R²
  5. Identify earliest arrival: (t_earliest, ν_earliest) = argmin_t Pⱼ
  6. Extrapolate ToA to ν_high:
       ToA = t_earliest − k_DM × DM × (ν_earliest⁻² − ν_high⁻²)
  7. Store [ToA, DM]
```

### Pairwise vs. Absolute Fitting

IMPIC supports two fitting modes controlled by `fit_pair`:

| Mode | `fit_pair` | Model | Advantage |
|------|-----------|-------|-----------|
| **Pairwise** (default) | `True` | `Δt = k_DM × DM × Δ(ν⁻²)` | Robust to mask boundary offsets; eliminates absolute time reference |
| **Absolute** | `False` | `t = k_DM × DM × (ν⁻² − ν_high⁻²) + ToA` | Jointly fits DM and ToA; fewer points needed |

The pairwise mode is recommended (and used in the paper) because it provides more robust DM estimation by relying only on relative delays between frequency pairs.

### RANSAC Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_points` | 100 | Points sampled per RANSAC iteration (N_sample) |
| `iterations` | 15 | Number of RANSAC iterations (N_iter) |
| `fit_pair` | `True` | Use pairwise fitting mode |

These defaults were selected via systematic grid search (Paper, Section 3.3.1), balancing computational cost (~0.18s average) with fitting robustness (weighted mean accuracy 83.6% within 1σ, 96.0% within 2σ).

## Implementation

### Files

| File | Description |
|------|-------------|
| [`utils/IMPIC.py`](IMPIC.py) | Standalone implementation — usable independently of the detection pipeline |
| [`utils/detector.py`](detector.py) | Pipeline integration — `FRTDetector.IMPIC()` method (functionally identical) |
| [`test_sample/test_impic.py`](../test_sample/test_impic.py) | Test examples with synthetic and real-parameter dispersion curves |

### Standalone Usage

```python
import numpy as np
from utils.IMPIC import IMPIC, compute_dispersive_delay

# Binary masks from Mask R-CNN, shape: (N_instances, N_freq, N_time)
masks = np.load("detected_masks.npy")

# Observation and preprocessing parameters
obs_params = {
    "freq_high": 1500.0,           # Upper band edge (MHz)
    "channel_bandwidth": 0.122070,  # Channel width (MHz)
    "sampling_interval": 4.9152e-5, # Time resolution (seconds)
    "downsample_time": 16,          # Time downsampling factor
    "downsample_freq": 16,          # Frequency downsampling factor
}

# RANSAC configuration
ransac_cfg = {
    "sample_points": 100,  # N_sample
    "iterations": 15,       # N_iter
    "fit_pair": True,       # Pairwise mode
}

results = IMPIC(masks, obs_params, ransac_cfg)

for i, (toa, dm) in enumerate(results):
    print(f"Instance {i}: ToA = {toa:.5f} s, DM = {dm:.2f} pc/cm³")
```

### Pipeline Usage

When running the full FRTSearch pipeline, IMPIC is called automatically:

```bash
python FRTSearch.py data.fits config.py --slide-size 128
```

The pipeline reads `obs_params` from the observation file header and `ransac_cfg` from the detector config.

## API Reference

### `IMPIC(masks, obs_params, ransac_cfg) → np.ndarray`

Main entry point. See docstring in [`utils/IMPIC.py`](IMPIC.py) for full parameter descriptions.

### `compute_dispersive_delay(f_low, f_high, dm) → float`

Computes the dispersive delay (in seconds) between two frequencies for a given DM.

## References

- Zhang, B., Wang, Y., Xie, X., et al. (2025). *FRTSearch: Unified Detection and Parameter Inference of Fast Radio Transients using Instance Segmentation*. Section 3.3, Algorithm 1.
- Lorimer, D. R. & Kramer, M. (2005). *Handbook of Pulsar Astronomy*. Cambridge University Press.
- Fischler, M. A. & Bolles, R. C. (1981). *Random Sample Consensus: A Paradigm for Model Fitting*. Comm. ACM, 24(6), 381–395.
