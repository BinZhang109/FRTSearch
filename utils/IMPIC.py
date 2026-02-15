"""
IMPIC: Iterative Mask-based Parameter Inference and Calibration

Standalone implementation of Algorithm 1 from:
    "FRTSearch: Unified Detection and Parameter Inference of Fast Radio Transients
     using Instance Segmentation" (Zhang et al. 2025), Section 3.3

This module provides the IMPIC algorithm as an independent, reusable component.
The same algorithm is integrated into the FRTSearch pipeline via FRTDetector.IMPIC()
in utils/detector.py. This standalone version allows users to apply IMPIC to any
set of binary segmentation masks without running the full detection pipeline.

Usage:
    from utils.IMPIC import IMPIC, compute_dispersive_delay

    results = IMPIC(masks, obs_params, ransac_cfg)

Author: Bin Zhang
Date: 2024.06.23
"""

import numpy as np
import random
from typing import Tuple, Callable, Dict, Optional
from scipy.optimize import curve_fit


# Dispersion constant in CGS units:
#   k_DM = 4.148808 x 10^3  MHz^2 pc^{-1} cm^3 s
#
# In this implementation we store it as 4148808.0 (i.e., k_DM * 10^6) so that
# the delay formula  Δt = k_DM * DM * Δ(ν^{-2})  yields time in *milliseconds*
# when frequencies are in MHz and DM is in pc cm^{-3}.  The final conversion to
# seconds is done by dividing by 1000 at the point of use.
DISPERSION_CONSTANT = 4148808.0


def compute_dispersive_delay(f_low: float, f_high: float, dm: float) -> float:
    """
    Compute the dispersive time delay between two frequency channels.

    Implements Equation (1) of Zhang et al. (2025):

        Δt ≈ k_DM × DM × (ν_low^{-2} - ν_high^{-2})

    Parameters
    ----------
    f_low : float
        Lower frequency channel (MHz).
    f_high : float
        Higher frequency channel (MHz).
    dm : float
        Dispersion Measure (pc cm^{-3}).

    Returns
    -------
    float
        Time delay in seconds.
    """
    return (1.0 / f_low ** 2 - 1.0 / f_high ** 2) * DISPERSION_CONSTANT * dm / 1000.0


def _ransac_curve_fit(
    func: Callable,
    sample_points: np.ndarray,
    eval_points: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Fit the dispersion model to a random subset and evaluate on all points.

    Uses Nonlinear Least Squares (NLS) via scipy.optimize.curve_fit on the
    sampled subset, then scores the fit using the coefficient of determination
    (R^2) computed over the full evaluation set.

    Parameters
    ----------
    func : Callable
        Model function f(x, *params) encoding the dispersion relation.
    sample_points : np.ndarray, shape (N_sample, 2)
        Randomly sampled points [Δt, Δ(ν^{-2})] used for fitting.
    eval_points : np.ndarray, shape (N_total, 2)
        All available points [Δt, Δ(ν^{-2})] used for R^2 evaluation.

    Returns
    -------
    params : np.ndarray
        Best-fit parameters from curve_fit (DM, or [DM, ToA]).
    r2 : float
        Coefficient of determination R^2 ∈ [0, 1].
    """
    delta_t_sample = sample_points[:, 0]
    delta_nu_sample = sample_points[:, 1]

    delta_t_eval = eval_points[:, 0]
    delta_nu_eval = eval_points[:, 1]

    popt, _ = curve_fit(func, xdata=delta_nu_sample, ydata=delta_t_sample)
    pred = func(delta_nu_eval, *popt)

    ss_tot = np.sum((delta_t_eval - np.mean(delta_t_eval)) ** 2)
    ss_res = np.sum((delta_t_eval - pred) ** 2)
    r2 = max(1.0 - ss_res / ss_tot, 0.0)

    return popt, r2


def IMPIC(
    masks: np.ndarray,
    obs_params: Dict[str, float],
    ransac_cfg: Dict[str, int],
) -> np.ndarray:
    """
    Iterative Mask-based Parameter Inference and Calibration (IMPIC).

    Given a set of binary segmentation masks (one per detected FRT instance),
    infers Dispersion Measure (DM) and Time of Arrival (ToA) for each instance
    by fitting the cold plasma dispersion relation via RANSAC.

    This implements Algorithm 1 of Zhang et al. (2025), Section 3.3:

        1. For each mask M_j, extract pixel coordinates
           P_j = {(t_k, ν_k) | M_j[ν_k, t_k] = 1}
        2. Convert pixel indices → physical units (seconds, MHz)
        3. Construct pairwise differences Δt and Δ(ν^{-2}) and filter by
           a temporal threshold to ensure geometric leverage
        4. RANSAC loop (N_iter iterations):
           a. Randomly sample N_sample point pairs
           b. Fit DM via NLS on Δt = k_DM × DM × Δ(ν^{-2})
           c. Compute R^2 over all pairs
           d. Retain the fit with highest R^2
        5. Identify earliest arrival (t_earliest, ν_earliest) from P_j
        6. Extrapolate ToA to ν_high: ToA = t_earliest − k_DM × DM × (ν_earliest^{-2} − ν_high^{-2})
        7. Return [ToA, DM] for each instance

    Parameters
    ----------
    masks : np.ndarray, shape (N_instances, N_freq, N_time)
        Binary segmentation masks from Mask R-CNN. Each mask delineates a
        single dispersive trajectory in the preprocessed dynamic spectrum.

    obs_params : dict
        Observation and preprocessing parameters:
        - 'freq_high'        : float — Upper edge of the band (MHz)
        - 'channel_bandwidth': float — Frequency resolution per channel (MHz)
        - 'sampling_interval': float — Time resolution per sample (seconds)
        - 'downsample_time'  : int   — Time downsampling factor applied during preprocessing
        - 'downsample_freq'  : int   — Frequency downsampling factor applied during preprocessing

    ransac_cfg : dict
        RANSAC fitting configuration:
        - 'sample_points': int  — N_sample, number of points per iteration (default: 100)
        - 'iterations'   : int  — N_iter, number of RANSAC iterations (default: 15)
        - 'fit_pair'     : bool — If True (default), use pairwise Δt vs Δ(ν^{-2}) fitting;
                                   if False, fit absolute t vs (ν^{-2} − ν_high^{-2})

    Returns
    -------
    np.ndarray, shape (N_instances, 2)
        Each row contains [ToA (seconds), DM (pc cm^{-3})].
        Rows are zero-filled for instances with insufficient mask pixels.

    Notes
    -----
    - The pairwise fitting mode (fit_pair=True) is more robust to mask
      boundary offsets because it eliminates the absolute time reference,
      relying only on relative delays between frequency pairs.
    - The temporal threshold filters out point pairs with small Δt,
      ensuring sufficient geometric leverage for accurate DM estimation.
    - This standalone function is equivalent to FRTDetector.IMPIC() in
      utils/detector.py; the pipeline version reads obs_params from the
      loaded observation file automatically.

    References
    ----------
    Zhang et al. (2025), "FRTSearch: Unified Detection and Parameter Inference
    of Fast Radio Transients using Instance Segmentation", Section 3.3, Algorithm 1.
    """
    n_sample = ransac_cfg.get("sample_points", 100)
    n_iter = ransac_cfg.get("iterations", 15)
    fit_pair = ransac_cfg.get("fit_pair", True)

    freq_high = obs_params["freq_high"]
    channel_bw = obs_params["channel_bandwidth"]
    tsamp = obs_params["sampling_interval"]
    ds_time = obs_params["downsample_time"]
    ds_freq = obs_params["downsample_freq"]

    output = np.zeros((masks.shape[0], 2))

    for midx, mask in enumerate(masks):
        assert mask.ndim == 2, f"Expected 2D mask, got shape {mask.shape}"

        # Step 1: Extract pixel coordinates  P_j = {(freq_idx, time_idx)}
        vf, vt = np.where(mask.astype(bool))
        if vt.shape[0] == 0:
            continue

        # Step 2: Convert pixel indices to physical units
        #   time  (s)   = pixel_t × downsample_time × sampling_interval
        #   freq  (MHz) = freq_high − pixel_f × downsample_freq × channel_bandwidth
        vt = vt.astype(np.float64) * ds_time * tsamp
        vf = freq_high - vf.astype(np.float64) * ds_freq * channel_bw

        # Step 3: Build fitting data
        if fit_pair:
            # Pairwise mode: Δt vs Δ(ν^{-2})
            # Apply temporal threshold = half the trajectory span
            threshold = (np.max(vt) - np.min(vt)) / 2.0
            difft = (vt[np.newaxis, :] - vt[:, np.newaxis]).flatten()
            difff = (1.0 / vf[np.newaxis, :] ** 2 - 1.0 / vf[:, np.newaxis] ** 2).flatten()

            selected = np.abs(difft) >= threshold
            difft = difft[selected]
            difff = difff[selected]

            if difft.shape[0] == 0:
                continue

            # Model: Δt = k_DM × DM × Δ(ν^{-2}) / 1000
            func = lambda x, dm: x * DISPERSION_CONSTANT * dm / 1000.0
        else:
            # Absolute mode: t vs (ν^{-2} − ν_high^{-2})
            difft = vt
            difff = 1.0 / vf ** 2 - 1.0 / freq_high ** 2

            # Model: t = k_DM × DM × (ν^{-2} − ν_high^{-2}) / 1000 + ToA
            func = lambda x, dm, toa: x * DISPERSION_CONSTANT * dm / 1000.0 + toa

        points = np.stack([difft, difff], axis=1)

        # Step 4: RANSAC loop — find best DM
        best_r2 = float("-inf")
        best_param = None

        for _ in range(n_iter):
            k = min(points.shape[0], n_sample)
            idx = random.sample(range(points.shape[0]), k=k)
            sampled = points[idx]

            try:
                cur_param, cur_r2 = _ransac_curve_fit(func, sampled, points)
            except RuntimeError:
                # curve_fit may fail on degenerate subsets
                continue

            if cur_r2 > best_r2:
                best_r2 = cur_r2
                best_param = cur_param

        if best_param is None:
            continue

        # Step 5–6: Derive ToA
        if fit_pair:
            dm = best_param[0]
            # Find earliest arrival in the trajectory
            earliest_idx = np.argmin(vt)
            t_earliest, f_earliest = vt[earliest_idx], vf[earliest_idx]
            # Extrapolate to ν_high
            delta_t = compute_dispersive_delay(f_earliest, freq_high, dm)
            toa = t_earliest - delta_t
        else:
            dm, toa = best_param

        output[midx, 0] = max(toa, 0.0)
        output[midx, 1] = dm

    return output
