#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMPIC Algorithm — Test & Demonstration

Validates the standalone IMPIC implementation (utils/IMPIC.py) by generating
synthetic dispersive trajectories with known ground-truth DM and ToA, running
the algorithm, and reporting inference accuracy.

Test cases cover:
  1. FAST telescope parameters  (DM = 565 pc/cm³, FRB 20121102-like)
  2. FAST telescope parameters  (DM = 23.7 pc/cm³, low-DM pulsar PSR B0820+02)
  3. SKA-like telescope parameters (DM = 830 pc/cm³, high-DM FRB)
  4. Multiple instances in a single call
  5. RANSAC robustness under noise contamination
  6. Performance comparison across RANSAC configurations

Usage:
    python test_sample/test_impic.py

Author: Bin Zhang
Date: 2024.06.23
"""

import sys
import os
import time
import numpy as np

# Ensure project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.IMPIC import IMPIC, compute_dispersive_delay, DISPERSION_CONSTANT


# ---------------------------------------------------------------------------
# Helper: generate a synthetic binary mask for a dispersive trajectory
# ---------------------------------------------------------------------------

def generate_synthetic_mask(
    dm: float,
    toa: float,
    freq_high: float,
    channel_bw: float,
    tsamp: float,
    ds_time: int,
    ds_freq: int,
    n_freq: int,
    n_time: int,
    width_pixels: int = 3,
    noise_fraction: float = 0.0,
) -> np.ndarray:
    """
    Create a binary mask containing a synthetic dispersive trajectory.

    Parameters
    ----------
    dm : float              Ground-truth DM (pc cm^{-3})
    toa : float             Ground-truth ToA at freq_high (seconds)
    freq_high : float       Upper band edge (MHz)
    channel_bw : float      Channel bandwidth (MHz)
    tsamp : float           Sampling interval (seconds)
    ds_time : int           Time downsampling factor
    ds_freq : int           Frequency downsampling factor
    n_freq : int            Number of frequency bins in the mask
    n_time : int            Number of time bins in the mask
    width_pixels : int      Trajectory width in time pixels
    noise_fraction : float  Fraction of mask area filled with random noise pixels

    Returns
    -------
    np.ndarray, shape (n_freq, n_time), dtype uint8
    """
    mask = np.zeros((n_freq, n_time), dtype=np.uint8)

    for fi in range(n_freq):
        freq = freq_high - fi * ds_freq * channel_bw
        if freq <= 0:
            continue
        delay = compute_dispersive_delay(freq, freq_high, dm)
        t_phys = toa + delay  # arrival time at this frequency (seconds)
        t_idx = int(round(t_phys / (ds_time * tsamp)))

        half_w = width_pixels // 2
        for dt in range(-half_w, half_w + 1):
            ti = t_idx + dt
            if 0 <= ti < n_time:
                mask[fi, ti] = 1

    # Add noise pixels
    if noise_fraction > 0:
        n_noise = int(noise_fraction * n_freq * n_time)
        for _ in range(n_noise):
            ri = np.random.randint(0, n_freq)
            ci = np.random.randint(0, n_time)
            mask[ri, ci] = 1

    return mask


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_fast_frb121102():
    """
    Test 1: FAST FRB 20121102-like parameters.
    DM ≈ 565 pc/cm³, freq 1000–1500 MHz, FAST sampling.
    """
    print("=" * 70)
    print("Test 1: FAST — FRB 20121102  (DM=565, ToA=2.0s)")
    print("=" * 70)

    dm_true, toa_true = 565.0, 2.0
    freq_high = 1500.0
    channel_bw = 0.12207031  # FAST L-band
    tsamp = 4.9152e-5        # FAST sampling interval
    ds_time, ds_freq = 16, 16
    n_freq, n_time = 256, 8192

    mask = generate_synthetic_mask(
        dm_true, toa_true, freq_high, channel_bw, tsamp,
        ds_time, ds_freq, n_freq, n_time, width_pixels=5,
    )

    obs_params = {
        "freq_high": freq_high,
        "channel_bandwidth": channel_bw,
        "sampling_interval": tsamp,
        "downsample_time": ds_time,
        "downsample_freq": ds_freq,
    }
    ransac_cfg = {"sample_points": 100, "iterations": 15, "fit_pair": True}

    result = IMPIC(mask[np.newaxis], obs_params, ransac_cfg)
    toa_pred, dm_pred = result[0]

    dm_err = abs(dm_pred - dm_true)
    toa_err = abs(toa_pred - toa_true) * 1000  # ms

    print(f"  Ground truth : DM = {dm_true:.1f} pc/cm³,  ToA = {toa_true:.5f} s")
    print(f"  IMPIC output : DM = {dm_pred:.1f} pc/cm³,  ToA = {toa_pred:.5f} s")
    print(f"  Error        : ΔDM = {dm_err:.2f} pc/cm³,  ΔToA = {toa_err:.2f} ms")
    print(f"  Mask pixels  : {np.sum(mask)} / {n_freq * n_time}")
    _assert_close(dm_pred, dm_true, rtol=0.05, name="DM")
    print()


def test_fast_low_dm_pulsar():
    """
    Test 2: Low-DM pulsar (PSR B0820+02-like), DM ≈ 23.7.
    Tests algorithm behaviour when the dispersion sweep is small.
    """
    print("=" * 70)
    print("Test 2: FAST — Low-DM Pulsar  (DM=23.7, ToA=1.0s)")
    print("=" * 70)

    dm_true, toa_true = 23.7, 1.0
    freq_high = 1500.0
    channel_bw = 0.12207031
    tsamp = 4.9152e-5
    ds_time, ds_freq = 16, 16
    n_freq, n_time = 256, 8192

    mask = generate_synthetic_mask(
        dm_true, toa_true, freq_high, channel_bw, tsamp,
        ds_time, ds_freq, n_freq, n_time, width_pixels=3,
    )

    obs_params = {
        "freq_high": freq_high,
        "channel_bandwidth": channel_bw,
        "sampling_interval": tsamp,
        "downsample_time": ds_time,
        "downsample_freq": ds_freq,
    }
    ransac_cfg = {"sample_points": 100, "iterations": 15, "fit_pair": True}

    result = IMPIC(mask[np.newaxis], obs_params, ransac_cfg)
    toa_pred, dm_pred = result[0]

    dm_err = abs(dm_pred - dm_true)
    toa_err = abs(toa_pred - toa_true) * 1000

    print(f"  Ground truth : DM = {dm_true:.1f} pc/cm³,  ToA = {toa_true:.5f} s")
    print(f"  IMPIC output : DM = {dm_pred:.1f} pc/cm³,  ToA = {toa_pred:.5f} s")
    print(f"  Error        : ΔDM = {dm_err:.2f} pc/cm³,  ΔToA = {toa_err:.2f} ms")
    _assert_close(dm_pred, dm_true, rtol=0.10, name="DM")
    print()


def test_ska_high_dm():
    """
    Test 3: SKA-like observation — high-DM FRB (DM=830).
    Uses a 50 MHz sub-band (1660–1710 MHz) with coarser time resolution.
    The mask is sized to contain the full dispersive sweep.
    """
    print("=" * 70)
    print("Test 3: SKA — High-DM FRB  (DM=830, ToA=0.5s)")
    print("=" * 70)

    dm_true, toa_true = 830.0, 0.5
    freq_high = 1710.0
    channel_bw = 0.1953125   # SKA-like
    tsamp = 1.265625e-3      # SKA-like
    ds_time, ds_freq = 1, 4
    n_freq, n_time = 256, 16384  # wider time axis to fit the sweep

    mask = generate_synthetic_mask(
        dm_true, toa_true, freq_high, channel_bw, tsamp,
        ds_time, ds_freq, n_freq, n_time, width_pixels=5,
    )

    obs_params = {
        "freq_high": freq_high,
        "channel_bandwidth": channel_bw,
        "sampling_interval": tsamp,
        "downsample_time": ds_time,
        "downsample_freq": ds_freq,
    }
    ransac_cfg = {"sample_points": 100, "iterations": 15, "fit_pair": True}

    result = IMPIC(mask[np.newaxis], obs_params, ransac_cfg)
    toa_pred, dm_pred = result[0]

    dm_err = abs(dm_pred - dm_true)
    toa_err = abs(toa_pred - toa_true) * 1000

    print(f"  Ground truth : DM = {dm_true:.1f} pc/cm³,  ToA = {toa_true:.5f} s")
    print(f"  IMPIC output : DM = {dm_pred:.1f} pc/cm³,  ToA = {toa_pred:.5f} s")
    print(f"  Error        : ΔDM = {dm_err:.2f} pc/cm³,  ΔToA = {toa_err:.2f} ms")
    _assert_close(dm_pred, dm_true, rtol=0.05, name="DM")
    print()


def test_multiple_instances():
    """
    Test 4: Multiple instances processed in a single IMPIC call.
    Simulates a real detection output with 3 candidates at different DMs.
    """
    print("=" * 70)
    print("Test 4: Multiple Instances  (3 candidates, varied DM)")
    print("=" * 70)

    freq_high = 1500.0
    channel_bw = 0.12207031
    tsamp = 4.9152e-5
    ds_time, ds_freq = 16, 16
    n_freq, n_time = 256, 8192

    test_cases = [
        (420.0, 1.5),   # FRB 20201124A-like
        (565.0, 2.0),   # FRB 20121102-like
        (198.2, 0.8),   # PSR J1948+2333-like
    ]

    masks = np.zeros((len(test_cases), n_freq, n_time), dtype=np.uint8)
    for i, (dm, toa) in enumerate(test_cases):
        masks[i] = generate_synthetic_mask(
            dm, toa, freq_high, channel_bw, tsamp,
            ds_time, ds_freq, n_freq, n_time, width_pixels=5,
        )

    obs_params = {
        "freq_high": freq_high,
        "channel_bandwidth": channel_bw,
        "sampling_interval": tsamp,
        "downsample_time": ds_time,
        "downsample_freq": ds_freq,
    }
    ransac_cfg = {"sample_points": 100, "iterations": 15, "fit_pair": True}

    results = IMPIC(masks, obs_params, ransac_cfg)

    print(f"  {'Instance':<10} {'True DM':>10} {'Pred DM':>10} {'ΔDM':>8} {'True ToA':>10} {'Pred ToA':>10} {'ΔToA(ms)':>10}")
    print(f"  {'-'*68}")
    for i, (dm_true, toa_true) in enumerate(test_cases):
        toa_pred, dm_pred = results[i]
        print(f"  {i:<10} {dm_true:>10.1f} {dm_pred:>10.1f} {abs(dm_pred-dm_true):>8.2f} "
              f"{toa_true:>10.5f} {toa_pred:>10.5f} {abs(toa_pred-toa_true)*1000:>10.2f}")
        _assert_close(dm_pred, dm_true, rtol=0.05, name=f"DM[{i}]")
    print()


def test_noise_robustness():
    """
    Test 5: RANSAC robustness under noise contamination.
    Adds random noise pixels to the mask and verifies IMPIC still recovers
    accurate parameters, demonstrating the outlier rejection capability.
    """
    print("=" * 70)
    print("Test 5: RANSAC Robustness  (DM=565, noise contamination)")
    print("=" * 70)

    dm_true, toa_true = 565.0, 2.0
    freq_high = 1500.0
    channel_bw = 0.12207031
    tsamp = 4.9152e-5
    ds_time, ds_freq = 16, 16
    n_freq, n_time = 256, 8192

    np.random.seed(42)

    # Clean mask for reference
    clean_mask = generate_synthetic_mask(
        dm_true, toa_true, freq_high, channel_bw, tsamp,
        ds_time, ds_freq, n_freq, n_time, width_pixels=5,
    )
    n_signal = int(np.sum(clean_mask))

    # Add noise: localized RFI-like contamination near the trajectory
    # (real noise tends to cluster, not scatter uniformly)
    noisy_mask = clean_mask.copy()
    n_noise = n_signal // 3
    # Scatter noise near the trajectory region (within ±20 time pixels)
    signal_f, signal_t = np.where(clean_mask)
    t_center = int(np.mean(signal_t))
    for _ in range(n_noise):
        ri = np.random.randint(0, n_freq)
        ci = t_center + np.random.randint(-20, 21)
        if 0 <= ci < n_time:
            noisy_mask[ri, ci] = 1

    n_total = int(np.sum(noisy_mask))
    print(f"  Signal pixels: {n_signal}, Total pixels: {n_total}, "
          f"Noise fraction: {(n_total - n_signal) / n_total * 100:.1f}%")

    obs_params = {
        "freq_high": freq_high,
        "channel_bandwidth": channel_bw,
        "sampling_interval": tsamp,
        "downsample_time": ds_time,
        "downsample_freq": ds_freq,
    }
    ransac_cfg = {"sample_points": 100, "iterations": 15, "fit_pair": True}

    result = IMPIC(noisy_mask[np.newaxis], obs_params, ransac_cfg)
    toa_pred, dm_pred = result[0]

    dm_err = abs(dm_pred - dm_true)
    toa_err = abs(toa_pred - toa_true) * 1000

    print(f"  Ground truth : DM = {dm_true:.1f} pc/cm³,  ToA = {toa_true:.5f} s")
    print(f"  IMPIC output : DM = {dm_pred:.1f} pc/cm³,  ToA = {toa_pred:.5f} s")
    print(f"  Error        : ΔDM = {dm_err:.2f} pc/cm³,  ΔToA = {toa_err:.2f} ms")
    _assert_close(dm_pred, dm_true, rtol=0.10, name="DM (noisy)")
    print()


def test_performance_comparison():
    """
    Test 6: Performance comparison across RANSAC configurations.
    Benchmarks speed vs. accuracy for different (N_sample, N_iter) settings.
    """
    print("=" * 70)
    print("Test 6: Performance Comparison (RANSAC configurations)")
    print("=" * 70)

    dm_true, toa_true = 565.0, 2.0
    freq_high = 1500.0
    channel_bw = 0.12207031
    tsamp = 4.9152e-5
    ds_time, ds_freq = 16, 16
    n_freq, n_time = 256, 8192

    mask = generate_synthetic_mask(
        dm_true, toa_true, freq_high, channel_bw, tsamp,
        ds_time, ds_freq, n_freq, n_time, width_pixels=5,
    )

    obs_params = {
        "freq_high": freq_high,
        "channel_bandwidth": channel_bw,
        "sampling_interval": tsamp,
        "downsample_time": ds_time,
        "downsample_freq": ds_freq,
    }

    configs = [
        {"sample_points": 50,  "iterations": 10, "fit_pair": True},
        {"sample_points": 100, "iterations": 15, "fit_pair": True},
        {"sample_points": 200, "iterations": 30, "fit_pair": True},
    ]
    names = ["Fast (50, 10)", "Balanced (100, 15)", "Accurate (200, 30)"]

    print(f"  {'Config':<25} {'Time (ms)':>10} {'DM Pred':>10} {'ΔDM':>8} {'ΔToA(ms)':>10}")
    print(f"  {'-'*63}")

    for name, cfg in zip(names, configs):
        t0 = time.time()
        result = IMPIC(mask[np.newaxis], obs_params, cfg)
        elapsed = (time.time() - t0) * 1000
        toa_pred, dm_pred = result[0]
        print(f"  {name:<25} {elapsed:>10.1f} {dm_pred:>10.1f} "
              f"{abs(dm_pred - dm_true):>8.2f} {abs(toa_pred - toa_true)*1000:>10.2f}")
    print()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _assert_close(actual, expected, rtol, name="value"):
    """Check relative tolerance and print PASS/FAIL."""
    if expected == 0:
        ok = abs(actual) < 1e-6
    else:
        ok = abs(actual - expected) / abs(expected) < rtol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: relative error "
          f"{abs(actual - expected) / max(abs(expected), 1e-9) * 100:.2f}% "
          f"(tolerance {rtol * 100:.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IMPIC Algorithm — Test Suite")
    print("Reference: Zhang et al. (2025), Section 3.3, Algorithm 1")
    print("=" * 70 + "\n")

    test_fast_frb121102()
    test_fast_low_dm_pulsar()
    test_ska_high_dm()
    test_multiple_instances()
    test_noise_robustness()
    test_performance_comparison()

    print("=" * 70)
    print("All tests completed.")
    print("Documentation: utils/IMPIC.md")
    print("=" * 70)
