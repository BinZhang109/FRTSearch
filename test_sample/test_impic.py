#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_impic.py — Demo: Mask R-CNN detection -> collect masks -> standalone IMPIC parameter inference

Usage:
    python test_sample/test_impic.py
    python test_sample/test_impic.py --file ./test_sample/FRB20121102_0038.fits --slide-size 128
"""

import sys
import os
import time
import gc
import argparse

import numpy as np
import torch

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from mmengine.config import Config
from mmdet.apis.inference import inference_detector
from utils import FRTDetector, PsrfitsFile, FilterbankFile
from utils.loader import cpu_nms, dynamic_correct, ops_in_ppl
from utils.IMPIC import IMPIC, compute_dispersive_delay


def collect_masks_single_window(detector):
    """
    Run Mask R-CNN inference on the currently loaded spectra and
    collect bboxes and masks (without IMPIC, coordinate_mapping=False).

    Returns:
        (bbox, masks) or None
        bbox:  np.ndarray, shape (N, 5) — [x1, y1, x2, y2, score]
        masks: np.ndarray, shape (N, H, W) — binary masks
    """
    if detector.fetch_raw_data() is None or len(detector.models) == 0:
        return None

    processed = detector.fetch_processed_data(apply_clipping=True)
    if processed is None:
        return None
    processed_3ch = processed[:, :, np.newaxis]

    all_box, all_mask = [], []
    for model in detector.models:
        try:
            model_cfg = model.cfg
            test_pipeline = detector._extract_test_pipeline(model_cfg)
            inp = processed_3ch if ops_in_ppl(test_pipeline, ["DynamicCorrect"]) \
                else dynamic_correct(processed_3ch)

            result = inference_detector(model, inp)
            if inp is not processed_3ch:
                del inp

            pred = result.pred_instances if hasattr(result, 'pred_instances') else \
                   result[0].pred_instances if isinstance(result, list) and len(result) > 0 else None
            if pred is None:
                continue

            bboxes = pred.bboxes.detach().cpu().numpy()
            scores = pred.scores.detach().cpu().numpy()
            bbox = np.concatenate([bboxes, scores.reshape(-1, 1)], axis=1)

            mt = pred.masks
            if mt is None:
                continue
            masks = mt.detach().cpu().numpy() if hasattr(mt, 'detach') else mt
            if masks.ndim == 3:
                masks = (masks > 0.5).astype(np.uint8)
            elif masks.ndim == 2:
                masks = (masks > 0.5).astype(np.uint8)[np.newaxis]
            else:
                continue

            idx = bbox[:, -1] >= detector.postprocess.threshold
            bbox, masks = bbox[idx], masks[idx]
            if bbox.shape[0] > 0:
                all_box.append(bbox)
                all_mask.append(masks)

        except Exception as e:
            print(f"  Model inference error: {e}")
            continue

    del processed, processed_3ch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(all_box) == 0:
        return None

    bbox = np.concatenate(all_box, axis=0)
    masks = np.concatenate(all_mask, axis=0)
    nms_cfg = detector.postprocess.nms_cfg
    bbox, masks = cpu_nms(bbox=bbox, masks=masks, nms_cfg=nms_cfg)
    return bbox, masks


def run_test(file_path, config_file, slide_size=32):
    """
    Full pipeline:
      1. Load model & observation data
      2. Sliding-window Mask R-CNN detection -> collect all masks
      3. Standalone IMPIC module for ToA / DM inference
      4. Print results
    """
    print("=" * 70)
    print("test_impic.py — Mask R-CNN Detection -> IMPIC Parameter Inference")
    print("=" * 70)
    print(f"Input file  : {file_path}")
    print(f"Config file : {config_file}")
    print(f"slide_size  : {slide_size}")
    print()

    # -- Step 1: Load model & observation data --------------------------
    t0 = time.time()
    cfg = Config.fromfile(config_file)
    detector = FRTDetector(cfg)
    detector.load_observation(file_path, slide_size=slide_size)

    print(f"[Step 1] Model & data loaded ({time.time() - t0:.2f}s)")
    print(f"  Sampling interval : {detector.sampling_interval:.6e} s")
    print(f"  Channel bandwidth : {detector.channel_bandwidth:.6f} MHz")
    print(f"  Frequency range   : {detector.preprocess.freq_range}")
    print(f"  Downsample (time) : {detector.preprocess.downsample_time}")
    print(f"  Downsample (freq) : {detector.preprocess.downsample_freq}")
    print()

    # -- Step 2: Sliding-window detection, collect masks ----------------
    t1 = time.time()
    step_samp = slide_size * detector.samples_per_subint
    step_time = step_samp * detector.sampling_interval
    nsamp = detector.total_samples
    samppersubint = detector.samples_per_subint

    all_bboxes, all_masks = [], []
    window_offsets = []       # start time of each mask's window
    index, start_time, slide_count = 0, 0.0, 0

    while index < nsamp:
        end_index = nsamp if nsamp < 1.5 * step_samp + index else index + step_samp

        # Load current window data
        if detector.file_extension.endswith('.fil'):
            detector.spectra = detector.observation_file.get_spectra_slide(
                start_time, index, end_index)
        elif detector.file_extension.endswith('.fits'):
            start_subint = int(index // samppersubint)
            end_subint = min(int(end_index // samppersubint),
                             int(detector.observation_file.nsubints - 1))
            detector.spectra = detector.observation_file.get_spectra_slide(
                start_time, start_subint, end_subint)

        # Detect
        res = collect_masks_single_window(detector)
        if res is not None:
            bbox, masks = res
            print(f"  Window {slide_count + 1} (t={start_time:.2f}s): {len(bbox)} candidates detected")
            all_bboxes.append(bbox)
            all_masks.append(masks)
            window_offsets.extend([start_time] * len(bbox))
        else:
            print(f"  Window {slide_count + 1} (t={start_time:.2f}s): no detections")

        detector.spectra = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        index = end_index
        start_time += step_time
        slide_count += 1

    t_det = time.time() - t1
    print(f"\n[Step 2] Detection complete: {slide_count} windows, elapsed {t_det:.2f}s")

    if len(all_bboxes) == 0:
        print("  No candidates detected, exiting.")
        detector.release_memory()
        return

    combined_bbox = np.concatenate(all_bboxes, axis=0)
    window_offsets = np.array(window_offsets)

    # Masks from different windows may have different time widths (last window shorter);
    # pad to uniform size before concatenation
    max_h = max(m.shape[1] for m in all_masks)
    max_w = max(m.shape[2] for m in all_masks)
    padded = []
    for m in all_masks:
        if m.shape[1] != max_h or m.shape[2] != max_w:
            p = np.zeros((m.shape[0], max_h, max_w), dtype=m.dtype)
            p[:, :m.shape[1], :m.shape[2]] = m
            padded.append(p)
        else:
            padded.append(m)
    combined_masks = np.concatenate(padded, axis=0)
    print(f"  Collected {len(combined_bbox)} masks total (before global NMS)")

    # Optional: global NMS across windows
    # combined_bbox, combined_masks = cpu_nms(combined_bbox, combined_masks,
    #                                          detector.postprocess.nms_cfg)
    # print(f"  After global NMS: {len(combined_bbox)} masks")

    print(f"  Mask shape: {combined_masks.shape}")
    print()

    # -- Step 3: Standalone IMPIC for ToA & DM inference ----------------
    t2 = time.time()

    obs_params = {
        "freq_high":         detector.preprocess.freq_range[-1],
        "channel_bandwidth": detector.channel_bandwidth,
        "sampling_interval": detector.sampling_interval,
        "downsample_time":   detector.preprocess.downsample_time,
        "downsample_freq":   detector.preprocess.downsample_freq,
    }
    ransac_cfg = dict(detector.postprocess.mapping.ransac_cfg)

    print("[Step 3] Running IMPIC (standalone module)")
    print(f"  obs_params : {obs_params}")
    print(f"  ransac_cfg : {ransac_cfg}")

    toa_dms = IMPIC(combined_masks, obs_params, ransac_cfg)

    # Add window offset to get absolute ToA
    toa_dms[:, 0] += window_offsets

    t_impic = time.time() - t2
    print(f"  IMPIC complete ({t_impic:.2f}s)")
    print()

    # -- Step 4: Compile & print results --------------------------------
    scores = combined_bbox[:, -1]
    results = np.column_stack([toa_dms, scores])  # (N, 3): [ToA, DM, score]

    # Filter invalid results (DM=0 indicates fitting failure)
    valid = results[:, 1] > 0
    results = results[valid]

    # Sort by confidence descending
    results = results[results[:, 2].argsort()[::-1]]

    print("=" * 70)
    print(f"Results: {len(results)} valid candidates")
    print(f"{'Rank':<6} {'ToA (s)':<14} {'DM (pc/cm3)':<16} {'Confidence':<12}")
    print("-" * 70)
    for i, (toa, dm, score) in enumerate(results, 1):
        print(f"{i:<6} {toa:<14.5f} {dm:<16.2f} {score:<12.4f}")
    print("=" * 70)

    print(f"\nTotal elapsed: {time.time() - t0:.2f}s  "
          f"(load {t0:.2f}s | detect {t_det:.2f}s | IMPIC {t_impic:.2f}s)")

    # Cleanup
    detector.release_memory()
    gc.collect()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask R-CNN -> IMPIC parameter inference test")
    parser.add_argument("--file", type=str,
                        default="./test_sample/FRB20121102_0038.fits",
                        help="Observation file path (.fits/.fil)")
    parser.add_argument("--config", type=str,
                        default="./configs/detector_FAST.py",
                        help="Detector config file")
    parser.add_argument("--slide-size", type=int, default=128,
                        help="Sliding window size (number of subints)")
    args = parser.parse_args()

    run_test(args.file, args.config, args.slide_size)
