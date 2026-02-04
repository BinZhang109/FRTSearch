#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Bin Zhang
Date: 2025.8.11

FRTSearch Core Detector Module
"""

import copy
import gc
import random
import time
from typing import Union, Dict, Tuple, Callable, Optional, List

import numpy as np
import torch
from scipy.optimize import curve_fit

from mmdet.apis.inference import inference_detector, init_detector
from mmengine.config import Config

from .loader import (cpu_nms, dynamic_correct, ops_in_ppl, 
                     dm_filtering, dm_shift, toa_shift, entity)
from .backend import FitsBackend


__all__ = ['FRTDetector', 'DISPERSION_CONSTANT']

DISPERSION_CONSTANT = 4148808.0


class FRTDetector:
    """Fast Radio Transient Detector using deep learning instance segmentation."""
    
    observation_params = ["sampling_interval", "frequency_channels", "channel_bandwidth"]

    def __init__(self, config: Union[str, Config]):
        self.backend = FitsBackend()
        
        if isinstance(config, Config):
            self.cfg = config
        else:
            self.cfg = Config.fromfile(config)

        self.models = []
        model_cfg = self.cfg.model
        
        if isinstance(model_cfg, dict):
            model_cfg = [model_cfg]

        for cfg in model_cfg:
            model = self._initialize_model(cfg)
            if model is not None:
                self.models.append(model)

        self.preprocess = self.cfg.preprocess
        self.postprocess = self.cfg.postprocess
        self.spectra = None
        self.is_loaded = False
        
        # Timing statistics
        self.timing = {
            'preprocess': 0.0,
            'detection': 0.0,
            'impic': 0.0
        }

    def _initialize_model(self, config: Dict) -> Optional[torch.nn.Module]:
        """Initialize a detection model from configuration."""
        model_cfg = config.model_config
        checkpoint = config.checkpoint
        dev = config.get("device", torch.cuda.current_device())

        try:
            model = init_detector(model_cfg, checkpoint, dev)
        except (Exception, AssertionError) as e:
            print(e)
            print(f"Failed in loading model from:\n config={model_cfg} \n checkpoint={checkpoint}")
            model = None

        return model

    def _extract_test_pipeline(self, model_cfg: Config) -> List:
        """Extract test pipeline configuration from model config."""
        try:
            if hasattr(model_cfg, 'test_dataloader'):
                return model_cfg.test_dataloader.dataset.pipeline
            elif hasattr(model_cfg, 'test_dataset'):
                return model_cfg.test_dataset.pipeline
            elif hasattr(model_cfg, 'data') and hasattr(model_cfg.data, 'test'):
                return model_cfg.data.test.pipeline
            else:
                return []
        except AttributeError:
            return []

    def _ransac_curve_fit(
            self,
            func: Callable,
            points: np.ndarray,
            eval_points: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Fit curve using RANSAC approach."""
        delta_t = points[:, 0]
        delta_f = points[:, 1]

        eval_delta_t = eval_points[:, 0]
        eval_delta_f = eval_points[:, 1]

        popt, pcov = curve_fit(func, xdata=delta_f, ydata=delta_t)
        pred = func(eval_delta_f, *popt)

        sst = np.sum((eval_delta_t - np.mean(eval_delta_t)) ** 2)
        sse = np.sum((eval_delta_t - pred) ** 2)
        fitness = max(1.0 - sse / sst, 0.0)

        return popt, fitness

    def compute_dispersive_delay(self, f1: float, f2: float, dm: float) -> float:
        """Calculate dispersive time delay between two frequencies."""
        return (1.0 / f1 ** 2 - 1.0 / f2 ** 2) * DISPERSION_CONSTANT * dm / 1000.0

    def IMPIC(self, masks: np.ndarray, ransac_cfg: Dict = None) -> np.ndarray:
        """Iterative Mask-based  Parameter Inference and Calibration."""
        num_sample = ransac_cfg.get("sample_points")
        iters = ransac_cfg.get("iterations")
        fit_pair = ransac_cfg.get("fit_pair", True)

        output = np.zeros((masks.shape[0], 2))

        for midx, mask in enumerate(masks):
            assert len(mask.shape) == 2, f"{mask.shape}"
            vf, vt = np.where(mask.astype(bool))
            if vt.shape[0] == 0:
                continue
           
            vt = vt.astype(np.float32) * self.preprocess.downsample_time * self.sampling_interval
            vf = vf.astype(np.float32) * self.preprocess.downsample_freq * self.channel_bandwidth
            vf = self.preprocess.freq_range[-1] - vf
            
            if fit_pair:
                threshold = (np.max(vt) - np.min(vt)) / 2
                difft = (vt[np.newaxis, :] - vt[:, np.newaxis]).flatten()
                difff = (1.0 / vf[np.newaxis, :] ** 2 - 1.0 / vf[:, np.newaxis] ** 2).flatten()

                selected_pairs = (np.abs(difft) >= threshold)
                difft = difft[selected_pairs]
                difff = difff[selected_pairs]

                func = lambda x, dm: x * DISPERSION_CONSTANT * dm / 1000.0
            else:
                difft = vt
                difff = (1.0 / vf ** 2 - 1.0 / self.preprocess.freq_range[-1] ** 2)
                func = lambda x, dm, toa: x * DISPERSION_CONSTANT * dm / 1000.0 + toa

            points = np.stack([difft, difff], axis=1)
            fitness = float("-inf")
            param = None
        
            for _ in range(iters):
                npoints = points.shape[0]
                npoints = min(npoints, num_sample)
                select_index = random.sample(list(range(points.shape[0])), k=npoints)
                fit_points = points[select_index]

                cur_param, cur_fitness = self._ransac_curve_fit(func, fit_points, points)

                if cur_fitness >= fitness:
                    param = cur_param
                    fitness = cur_fitness

            if param is None:
                continue

            if fit_pair:
                dm = param
                mintime_index = np.argmin(vt)
                mintime, minfreq = vt[mintime_index], vf[mintime_index]
                delta_t = self.compute_dispersive_delay(minfreq, self.preprocess.freq_range[-1], dm)
                toa = mintime - delta_t
            else:
                dm, toa = param

            output[midx, 0] = max(toa, 0.0)
            output[midx, 1] = dm
            
        return output

    def transfer_to_device(self, device: Union[str, int]) -> None:
        """Transfer all models to specified compute device."""
        for model in self.models:
            model.to(device)

    def release_memory(self) -> None:
        """Release all cached data to free memory."""
        attrs_to_clear = ['spectra', 'observation_file']
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_spectra(self) -> None:
        """Release only spectral data while preserving metadata."""
        if hasattr(self, 'spectra'):
            self.spectra = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_observation(self, filepath: str, slide_size: Optional[int] = None) -> bool:
        """Load radio observation file (PSRFITS or Filterbank)."""
        self.observation_path = filepath

        observation_file = self.backend.get(filepath)
        if observation_file is None:
            raise IOError(f"Failed loading file {filepath}")

        self.file_extension = observation_file.ext
        self.observation_file = observation_file
        self.channel_bandwidth = observation_file.df
        self.sampling_interval = observation_file.tsamp
        self.samples_per_subint = observation_file.nsamp_per_subint
        
        if slide_size is None or slide_size == -1:
            self.spectra = self.observation_file.get_spectra_all()
            print("Loaded full spectra data (slide_size is None)")
            
        self.num_channels = observation_file.nchan
        self.total_samples = observation_file.nspec
        self.frequency_channels = observation_file.freqs
        self.is_loaded = True

        return True

    def fetch_raw_data(self) -> Optional[np.ndarray]:
        """Retrieve raw spectral data array."""
        return None if self.spectra is None else self.spectra.data

    def fetch_processed_data(self, apply_clipping: bool = True) -> Optional[np.ndarray]:
        """Get preprocessed spectral data ready for inference."""
        if self.spectra is None:
            return None
        
        # Start preprocess timing
        t_preprocess_start = time.time()
            
        try:
            data = copy.deepcopy(self.spectra)

            shape = data.data.shape
            assert len(shape) == 2, f"{len(shape)} vs. 2"
            print("raw data shape:", data.data.shape)

            nsub = self.num_channels // self.preprocess.downsample_freq
            if nsub is not None:
                data.subband(nsub, padval='mean')

            data.downsample(self.preprocess.downsample_time)
            data.mask_baseband_and_scale(basebandStd=1.0, indep=False)
            data.exp_normalization_plus()

            processed_data = data.data

            if apply_clipping:
                processed_data = dynamic_correct(processed_data, scaling=self.preprocess.scaling)

            print("Preprocessed data shape:", processed_data.shape)

            del data
            gc.collect()

            # End preprocess timing
            self.timing['preprocess'] = time.time() - t_preprocess_start

            return processed_data

        except Exception as e:
            print(f"Error in fetch_processed_data: {e}")
            if 'data' in locals():
                del data
            gc.collect()
            return None

    def get_results(self, coordinate_mapping: bool = True, slide_size: int = -1) -> Optional[np.ndarray]:
        """Run detection pipeline on loaded observation."""
        if slide_size < 0 or slide_size is None:
            results = self._detect_single_window(coordinate_mapping=coordinate_mapping)
        else:
            step_samp = slide_size * self.samples_per_subint
            step_time = step_samp * self.sampling_interval
            nsamp = self.total_samples

            results = []
            index = 0
            start_time = 0

            while index < nsamp:
                if nsamp < 1.5 * step_samp + index:
                    end_index = nsamp
                else:
                    end_index = index + step_samp

                if self.file_extension.endswith('.fil'):
                    print(f"start_timesamp: {index}, end_timesamp: {end_index}")
                    self.spectra = self.observation_file.get_spectra_slide(start_time, index, end_index)
                elif self.file_extension.endswith('.fits'):
                    start_subint = int(index // self.samples_per_subint)
                    end_subint = min(int(end_index // self.samples_per_subint), int(self.observation_file.nsubints - 1))
                    print(f"start_subint: {start_subint}, end_subint: {end_subint}")
                    self.spectra = self.observation_file.get_spectra_slide(start_time, start_subint, end_subint)
                else:
                    raise ValueError(f"Unknown file type: {self.file_extension}")

                res = self._detect_single_window(coordinate_mapping=coordinate_mapping, window_offset=start_time)
                
                if res is not None:
                    if coordinate_mapping:
                        res[:, 0] += start_time
                    results.append(res)

                index = end_index
                start_time += step_time

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if coordinate_mapping and len(results) > 0:
                results = np.concatenate(results, axis=0)

            self.release_memory()
            gc.collect()

        return results

    def _detect_single_window(self, coordinate_mapping: bool = True, window_offset: float = 0) -> Optional[np.ndarray]:
        """Execute detection on a single data window."""
        if self.fetch_raw_data() is None or len(self.models) == 0:
            return None

        try:
            processed_fits_data = self.fetch_processed_data(apply_clipping=True)
            if processed_fits_data is None:
                return None
                
            processed_fits_data = processed_fits_data[:, :, np.newaxis]

            all_model_box = []
            all_model_mask = []

            # Start detection timing
            t_detection_start = time.time()
            
            for model in self.models:
                try:
                    model_cfg = model.cfg
                    test_pipeline = self._extract_test_pipeline(model_cfg)
                    
                    if ops_in_ppl(test_pipeline, ["DynamicCorrect"]):
                        inp = processed_fits_data
                    else:
                        inp = dynamic_correct(processed_fits_data)

                    result = inference_detector(model, inp)

                    if inp is not processed_fits_data:
                        del inp

                    if hasattr(result, 'pred_instances'):
                        pred_instances = result.pred_instances
                    elif isinstance(result, list) and len(result) > 0:
                        pred_instances = result[0].pred_instances
                    else:
                        continue

                    if hasattr(pred_instances, 'bboxes') and hasattr(pred_instances, 'scores'):
                        bboxes = pred_instances.bboxes.detach().cpu().numpy()
                        scores = pred_instances.scores.detach().cpu().numpy()
                        bbox = np.concatenate([bboxes, scores.reshape(-1, 1)], axis=1)
                    else:
                        continue

                    if hasattr(pred_instances, 'masks') and pred_instances.masks is not None:
                        masks_tensor = pred_instances.masks
                        if hasattr(masks_tensor, 'detach'):
                            masks = masks_tensor.detach().cpu().numpy()
                        else:
                            masks = masks_tensor
                            
                        if len(masks.shape) == 3:
                            masks = (masks > 0.5).astype(np.uint8)
                            masks = [masks[i] for i in range(masks.shape[0])]
                        elif len(masks.shape) == 2:
                            masks = [(masks > 0.5).astype(np.uint8)]
                        else:
                            continue
                    else:
                        continue

                    print(f"Detected {len(bbox)} objects with masks")

                    if bbox.shape[0] == 0:
                        continue

                    if isinstance(masks, list):
                        masks = np.stack(masks, axis=0)

                    index = (bbox[:, -1] >= self.postprocess.threshold)
                    bbox = bbox[index]
                    masks = masks[index]

                    all_model_box.append(bbox)
                    all_model_mask.append(masks)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing model: {e}")
                    continue

            del processed_fits_data

            if len(all_model_box) > 0:
                bbox = np.concatenate(all_model_box, axis=0)
                masks = np.concatenate(all_model_mask, axis=0)
            else:
                return None

            if bbox.shape[0] == 0:
                return None

            nms_cfg = self.postprocess.nms_cfg
            bbox, masks = cpu_nms(bbox=bbox, masks=masks, nms_cfg=nms_cfg)

            # End detection timing
            self.timing['detection'] = time.time() - t_detection_start
            
            self.release_spectra()

            if not coordinate_mapping:
                return bbox, masks

            for name in self.observation_params:
                if not hasattr(self, name):
                    return None

            # Start IMPIC timing
            t_impic_start = time.time()
            fit_cfg = self.postprocess["mapping"].copy()
            toa_dms = self.IMPIC(masks, **fit_cfg)
            self.timing['impic'] = time.time() - t_impic_start

            scores = bbox[:, -1:]
            toa_dms = np.concatenate([toa_dms, scores], axis=1)

            aug_cfg = self.postprocess.get("aug_cfg", None)
            if aug_cfg:
                cfg = aug_cfg.copy()
                name = cfg.pop("type")
                # Map function name to the actual function
                aug_func_map = {
                    'dm_filtering': dm_filtering,
                    'dm_shift': dm_shift,
                    'toa_shift': toa_shift,
                    'entity': entity
                }
                aug_func = aug_func_map.get(name, None)
                if aug_func is not None:
                    toa_dms = aug_func(toa_dms, **cfg)

            return toa_dms

        except Exception as e:
            print(f"Error in _detect_single_window: {e}")
            self.release_spectra()
            gc.collect()
            return None