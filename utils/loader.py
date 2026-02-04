"""
Author: Bin Zhang
Date: 2025.8.11
"""

import os
import os.path as osp
import numpy as np
import tqdm
import mmcv
from functools import partial
from mmdet.registry import DATASETS, TRANSFORMS
from mmdet.datasets import CocoDataset
from mmdet.structures.bbox import bbox_overlaps
from mmengine.logging import print_log
from mmcv.transforms import BaseTransform  
from pycocotools import mask as maskutil
import pycocotools.mask as maskUtil
from mmengine.config import Config, ConfigDict
from typing import List, Union

def encode_masks_to_rle(masks):
    if masks is None or len(masks) == 0:
        return []
    
    encoded_masks = []
    for mask in masks:
        if isinstance(mask, dict) and 'counts' in mask:
            encoded_masks.append(mask)
        else:
            if hasattr(mask, 'detach'):
                mask = mask.detach().cpu().numpy()
            elif hasattr(mask, 'numpy'):
                mask = mask.numpy()
            if len(mask.shape) == 2:
                mask = mask.astype(np.uint8)
                mask = np.asfortranarray(mask)
                rle = maskUtil.encode(mask)
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('utf-8')
                encoded_masks.append(rle)
            else:
                print(f"Warning: Unexpected mask shape {mask.shape}, skipping")
                continue
                
    return encoded_masks

def cpu_nms(
        bbox : np.ndarray,
        masks : np.ndarray,
        nms_cfg : ConfigDict):

    bbox = bbox.copy()

    num_proposal = bbox.shape[0]
    max_cand = nms_cfg.max_candidates
    thrsh = nms_cfg.iou_threshold

    if num_proposal <= max_cand:
        return bbox, masks

    score = bbox[:, -1]
    index = np.argsort(-1.0 * score)
    bbox = bbox[index]
    masks = masks[index]
    try:
        encoded_mask = encode_masks_to_rle(masks)
        
        if len(encoded_mask) == 0:
            print("Warning: No valid masks for NMS, returning top candidates by score")
            left_index = np.arange(min(max_cand, num_proposal))
            return bbox[left_index], masks[left_index]
        
        if len(encoded_mask) != len(masks):
            print(f"Warning: Encoded {len(encoded_mask)} masks but expected {len(masks)}")
            min_len = min(len(encoded_mask), len(masks))
            encoded_mask = encoded_mask[:min_len]
            bbox = bbox[:min_len]
            masks = masks[:min_len]
            num_proposal = min_len
        
        iscrowd = [0 for _ in encoded_mask]
        ious = maskUtil.iou(encoded_mask, encoded_mask, iscrowd)

        indexes = np.linspace(0, num_proposal - 1, num_proposal)
        for i in range(num_proposal):
            if bbox[i, -1] == 0.0:
                continue

            overlap = ious[i]
            index = np.where((overlap >= thrsh) & (indexes > i))[0]
            bbox[index, -1] = 0.0

        left_index = np.where(bbox[:, -1] > 0.0)[0]
        if left_index.shape[0] >= max_cand:
            left_index = left_index[:max_cand]

        return bbox[left_index], masks[left_index]
        
    except Exception as e:
        print(f"Error in NMS: {e}")
        print("Falling back to score-based selection")
        left_index = np.arange(min(max_cand, num_proposal))
        return bbox[left_index], masks[left_index]

def dynamic_correct(
        data : np.ndarray,
        scaling : float = 0.8,
        lower_size : float =1.5,
        upper_size : float =3.0
    ):

    plotDataFGAvg = np.mean(data)
    plotDataFGstd = np.std(data)
    plotVmin = plotDataFGAvg - lower_size * plotDataFGstd * scaling
    plotVmax = plotDataFGAvg + upper_size * plotDataFGstd * scaling

    output = np.clip(data, a_min=plotVmin, a_max=plotVmax)
    output = (output - plotVmin) / (plotVmax - plotVmin)

    return output

def ops_in_ppl(ppl_cfg : List[Config], query_ops : List[str]):

    for cfg in ppl_cfg:
        if cfg.type in query_ops:
            return True

    return False


def dm_filtering(
        toa_dm : np.ndarray,
        *,
        threshold : float,
    ):

    if toa_dm.shape[0] == 0:
        return toa_dm

    dm_flag = toa_dm[:, 1] >= threshold
    filtered_samples = toa_dm[dm_flag]

    return filtered_samples


def dm_shift(
        toa_dm : np.ndarray,
        *,
        shifts : Union[float, List[float]],
        trigger_dm : float = -1,
        min_dm : float = 0.0,
        max_dm : float = 6000.0
    ):

    if toa_dm.shape[0] == 0:
        return toa_dm

    if isinstance(shifts, float):
        shifts = [shifts]

    triggered_index = toa_dm[:, 1] >= trigger_dm
    augment_samples = toa_dm[triggered_index]

    if augment_samples.shape[0] == 0:
        return toa_dm

    output = [toa_dm]

    for shift in shifts:
        new_sample = augment_samples.copy()
        new_dm = augment_samples[:, 1] + shift
        new_dm = np.clip(new_dm, a_min=min_dm, a_max=max_dm)
        new_sample[:, 1] = new_dm

        output.append(new_sample)

    return np.concatenate(output, axis=0)


def toa_shift(
        toa_dm: np.ndarray,
        *,
        shifts: Union[float, List[float]],
        min_toa: float = 0.0,
        max_toa: float = 12.88
):
    if toa_dm.shape[0] == 0:
        return toa_dm

    if isinstance(shifts, float):
        shifts = [shifts]

    augment_samples = toa_dm

    if augment_samples.shape[0] == 0:
        return toa_dm

    output = [toa_dm]

    for shift in shifts:
        new_sample = augment_samples.copy()
        new_toa = augment_samples[:, 0] + shift
        new_toa = np.clip(new_toa, a_min=min_toa, a_max=max_toa)
        new_sample[:, 0] = new_toa

        output.append(new_sample)

    return np.concatenate(output, axis=0)


def entity(toa_dm : np.ndarray):

    return toa_dm


def mask_overlaps(mask1: np.ndarray, mask2: np.ndarray):
    n1 = mask1.shape[0]
    n2 = mask2.shape[0]

    mask1 = np.reshape(mask1, [n1, -1])
    mask2 = np.reshape(mask2, [n2, -1]).transpose([1, 0])

    intersect = np.matmul(mask1, mask2)
    union = np.sum(mask1, axis=1)[:, np.newaxis] + np.sum(mask2, axis=0)[np.newaxis, :]

    iou = intersect / union

    return iou


@DATASETS.register_module()
class FastData(CocoDataset):
    def __init__(self, *args, **kwargs):
        super(FastData, self).__init__(*args, **kwargs)
        
    def evaluate_det_segm(self, results, **kwargs):
        img_ids = self.coco.get_img_ids()
        num_img = len(results)
        
        assert len(img_ids) == num_img, f"{len(img_ids)} vs {num_img}"
        
        labeled_positive = 0
        recall_image = 0
        fp = 0
        
        positive_iou = kwargs.get("positive_iou", 0.1)
        
        for i in tqdm.tqdm(range(num_img)):
            img_id = img_ids[i]
            annoIds = self.coco.get_ann_ids(img_ids=img_id)
            annos = self.coco.loadAnns(annoIds)
            img = self.coco.loadImgs(img_id)[0]
            
            if len(annos) > 0:
                labeled_positive += 1
            
            if isinstance(results[i], dict):
                res = results[i].get('pred_instances', {}).get('masks', [])
            else:
                res = results[i][1][0] if len(results[i]) > 1 else []
            
            if len(annos) > 0 and len(res) > 0:
                maskgts = [anno["segmentation"] for anno in annos]
                iscrowd = [anno["iscrowd"] for anno in annos]
                iou = maskutil.iou(res, maskgts, iscrowd)
                
                iou = np.max(iou, axis=1)
                fp += np.sum((iou < positive_iou))
                recall_image += float(np.max(iou) >= positive_iou)
                if np.max(iou) >= positive_iou:
                    print(img["file_name"])
            else:
                fp += len(res)
        
        if labeled_positive > 0:
            image_level_recall = float(recall_image) / labeled_positive
        else:
            image_level_recall = float("nan")
        
        fp_per_image = float(fp) / num_img
        
        eval_results = {
            f"image_level_recall@{positive_iou:.2f}": image_level_recall,
            f"false_positive_per_image@{positive_iou:.2f}": fp_per_image
        }
        
        return eval_results


@TRANSFORMS.register_module()
class LoadImageFromNumpy(BaseTransform):
    """Load an image from numpy file."""
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32
        self.loader = partial(np.load, allow_pickle=True)

    def transform(self, results):
        """Transform function to load image from numpy file."""
        
        filename = None
        if 'img_path' in results:
            filename = results['img_path']
        elif 'filename' in results:
            filename = results['filename']
        elif 'img_info' in results:
            img_prefix = results.get('img_prefix', '')
            filename = osp.join(img_prefix, results['img_info']['filename'])
        else:
            img_prefix = results.get('img_prefix', '')
            if 'ori_filename' in results:
                filename = osp.join(img_prefix, results['ori_filename'])

        if filename is None:
            raise ValueError("Cannot find image filename in results")

 
        try:
            img = self.loader(filename)
            if self.to_float32:
                img = img.astype(np.float32)
            
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Loaded image should be numpy array, got {type(img)}")
      
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            raise
        
        results['img'] = img
        results['img_path'] = filename
        results['ori_filename'] = osp.basename(filename)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1.0, 1.0], dtype=np.float32)
        if 'img_norm_cfg' not in results:
            results['img_norm_cfg'] = dict(mean=[0.0], std=[1.0], to_rgb=False)
        
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(to_float32={self.to_float32})'

@TRANSFORMS.register_module()
class DynamicCorrect(BaseTransform):
    """Correct the dynamic numerical range."""
    def __init__(self, factors=0.8):
        if isinstance(factors, (list, tuple)):
            self.factors = list(factors)
        else:
            self.factors = [factors]

    def transform(self, results):
        """Transform function to correct dynamic range."""
        img = results['img']
        
    
        if not isinstance(img, np.ndarray):
            raise ValueError(f"Image should be numpy array, got {type(img)}")
        
        corrected = []
        for s in self.factors:
            corrected_img = dynamic_correct(img, scaling=s)
            corrected.append(corrected_img)
        
        
        if len(corrected) == 1:
            output = corrected[0]
        else:
            output = np.stack(corrected, axis=-1)
        
        results['img'] = output
        results['img_shape'] = output.shape[:2]
        
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(factors={self.factors})'