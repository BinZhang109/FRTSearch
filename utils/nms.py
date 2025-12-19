import numpy as np
import pycocotools.mask as maskUtil
from mmengine.config import Config, ConfigDict
from typing import List
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