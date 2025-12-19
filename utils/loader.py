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
from .nms import dynamic_correct
from pycocotools import mask as maskutil


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