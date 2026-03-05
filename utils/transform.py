"""
Author: Bin Zhang
Date: 2025.08.23
"""
import random
import numpy as np
import cv2
import mmcv

from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks
from collections import deque


@TRANSFORMS.register_module()
class CropAndPaste(object):

    """
    Randomly crop a positive patch from data and paste them on other images
    with random resize and color decay
    """

    def __init__(
            self,
            queue_size = 50,
            prob = 0.5,
            max_crop_per_img = 3,
            max_paste_per_img = 3,
            scale_factor = (0.5, 1.0),
            contrast_factor = (0.5, 1.5),
            ):

        self.queue = deque(maxlen=queue_size)
        if scale_factor is None:
            self.scale_factor = (0.5, 1.0)
        else:
            assert isinstance(scale_factor, (list, tuple))
            assert len(scale_factor) == 2
            self.scale_factor = scale_factor

        assert isinstance(contrast_factor, (float, list, tuple))
        self.contrast_factor = contrast_factor
        self.prob = prob
        self.max_crop_per_img = max_crop_per_img
        self.max_paste_per_img = max_paste_per_img

    def _get_gt_data(self, results):
        """Get ground truth data safely."""
        bboxes = None
        masks = None
        labels = None
        
        if "gt_instances" in results and results["gt_instances"] is not None:
            gt_instances = results["gt_instances"]
            if hasattr(gt_instances, 'bboxes') and gt_instances.bboxes is not None:
                bboxes = gt_instances.bboxes
            if hasattr(gt_instances, 'masks') and gt_instances.masks is not None:
                masks = gt_instances.masks
            if hasattr(gt_instances, 'labels') and gt_instances.labels is not None:
                labels = gt_instances.labels
        
        if bboxes is None and "gt_bboxes" in results:
            bboxes = results["gt_bboxes"]
        if masks is None and "gt_masks" in results:
            masks = results["gt_masks"]
        if labels is None and "gt_labels" in results:
            labels = results["gt_labels"]
            
        return bboxes, masks, labels

    def _crop_patch(self, results):
        img = results["img"]
        
        bboxes, masks, labels = self._get_gt_data(results)
        
        if bboxes is None or masks is None or labels is None:
            return
            
        if hasattr(bboxes, 'tensor'):
            bboxes = bboxes.tensor.cpu().numpy()
        elif hasattr(bboxes, 'numpy'):
            bboxes = bboxes.numpy()
        elif not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)
        
        if hasattr(labels, 'tensor'):
            labels = labels.tensor.cpu().numpy()
        elif hasattr(labels, 'numpy'):
            labels = labels.numpy()
        elif not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        if len(bboxes) == 0 or len(labels) == 0:
            return
            
        if len(bboxes) != len(labels):
            return

        num_patch = bboxes.shape[0]
        num_patch = min(num_patch, self.max_crop_per_img)
        if num_patch == 0:
            return
            
        indexes = random.sample(range(bboxes.shape[0]), k=num_patch)
        img_h, img_w = img.shape[:2]

        for index in indexes:
            try:
                box = bboxes[index]
                box = box.copy()
                box[0::2] = np.clip(box[0::2], 0, img_w)
                box[1::2] = np.clip(box[1::2], 0, img_h)
                xmin, ymin, xmax, ymax = [int(v) for v in box]

                if xmax <= xmin or ymax <= ymin:
                    continue

                patch = img[ymin:ymax, xmin:xmax].copy()
                if patch.size == 0:
                    continue
                    
                label = np.array(labels[index])

                mask = masks[index].crop(bbox=box.astype(np.int64))
                self.queue.append(
                    dict(patch=patch, label=label, mask=mask)
                )
            except Exception as e:
                continue

    def _render_patch(self, results):
        if len(self.queue) == 0:
            return
            
        num_patch = min(
            len(self.queue), np.random.randint(1, self.max_paste_per_img+1))

        img = results["img"].copy()
        img_h, img_w = img.shape[:2]

        new_masks = []
        new_boxes = []
        new_labels = []

        for i in range(num_patch):
            try:
                patch_item = random.choice(self.queue)
                patch = patch_item["patch"]
                mask = patch_item["mask"]
                label = patch_item["label"]

                ph, pw = patch.shape[:2]
                if ph == 0 or pw == 0:
                    continue
                    
                scale_h = np.random.uniform(*self.scale_factor)
                scale_w = np.random.uniform(*self.scale_factor)

                ph = min(int(ph * scale_h), img_h)
                pw = min(int(pw * scale_w), img_w)
                
                if ph == 0 or pw == 0:
                    continue

                try:
                    patch = mmcv.imresize(patch, size=(pw, ph))
                except:
                    patch = cv2.resize(patch, (pw, ph))
                    
                mask = mask.resize(out_shape=(ph, pw))

                color_decay = np.random.uniform(*self.contrast_factor)
                patch = patch * color_decay

                if img_w - pw <= 0 or img_h - ph <= 0:
                    continue
                    
                pos_x = np.random.randint(0, img_w - pw + 1)
                pos_y = np.random.randint(0, img_h - ph + 1)

                canvas = img[pos_y:pos_y+ph, pos_x:pos_x+pw]
                bitmask = mask.to_ndarray()[0]

                if len(canvas.shape) == 3:
                    img_bitmask = np.expand_dims(bitmask, axis=2)
                else:
                    img_bitmask = bitmask

                if len(patch.shape) < len(canvas.shape):
                    patch = np.expand_dims(patch, axis=-1)

                img[pos_y:pos_y + ph, pos_x:pos_x + pw] = canvas * (1 - img_bitmask) + img_bitmask * patch
                empty_mask = np.zeros((img_h, img_w)).astype(np.uint8)
                empty_mask[pos_y:pos_y + ph, pos_x:pos_x + pw] = bitmask

                new_masks.append(empty_mask)
                new_boxes.append(np.array([pos_x, pos_y, pos_x+pw, pos_y+ph], dtype=np.float32))
                new_labels.append(label)
                
            except Exception as e:
                continue

        if len(new_boxes) > 0:
            try:
                new_boxes = np.array(new_boxes)
                new_labels = np.array(new_labels, dtype=np.int64)
                new_masks = np.stack(new_masks, axis=0)

                bboxes, masks, labels = self._get_gt_data(results)
                
                if bboxes is not None and masks is not None and labels is not None:
                    if hasattr(bboxes, 'tensor'):
                        old_bboxes = bboxes.tensor.cpu().numpy()
                    elif hasattr(bboxes, 'numpy'):
                        old_bboxes = bboxes.numpy()
                    else:
                        old_bboxes = np.array(bboxes)
                        
                    if hasattr(labels, 'tensor'):
                        old_labels = labels.tensor.cpu().numpy()
                    elif hasattr(labels, 'numpy'):
                        old_labels = labels.numpy()
                    else:
                        old_labels = np.array(labels)

                    old_masks = masks.to_ndarray()

                    combined_bboxes = np.concatenate([old_bboxes, new_boxes], axis=0)
                    combined_labels = np.concatenate([old_labels, new_labels], axis=0)
                    combined_masks = np.concatenate([old_masks, new_masks], axis=0)

                    if "gt_instances" in results:
                        import torch
                        gt_instances = results["gt_instances"]
                        gt_instances.bboxes = torch.from_numpy(combined_bboxes).float()
                        gt_instances.labels = torch.from_numpy(combined_labels).long()
                        gt_instances.masks = BitmapMasks(combined_masks, img_h, img_w)
                    else:
                        results["gt_bboxes"] = combined_bboxes
                        results["gt_labels"] = combined_labels
                        results["gt_masks"] = BitmapMasks(combined_masks, img_h, img_w)
                        
            except Exception as e:
                pass

        results["img"] = img

    def __call__(self, results):
        try:
            if results.get('test_mode', False):
                return results
                
            bboxes, masks, labels = self._get_gt_data(results)
            
            if bboxes is None or masks is None or labels is None:
                return results

            self._crop_patch(results)

            if len(self.queue) == 0 or np.random.uniform(0.0, 1.0) >= self.prob:
                return results

            self._render_patch(results)

        except Exception as e:
            pass

        return results
