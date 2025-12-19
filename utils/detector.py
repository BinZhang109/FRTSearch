import os.path as osp
import numpy as np
import random
import torch
import utils.augmentation as augs
from typing import Union, List, Dict, Tuple, Callable
from mmdet.apis.inference import inference_detector, init_detector
from mmengine.fileio import FileClient  
from mmengine.config import Config      
from mmengine.logging import print_log  
import copy
from scipy.optimize import curve_fit
import gc
from .nms import cpu_nms,dynamic_correct,ops_in_ppl
from .backend import FitsBackend

DM_CONSTANT = 4148808.0


class FitsDetector(object):
    fits_param_names = ["tsamp", "fchannel", "df"]

    def __init__(
            self,
            config: Union[str, Config],
    ):
        self.client = FitsBackend()
        if isinstance(config, Config):
            self.cfg = config
        else:
            self.cfg = Config.fromfile(config)

        self.models = list()
        model_cfg = self.cfg.model
        if isinstance(model_cfg, dict):
            model_cfg = [model_cfg]

        for cfg in model_cfg:
            model = self.build_model(cfg)
            if model is not None:
                self.models.append(model)

        self.preprocess = self.cfg.preprocess
        self.postprocess = self.cfg.postprocess
        # 初始化为None
        self.spectra = None
        self.loaded = False

    def build_model(self, config):

        model_cfg = config.model_config
        checkpoint = config.checkpoint
        dev = config.get(
            "device", torch.cuda.current_device())

        try:
            model = init_detector(model_cfg, checkpoint, dev)
        except (Exception, AssertionError) as e:
            print(e)
            print(f"Failed in loading model from:\n config={model_cfg} \n checkpoint={checkpoint}")
            model = None

        return model

    def to_device(self, device: Union[str, int]):

        for model in self.models:
            model.to(device)

    def clear(self):
        """Clear data attributes to free memory."""
        # 清理所有可能占用内存的属性
        attrs_to_clear = ['spectra', 'rawdatafile']
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                setattr(self, attr, None)
        
        # 强制垃圾回收
        gc.collect()
        # 清理GPU缓存（如果使用GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def clear_spectra_only(self):
        """只清理spectra数据，保留其他必要的元数据"""
        if hasattr(self, 'spectra'):
            self.spectra = None
        
        # 轻量级垃圾回收
        gc.collect()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def set_fits_params(self, **kwargs):

        for key, val in kwargs.items():
            if key in self.fits_param_names:
                setattr(self, key, val)

    def get_time_delay(self, f1, f2, dm):

        """
        :param f1: frequency1 unit: MHz
        :param f2: frequency2 unit: MHz
        :param dm: dm factor
        :return: delta time unit: s
        """

        return (1.0 / f1 ** 2 - 1.0 / f2 ** 2) * DM_CONSTANT * dm / 1000.0

    def _successful_loading_guard(self):

        if not self.loaded:
            raise RuntimeError("No data is loaded into detector")

    def load_fits(self, filepath: str,slide_size=None):

        self.fits_name = filepath

        rawdatafile = self.client.get(filepath)
        if rawdatafile is None:
            raise IOError(f"Failed loading file {filepath}")

        self.ext=rawdatafile.ext
        self.rawdatafile = rawdatafile
        self.df = rawdatafile.df #要用psrfits和filterbank类取绝对值后的df
        self.tsamp = rawdatafile.tsamp
        self.samppersubint = rawdatafile.nsamp_per_subint
        if slide_size is None or slide_size == -1:  # 考虑到您的代码中使用-1作为默认值:
            self.spectra = self.rawdatafile.get_spectra_all() # 文件太大，显存读取会kill,就选择slide_Size
            print("Loaded full spectra data (slide_size is None)")
        self.obsnchan = rawdatafile.nchan
        self.nsamp = rawdatafile.nspec
        self.fchannel = rawdatafile.freqs
        self.loaded = True
    
        
        return True

    def _fit_points(
            self,
            func: Callable,
            points: np.ndarray,
            eval_points: np.ndarray
    ):

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

    def mask2ft(self, masks: np.ndarray, ransac_cfg: Dict = None):

        num_sample = ransac_cfg.get("sample_points")
        iters = ransac_cfg.get("iterations")
        fit_pair = ransac_cfg.get("fit_pair", True)

        output = np.zeros((masks.shape[0], 2))

        for midx, mask in enumerate(masks):
            assert len(mask.shape) == 2, f"{mask.shape}"
            vf, vt = np.where(mask.astype(bool))
            if vt.shape[0] == 0:
                continue
           
            vt = vt.astype(np.float32) * self.preprocess.downsample_time * self.tsamp
            vf = vf.astype(np.float32) * self.preprocess.downsample_freq * self.df

            vf = self.preprocess.freq_range[-1]-vf
            
          
            if fit_pair:
                threshold = (np.max(vt) - np.min(vt)) / 2

                difft = (vt[np.newaxis, :] - vt[:, np.newaxis]).flatten()
                difff = (1.0 / vf[np.newaxis, :] ** 2 - 1.0 / vf[:, np.newaxis] ** 2).flatten()

                selected_pairs = (np.abs(difft) >= threshold)
                difft = difft[selected_pairs]
                difff = difff[selected_pairs]

                func = lambda x, dm: x * DM_CONSTANT * dm / 1000.0
            else:
                difft = vt
                difff = (1.0 / vf ** 2 - 1.0 / self.preprocess.freq_range[-1] ** 2)
                #注意这里推最高频率的到达时间，使用config的最高频率
                #fits和fil的最高频率是反的
                func = lambda x, dm, toa: x * DM_CONSTANT * dm / 1000.0 + toa

            points = np.stack([difft, difff], axis=1)
            fitness = float("-inf")
            param = None
        
            for _ in range(iters):
                npoints = points.shape[0]
                npoints = min(npoints, num_sample)
                select_index = random.sample(list(range(points.shape[0])), k=npoints)
                fit_points = points[select_index]

                cur_param, cur_fitness = self._fit_points(func, fit_points, points)

                if cur_fitness >= fitness:
                    param = cur_param
                    fitness = cur_fitness

            if param is None:
                continue

            if fit_pair:
                dm = param
                mintime_index = np.argmin(vt)
                mintime, minfreq = vt[mintime_index], vf[mintime_index]
                delta_t = self.get_time_delay(minfreq, self.preprocess.freq_range[-1], dm)
                toa = mintime - delta_t
            else:
                dm, toa = param

            output[midx, 0] = max(toa, 0.0)
            output[midx, 1] = dm
        return output

    def _get_test_pipeline(self, model_cfg):
        """获取测试pipeline，兼容MMDet2和MMDet3的配置结构"""
        try:
            # MMDet3 方式
            if hasattr(model_cfg, 'test_dataloader'):
                return model_cfg.test_dataloader.dataset.pipeline
            elif hasattr(model_cfg, 'test_dataset'):
                return model_cfg.test_dataset.pipeline
            # MMDet2 方式（向后兼容）
            elif hasattr(model_cfg, 'data') and hasattr(model_cfg.data, 'test'):
                return model_cfg.data.test.pipeline
            else:
                # 如果都没有，返回空列表
                print("Warning: Cannot find test pipeline in model config")
                return []
        except AttributeError:
            print("Warning: Error accessing test pipeline, using default behavior")
            return []
        
        
    def get_results(self, coordinate_mapping: bool = True, slide_size: int = -1):

        if slide_size < 0 or slide_size is None:
            results = self.get_results_single(coordinate_mapping=coordinate_mapping)
        else:

            step_samp = slide_size * self.samppersubint
            step_time = step_samp * self.tsamp
            nsamp = self.nsamp

            results = []
            index = 0
            start_time = 0

            while index < nsamp:
                if nsamp < 1.5 * step_samp + index:
                    end_index = nsamp
                else:
                    end_index = index + step_samp
         
                # Determine file type and call appropriate get_spectra_slide function
                if self.ext.endswith('.fil'):
                    # For filterbank files, use sample indices directly
                    print(f"start_timesamp: {index}, end_timesamp: {end_index}")
                    self.spectra = self.rawdatafile.get_spectra_slide(start_time, index, end_index)
                elif self.ext.endswith('.fits'):
                    # For PSRFITS files, convert sample indices to subint indices
                    start_subint = int(index // self.samppersubint)
                    # end_subint = (end_index + self.samppersubint - 1) // self.samppersubint
                    end_subint = min(int(end_index // self.samppersubint), int(self.rawdatafile.nsubints - 1))
                    print(f"start_subint: {start_subint}, end_subint: {end_subint}")
                    self.spectra = self.rawdatafile.get_spectra_slide(start_time, start_subint, end_subint)
                else:
                    raise ValueError(f"Unknown file type: {self.ext}")
    
                # 传递start_time给get_results_single
                res = self.get_results_single(coordinate_mapping=coordinate_mapping, slide_start_time=start_time)
                if res is not None:
                    if coordinate_mapping:
                        res[:, 0] += start_time

                    results.append(res)

                index = end_index
                start_time += step_time

                # 在每次循环后进行轻量级清理
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if coordinate_mapping and len(results) > 0:
                results = np.concatenate(results, axis=0)
                
            # 强制清理内存
            self.clear()
            gc.collect()
        return results

   
    def get_results_single(self, coordinate_mapping: bool = True, slide_start_time: float = 0):
        """使用get_raw_data()访问数据，不直接引用self.raw_data"""
        if self.get_raw_data() is None or len(self.models) == 0:
            return None

        try:
            processed_fits_data = self.get_processed_data(clip_value=True)
            if processed_fits_data is None:
                return None
                
            processed_fits_data = processed_fits_data[:, :, np.newaxis]

            all_model_box = []
            all_model_mask = []

            for model in self.models:
                try:
                    model_cfg = model.cfg
                    
                    # 修改这里：使用新的方法获取test pipeline
                    test_pipeline = self._get_test_pipeline(model_cfg)
                    if ops_in_ppl(test_pipeline, ["DynamicCorrect"]):
                        inp = processed_fits_data  # dynamic correct follows the config in mmdet
                    else:
                        inp = dynamic_correct(processed_fits_data)  # manual dynamic correct with default scaling

                    # MMDet3中inference_detector返回DetDataSample对象
                    result = inference_detector(model, inp)
                    
                    # 立即清理输入数据的引用
                    if inp is not processed_fits_data:
                        del inp
                    
                    # 处理MMDet3的返回格式
                    if hasattr(result, 'pred_instances'):
                        pred_instances = result.pred_instances
                    elif isinstance(result, list) and len(result) > 0:
                        pred_instances = result[0].pred_instances
                    else:
                        print("No valid predictions found")
                        continue
                        
                    # 提取bbox和scores - 立即转为numpy并从GPU移除
                    if hasattr(pred_instances, 'bboxes') and hasattr(pred_instances, 'scores'):
                        bboxes = pred_instances.bboxes.detach().cpu().numpy()
                        scores = pred_instances.scores.detach().cpu().numpy()
                        labels = pred_instances.labels.detach().cpu().numpy() if hasattr(pred_instances, 'labels') else np.zeros(len(scores))
                        
                        bbox = np.concatenate([bboxes, scores.reshape(-1, 1)], axis=1)
                    else:
                        print("No bboxes found in prediction")
                        continue

                    # 提取masks - 立即转为numpy并从GPU移除
                    if hasattr(pred_instances, 'masks') and pred_instances.masks is not None:
                        masks_tensor = pred_instances.masks
                        if hasattr(masks_tensor, 'detach'):
                            masks = masks_tensor.detach().cpu().numpy()
                        else:
                            masks = masks_tensor
                            
                        # 确保masks是正确的格式
                        if len(masks.shape) == 3:  # (N, H, W)
                            masks = (masks > 0.5).astype(np.uint8)
                            masks = [masks[i] for i in range(masks.shape[0])]
                        elif len(masks.shape) == 2:  # (H, W) - 单个mask
                            masks = [(masks > 0.5).astype(np.uint8)]
                        else:
                            print(f"Unexpected masks shape: {masks.shape}")
                            continue
                    else:
                        print("No masks found in prediction")
                        continue

                    print(f"Detected {len(bbox)} objects with masks")
                    
                    if bbox.shape[0] == 0:
                        print("No bbox!!!!")
                        continue

                    # 转换masks为numpy数组
                    if isinstance(masks, list):
                        masks = np.stack(masks, axis=0)

                    # 过滤低置信度预测
                    index = (bbox[:, -1] >= self.postprocess.threshold)
                    bbox = bbox[index]
                    masks = masks[index]

                    all_model_box.append(bbox)
                    all_model_mask.append(masks)
                    
                    # 清理GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing model: {e}")
                    continue

            # 清理processed_fits_data
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
            
            # 只清理当前的spectra数据，但保留rawdatafile等元数据
            self.clear_spectra_only()
            
            if not coordinate_mapping:
                return bbox, masks

            for name in self.fits_param_names:
                if not hasattr(self, name):
                    return None

            # traverse back to astronomical coordinate
            fit_cfg = self.postprocess["mapping"].copy()
            fit_cfg = fit_cfg.copy()
            # print("fit_cfg:",fit_cfg)
            toa_dms = self.mask2ft(masks, **fit_cfg)

            scores = bbox[:, -1:]
            toa_dms = np.concatenate([toa_dms, scores], axis=1)

            # augment the results
            aug_cfg = self.postprocess.get("aug_cfg", None)
            if aug_cfg:
                cfg = aug_cfg.copy()
                name = cfg.pop("type")
                aug_func = getattr(augs, name, None)

                if aug_func is not None:
                    toa_dms = aug_func(toa_dms, **cfg)
                    
            return toa_dms
            
        except Exception as e:
            print(f"Error in get_results_single: {e}")
            # 只清理当前的spectra数据，但保留rawdatafile等元数据
            self.clear_spectra_only()
            gc.collect()
            return None
        
    def get_raw_data(self):
        """直接返回spectra.data，不再单独存储raw_data"""
        return None if self.spectra is None else self.spectra.data
    
    def get_spectra(self):
        return getattr(self, "spectra", None)

    def get_processed_data(self, clip_value=True):
    
        if self.spectra is None:
            return None
        try:
            # 保持原来的深拷贝方式
            data = copy.deepcopy(self.spectra)
            
            # check shape
            shape = data.data.shape
            assert len(shape) == 2, f"{len(shape)} vs. 2"
            print("raw data shape:", data.data.shape)

            # Subband data
            nsub = self.obsnchan // self.preprocess.downsample_freq
            if (nsub is not None):
                data.subband(nsub, padval='mean')

            # Downsample
            data.downsample(self.preprocess.downsample_time)

            # Masking and scaling
            data.mask_baseband_and_scale(basebandStd=1.0, indep=False)

            # exp normalization
            data.exp_normalization_plus()

            # 获取处理后的数据
            processed_data = data.data
            
            if clip_value:
                processed_data = dynamic_correct(processed_data, scaling=self.preprocess.scaling)
            
            print("processed data shape:", processed_data.shape)
            
            
            # 立即清理中间数据对象
            del data
            gc.collect()
            
            return processed_data
            
        except Exception as e:
            print(f"Error in get_processed_data: {e}")
            # 确保清理
            if 'data' in locals():
                del data
            gc.collect()
            return None