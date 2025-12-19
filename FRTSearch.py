import os
import argparse
import tqdm
import logging
import traceback
import gc
import torch
import torch.multiprocessing as mp
from multiprocessing import Pool
import pickle
import time

from utils import FitsDetector, waterfall
from mmengine.config import Config

def plot_worker(plot_task):
    """绘图工作进程"""
    try:
        from utils import psrfits, filterbank
        
        file_path = plot_task['file_path']
        toa = plot_task['toa']
        nsubint = plot_task['nsubint']
        dm = plot_task['dm']
        downf = plot_task['downf']
        downt = plot_task['downt']
        tbox = plot_task['tbox']
        scaling = plot_task['scaling']
        basebandStd = plot_task['basebandStd']
        output_dir = plot_task['output_dir']
        pulsar = plot_task['pulsar']
        confidence = plot_task['confidence']
        
        # 重新加载文件
        if file_path.endswith(".fil"):
            rawdatafile = filterbank.FilterbankFile(file_path)
        elif file_path.endswith(".fits"):
            rawdatafile = psrfits.PsrfitsFile(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return False
        
        # 调用waterfall绘图
        waterfall(
            rawdatafile=rawdatafile,
            start=toa,
            duration=nsubint,
            dm=dm,
            fd=downf,
            downsamp=downt,
            tboxwindow=tbox,
            scaling=scaling,
            basebandStd=basebandStd,
            plot_dir=output_dir,
            pulsar=pulsar,
            confidence=confidence,
            plot_only=True  # 只绘图，不重复计算
        )
        
        print(f"Successfully plotted: {pulsar}_toa{toa}_DM{dm}.png")
        return True
        
    except Exception as e:
        print(f"Error plotting {plot_task.get('pulsar', 'unknown')}: {e}")
        traceback.print_exc()
        return False

def main(args):
    """主函数 - FRT搜索"""
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # 检查文件格式
    if not (args.input.endswith('.fits') or args.input.endswith('.fil')):
        raise ValueError("Input file must be a .fits or .fil file")
    
    # 加载配置
    config = Config.fromfile(args.config)
    model = config.model
    device_id = 0  # 默认使用GPU 0
    
    if isinstance(model, list):
        for m in model:
            m.device = f"cuda:{device_id}"
    else:
        model.device = f"cuda:{device_id}"
    
    # 初始化检测器
    detector = FitsDetector(config)
    detector.to_device(device_id)
    torch.cuda.set_device(device_id)
    print("FRTSearch detector initialized")
    
    # 设置输出目录
    input_basename = os.path.basename(args.input)
    input_dir = os.path.dirname(args.input) or "."
    file_prefix = os.path.splitext(input_basename)[0]
    
    param_dir = os.path.join(input_dir, f"{file_prefix}_results")
    output_dir = os.path.join(param_dir, "plots") if not args.no_plots else None
    
    if not os.path.exists(param_dir):
        os.makedirs(param_dir, exist_ok=True)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FRTSearch - Fast Radio Transients Search")
    print(f"{'='*60}")
    print(f"Input file: {args.input}")
    print(f"Config: {args.config}")
    print(f"Slide size: {args.slide_size} subints")
    print(f"Output directory: {param_dir}")
    print(f"{'='*60}\n")
    
    try:
        # 加载文件并进行检测
        print(f"Processing: {input_basename}")
        detector.load_fits(args.input, slide_size=args.slide_size)
        
        params = detector.get_results(
            coordinate_mapping=True,
            slide_size=args.slide_size
        )
        
        if params is None or len(params) == 0:
            print("No candidates detected.")
            return
        
        print(f"Detected {len(params)} candidate(s)")
        
        # 获取文件信息
        from utils import psrfits, filterbank
        if args.input.endswith(".fil"):
            rawdatafile = filterbank.FilterbankFile(args.input)
        else:
            rawdatafile = psrfits.PsrfitsFile(args.input)
        
        # 获取频率范围
        start_freq = rawdatafile.freqs.min()
        end_freq = rawdatafile.freqs.max()
        
        # 获取预处理参数
        downt = detector.preprocess.downsample_time
        downf = detector.preprocess.downsample_freq
        nsubint = detector.preprocess.nsubint
        tbox = detector.preprocess.tbox
        basebandStd = detector.preprocess.basebandStd
        scaling = detector.preprocess.scaling
        
        # 提取pulsar名称
        pulsar = extract_pulsar_name(input_basename)
        
        # ========== 关键修改：先计算所有候选的参数 ==========
        print(f"\nComputing candidate parameters...")
        candidate_params = []
        
        for idx, param in enumerate(tqdm.tqdm(params, desc="Computing parameters"), 1):
            toa, dm, confidence = param
            
            # 调用waterfall函数计算参数（不绘图）
            result = waterfall(
                rawdatafile=rawdatafile,
                start=toa,
                duration=nsubint,
                dm=dm,
                fd=downf,
                downsamp=downt,
                tboxwindow=tbox,
                scaling=scaling,
                basebandStd=basebandStd,
                plot_dir=output_dir,
                pulsar=pulsar,
                confidence=confidence,
                compute_only=True  # 只计算，不绘图
            )
            
            if result is not None:
                snr, width_ms, mjd_toa = result
                candidate_params.append({
                    'cand_id': idx,
                    'toa': toa,
                    'mjd_toa': mjd_toa,
                    'dm': dm,
                    'confidence': confidence,
                    'snr': snr,
                    'width_ms': width_ms,
                    'start_freq': start_freq,
                    'end_freq': end_freq
                })
            else:
                # 如果计算失败，使用默认值
                toa_mjd = float(rawdatafile.mjd) + (toa / 86400.0)
                candidate_params.append({
                    'cand_id': idx,
                    'toa': toa,
                    'mjd_toa': toa_mjd,
                    'dm': dm,
                    'confidence': confidence,
                    'snr': 0.0,
                    'width_ms': 0.0,
                    'start_freq': start_freq,
                    'end_freq': end_freq
                })
        
        # ========== 写参数文件 ==========
        param_file = os.path.join(param_dir, f"{file_prefix}_candidates.txt")
        
        with open(param_file, "w") as f:
            # 写入表头
            f.write("# FRTSearch Candidates\n")
            f.write(f"# File: {input_basename}\n")
            f.write(f"# Frequency range: {start_freq:.2f} - {end_freq:.2f} MHz\n")
            f.write("#\n")
            f.write("# Columns: CandID | ToA(MJD) | DM(pc/cm³) | Confidence | SNR | Width(ms) | StartFreq(MHz) | EndFreq(MHz) | Filename\n")
            f.write("#" + "-"*120 + "\n")
            
            for cand in candidate_params:
                f.write(f"{cand['cand_id']:4d} | "
                       f"{cand['mjd_toa']:.10f} | "
                       f"{cand['dm']:8.2f} | "
                       f"{cand['confidence']:6.4f} | "
                       f"{cand['snr']:7.2f} | "
                       f"{cand['width_ms']:7.2f} | "
                       f"{cand['start_freq']:8.2f} | "
                       f"{cand['end_freq']:8.2f} | "
                       f"{input_basename}\n")
        
        print(f"\nResults saved to: {param_file}")
        
        # ========== 执行绘图 ==========
        if not args.no_plots and len(candidate_params) > 0:
            print(f"\n{'='*60}")
            print(f"Generating diagnostic plots...")
            print(f"{'='*60}\n")
            
            # 准备绘图任务
            plot_tasks = []
            for cand in candidate_params:
                plot_task = {
                    'file_path': args.input,
                    'toa': cand['toa'],
                    'nsubint': nsubint,
                    'dm': cand['dm'],
                    'downf': downf,
                    'downt': downt,
                    'tbox': tbox,
                    'scaling': scaling,
                    'basebandStd': basebandStd,
                    'output_dir': output_dir,
                    'pulsar': pulsar,
                    'confidence': cand['confidence'],
                    'cand_id': cand['cand_id']
                }
                plot_tasks.append(plot_task)
            
            # 根据置信度排序
            if args.sort_by_score:
                plot_tasks = sorted(plot_tasks, key=lambda x: x['confidence'], reverse=True)
                print("Sorted by confidence score")
            
            # 限制绘图数量
            if args.max_plots > 0:
                plot_tasks = plot_tasks[:args.max_plots]
                print(f"Limited to {len(plot_tasks)} plots")
            
            # 并行绘图
            print(f"Using {args.num_plot_workers} worker(s)...")
            with Pool(processes=args.num_plot_workers) as pool:
                results = list(tqdm.tqdm(
                    pool.imap(plot_worker, plot_tasks),
                    total=len(plot_tasks),
                    desc="Plotting"
                ))
            
            success_count = sum(results)
            print(f"\nPlotting complete: {success_count}/{len(plot_tasks)} successful")
            print(f"Plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error processing {args.input}: {e}")
        logging.exception(e)
        traceback.print_exc()
    finally:
        # 清理
        detector.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def extract_pulsar_name(filename):
    """从文件名提取pulsar名称"""
    import re
    basename = os.path.basename(filename)
    basename = re.sub(r'\.(fits|fil|npy)$', '', basename)
    
    # 检查FRB
    if 'FRB' in basename:
        frb_match = re.search(r'FRB\d+[A-Z]?', basename)
        if frb_match:
            return frb_match.group(0)
    
    # 提取源名称
    parts = basename.split('_')
    if len(parts) > 0:
        source_name = parts[0]
        return f"PSR_{source_name}"
    
    return "PSR_Unknown"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FRTSearch",
        description="""
╔══════════════════════════════════════════════════════════════╗
║                        FRTSearch                              ║
║          Fast Radio Transients Search Tool                    ║
║                                                               ║
║  A deep learning-based detection tool for fast radio         ║
║  transients in pulsar search data (PSRFITS/Filterbank)       ║
╚══════════════════════════════════════════════════════════════╝
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection without plots
  python frtsearch.py data.fits config.py --slide-size 64 --no-plots
  
  # Detection with diagnostic plots
  python frtsearch.py data.fil config.py --slide-size 128
  
  # Limited plotting with sorting
  python frtsearch.py data.fits config.py --slide-size 64 --max-plots 10 --sort-by-score
        """
    )
    
    # 必需参数
    parser.add_argument("input", 
                       help="Input PSRFITS (.fits) or Filterbank (.fil) file")
    parser.add_argument("config", 
                       help="Path to Mask R-CNN configuration file")
    parser.add_argument("--slide-size", 
                       help="Sliding window size (number of subints to read at once)",
                       type=int, required=True)
    
    # 绘图控制参数
    parser.add_argument("--no-plots", 
                       help="Disable diagnostic plot generation (default: enabled)",
                       action="store_true")
    parser.add_argument("--num-plot-workers", 
                       help="Number of parallel workers for plotting (default: 4)",
                       type=int, default=4)
    parser.add_argument("--max-plots", 
                       help="Maximum number of plots to generate, 0=unlimited (default: 0)",
                       type=int, default=0)
    parser.add_argument("--sort-by-score", 
                       help="Sort plots by confidence score (highest first)",
                       action="store_true")
    
    # 日志参数
    parser.add_argument("--no-log", 
                       help="Do not output log files",
                       action="store_true")
    
    args = parser.parse_args()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, this may be very slow!")
    
    # 开始处理
    tic = time.time()
    main(args)
    toc = time.time()
    
    print(f"\n{'='*60}")
    print(f"Total time: {toc - tic:.2f}s")
    print(f"{'='*60}")