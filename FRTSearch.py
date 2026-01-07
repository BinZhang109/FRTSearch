#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Bin Zhang
Date: 2025.8.11

FRTSearch - Fast Radio Transients Search Tool (Single File Mode)

Usage:
    python FRTSearch.py data.fits config.py --slide-size 128
    python FRTSearch.py data.fil config.py --slide-size 64 --no-plots
"""

import os
import argparse
import tqdm
import logging
import traceback
import gc
import re
import time

import torch
from multiprocessing import Pool

from utils import FRTDetector, waterfall
from mmengine.config import Config


def plot_worker(plot_task):
    """Plot worker process for parallel plotting"""
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
        confidence = plot_task.get('score', None)
        
        # Reload the data file
        if file_path.endswith(".fil"):
            rawdatafile = filterbank.FilterbankFile(file_path)
        elif file_path.endswith(".fits"):
            rawdatafile = psrfits.PsrfitsFile(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return False
        
        # Generate waterfall plot
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
            confidence=confidence
        )
        
        print(f"Successfully plotted: {pulsar}_toa{toa:.2f}_DM{dm:.2f}.png")
        return True
        
    except Exception as e:
        print(f"Error plotting {plot_task.get('pulsar', 'unknown')}: {e}")
        traceback.print_exc()
        return False


def extract_pulsar_name(filename):
    """Extract pulsar/source name from filename"""
    basename = os.path.basename(filename)
    basename = re.sub(r'\.(fits|fil|npy)$', '', basename)
    
    # Check for FRB naming pattern
    if 'FRB' in basename:
        frb_match = re.search(r'FRB\d+[A-Z]?', basename)
        if frb_match:
            return frb_match.group(0)
    
    # Extract source name from filename parts
    parts = basename.split('_')
    if len(parts) > 0:
        source_name = parts[0]
        return f"PSR_{source_name}"
    
    return "PSR_Unknown"


def main(args):
    """Main function - Single file FRT search"""
    
    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    if not (args.input.endswith('.fits') or args.input.endswith('.fil')):
        raise ValueError("Input file must be a .fits or .fil file")
    
    # Load configuration
    config = Config.fromfile(args.config)
    model = config.model
    device_id = 0
    
    if isinstance(model, list):
        for m in model:
            m.device = f"cuda:{device_id}"
    else:
        model.device = f"cuda:{device_id}"
    
    # Initialize detector
    detector = FRTDetector(config)
    detector.transfer_to_device(device_id)
    torch.cuda.set_device(device_id)
    print("FRTSearch detector initialized")
    
    # Setup output directories
    input_basename = os.path.basename(args.input)
    input_dir = os.path.dirname(args.input) or "."
    file_prefix = os.path.splitext(input_basename)[0]
    ext = args.input.split(".")[-1]
    
    param_dir = os.path.join(input_dir, f"{file_prefix}_params")
    output_dir = os.path.join(input_dir, f"{file_prefix}_png")
    
    if not os.path.exists(param_dir):
        os.makedirs(param_dir, exist_ok=True)
    if not args.no_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FRTSearch - Fast Radio Transients Search")
    print(f"{'='*60}")
    print(f"Input file: {args.input}")
    print(f"Config: {args.config}")
    print(f"Slide size: {args.slide_size} subints")
    print(f"Output directory: {param_dir}")
    print(f"{'='*60}\n")
    
    plot_tasks = []
    
    try:
        # ========== Stage 1: Detection + Inference (combined) ==========
        print(f"Processing: {input_basename}")
        
        # Load observation file
        detector.load_observation(args.input, slide_size=args.slide_size)
        
        # Detection + coordinate mapping (inference) combined
        params = detector.get_results(
            coordinate_mapping=True,
            slide_size=args.slide_size
        )
        
        if params is None or len(params) == 0:
            print("No candidates detected.")
            return
        
        print(f"Detected {len(params)} candidate(s)")
        
        # Get preprocessing parameters
        downt = detector.preprocess.downsample_time
        downf = detector.preprocess.downsample_freq
        nsubint = detector.preprocess.nsubint
        tbox = detector.preprocess.tbox
        basebandStd = detector.preprocess.basebandStd
        scaling = detector.preprocess.scaling
        
        # Extract pulsar/source name
        pulsar = extract_pulsar_name(input_basename)
        if pulsar == 'PSR_Unknown':
            pulsar = file_prefix
        
        # Write parameter file
        param_file = os.path.join(param_dir, f"{file_prefix}.txt")
        
        with open(param_file, "w") as f:
            for param in params:
                toa, dm, score = param
                
                # Write parameters (format consistent with batch processing)
                cmd = f"{args.input} {toa:.6f} {nsubint} " \
                      f"{dm:.6f} {downf} {downt} {tbox} " \
                      f"{scaling} {basebandStd} {score:.4f} {output_dir}\n"
                f.write(cmd)
                
                # Collect plotting tasks
                if not args.no_plots:
                    plot_task = {
                        'file_path': args.input,
                        'toa': toa,
                        'nsubint': nsubint,
                        'dm': dm,
                        'downf': downf,
                        'downt': downt,
                        'tbox': tbox,
                        'scaling': scaling,
                        'basebandStd': basebandStd,
                        'output_dir': output_dir,
                        'pulsar': pulsar,
                        'score': score
                    }
                    plot_tasks.append(plot_task)
        
        print(f"\nResults saved to: {param_file}")
        
    except Exception as e:
        print(f"Error processing {args.input}: {e}")
        logging.exception(e)
        traceback.print_exc()
    finally:
        # Clean up detector memory
        detector.release_memory()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========== Stage 2: Plotting (separate execution) ==========
    if not args.no_plots and len(plot_tasks) > 0:
        print(f"\n{'='*60}")
        print(f"Stage 2: Generating diagnostic plots")
        print(f"{'='*60}\n")
        
        # Sort by confidence score
        if args.sort_by_score:
            plot_tasks = sorted(plot_tasks, key=lambda x: x.get('score', 0), reverse=True)
            print("Sorted by confidence score")
        
        # Limit number of plots
        if args.max_plots > 0:
            plot_tasks = plot_tasks[:args.max_plots]
            print(f"Limited to {len(plot_tasks)} plots")
        
        # Parallel plotting
        print(f"Plotting {len(plot_tasks)} candidates using {args.num_plot_workers} worker(s)...")
        
        if args.num_plot_workers > 1:
            with Pool(processes=args.num_plot_workers) as pool:
                results = list(tqdm.tqdm(
                    pool.imap(plot_worker, plot_tasks),
                    total=len(plot_tasks),
                    desc="Plotting"
                ))
        else:
            # Single process mode
            results = []
            for task in tqdm.tqdm(plot_tasks, desc="Plotting"):
                results.append(plot_worker(task))
        
        success_count = sum(results)
        print(f"\nPlotting complete: {success_count}/{len(plot_tasks)} successful")
        print(f"Plots saved to: {output_dir}")


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
  python FRTSearch.py data.fits config.py --slide-size 64 --no-plots
  
  # Detection with diagnostic plots
  python FRTSearch.py data.fil config.py --slide-size 128
  
  # Limited plotting with sorting
  python FRTSearch.py data.fits config.py --slide-size 64 --max-plots 10 --sort-by-score
        """
    )
    
    # Required arguments
    parser.add_argument("input", 
                       help="Input PSRFITS (.fits) or Filterbank (.fil) file")
    parser.add_argument("config", 
                       help="Path to Mask R-CNN configuration file")
    parser.add_argument("--slide-size", "--slide",
                       dest="slide_size",
                       help="Sliding window size (number of subints to read at once)",
                       type=int, default=-1)
    
    # Plotting control arguments
    parser.add_argument("--no-plots", 
                       help="Disable diagnostic plot generation",
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
    
    # Logging arguments
    parser.add_argument("--no-log", 
                       help="Do not output log files",
                       action="store_true")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, this may be very slow!")
    
    # Start processing
    print("\n" + "="*60)
    print("Stage 1: Detection + Parameter Inference")
    print("="*60)
    
    tic = time.time()
    main(args)
    toc = time.time()
    
    print(f"\n{'='*60}")
    print(f"Total time: {toc - tic:.2f}s")
    print(f"{'='*60}")