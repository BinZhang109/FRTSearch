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
    
    # Timing statistics
    timing = {}
    t_total = time.time()
    
    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    if not (args.input.endswith('.fits') or args.input.endswith('.fil')):
        raise ValueError("Input file must be a .fits or .fil file")
    
    # Step 1: Load configuration
    t1 = time.time()
    config = Config.fromfile(args.config)
    model = config.model
    device_id = 0
    
    if isinstance(model, list):
        for m in model:
            m.device = f"cuda:{device_id}"
    else:
        model.device = f"cuda:{device_id}"
    
    # Step 2: Initialize and load model + Load observation data
    detector = FRTDetector(config)
    detector.transfer_to_device(device_id)
    torch.cuda.set_device(device_id)
    detector.load_observation(args.input, slide_size=args.slide_size)
    timing['1. Model load and build'] = time.time() - t1
    print(f"[1/4] Model loaded and data loaded ({timing['1. Model load and build']:.2f}s)")
    
    # Setup output paths (save to current directory)
    input_basename = os.path.basename(args.input)
    input_dir = os.path.dirname(args.input) or "."
    file_prefix = os.path.splitext(input_basename)[0]
    
    # Use current directory instead of subdirectories
    output_dir = "."
    
    print(f"\n{'='*60}")
    print(f"FRTSearch - Fast Radio Transients Search")
    print(f"{'='*60}")
    print(f"Input file: {args.input}")
    print(f"Config: {args.config}")
    print(f"Slide size: {args.slide_size} subints")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    plot_tasks = []
    
    try:
        # Step 3 & 4 & 5: Preprocess + Detection + Parameter inference
        print(f"[2/4] Running preprocess, detection and parameter inference...")
        params = detector.get_results(
            coordinate_mapping=True,
            slide_size=args.slide_size
        )
        
        # Get accurate timing from detector
        timing['2. Preprocess'] = detector.timing['preprocess']
        timing['3. Detection'] = detector.timing['detection']
        timing['4. Parameter Inference'] = detector.timing['impic']
        print(f"      Preprocess completed ({timing['2. Preprocess']:.2f}s)")
        print(f"      Detection completed ({timing['3. Detection']:.2f}s)")
        print(f"      Parameter inference completed ({timing['4. Parameter Inference']:.2f}s)")
        
        if params is None or len(params) == 0:
            print("\n[Result] No candidates detected.")
            return
        
        print(f"\n[Result] Detected {len(params)} candidate(s)")
        
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
        
        # Write parameter file to current directory
        param_file = f"{file_prefix}.txt"
        fits_path = os.path.abspath(args.input)
        
        with open(param_file, "w") as f:
            # Write header
            f.write(f"# FRTSearch Detection Results\n")
            f.write(f"# Columns: ID FILE_Path ToA(s) DM(pc/cm³) Time_Downsample Freq_Downsample Confidence\n")
            f.write(f"#\n")
            
            for idx, param in enumerate(params, start=1):
                toa, dm, score = param
                # Format: ID FITS_Path TOA DM Time_Downsample Freq_Downsample Confidence
                f.write(f"{idx} {fits_path} {toa:.6f} {dm:.2f} {downt} {downf} {score:.6f}\n")
                
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
        
        print(f"Results saved to: {param_file}")
        
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
    
    # Step 6: Plotting
    timing['5. Plotting'] = 0
    if not args.no_plots and len(plot_tasks) > 0:
        print(f"\n[4/4] Generating diagnostic plots...")
        t1 = time.time()
        
        # Sort by confidence score
        if args.sort_by_score:
            plot_tasks = sorted(plot_tasks, key=lambda x: x.get('score', 0), reverse=True)
            print("      Sorted by confidence score")
        
        # Limit number of plots
        if args.max_plots > 0:
            plot_tasks = plot_tasks[:args.max_plots]
            print(f"      Limited to {len(plot_tasks)} plots")
        
        # Parallel plotting
        print(f"      Plotting {len(plot_tasks)} candidates using {args.num_plot_workers} worker(s)...")
        
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
        timing['5. Plotting'] = time.time() - t1
        print(f"      Plotting complete: {success_count}/{len(plot_tasks)} successful ({timing['5. Plotting']:.2f}s)")
        print(f"      Plots saved to current directory")
    
    # Calculate total time
    timing['Total'] = time.time() - t_total
    print(f"\nTotal processing time: {timing['Total']:.2f}s")


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
    print("FRTSearch Pipeline")
    print("="*60)
    
    tic = time.time()
    main(args)
    toc = time.time()
    
    print(f"Program finished. Total elapsed time: {toc - tic:.2f}s")