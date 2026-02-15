#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRTSearch Test Samples - Python Script Version

This script evaluates FRTSearch detection instances and parameter configurations.
Converted from test_samples.ipynb for terminal execution.
"""

import sys
import os
import time
import argparse

# Add project root to path and change to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}\n")

from utils import FRTDetector, PsrfitsFile, FilterbankFile, waterfall_jupyter
from mmengine.config import Config


def detect_and_plot(file_path, config_file, slide_size=8, rank_to_display=1, 
                    freq_range=None, nsubint=None):
    """
    FRB detection and plotting with timing statistics.
    
    Args:
        file_path: Path to observation file (.fits or .fil)
        config_file: Path to detector config file
        slide_size: Sliding window size for detection
        rank_to_display: Which ranked candidate to plot (1 = best)
        freq_range: Optional (freq_min, freq_max) to override config
        nsubint: Optional number of subints for plotting to override config
    
    Returns:
        Path to saved PNG image or None
    """
    times = {}
    t0 = time.time()
    
    # Stage 1: Model load and build + Data loading (combined)
    t1 = time.time()
    cfg = Config.fromfile(config_file)
    if freq_range:
        cfg.preprocess['freq_range'] = freq_range
    if nsubint is not None:
        cfg.preprocess['nsubint'] = nsubint
    detector = FRTDetector(cfg)
    detector.load_observation(file_path, slide_size=slide_size)
    times['Model load and build'] = time.time() - t1
    
    # Stage 2-4: Preprocess + Detection + IMPIC
    results = detector.get_results(coordinate_mapping=True, slide_size=slide_size)
    
    # Get accurate timing from detector
    times['Preprocess'] = detector.timing['preprocess']
    times['Detection'] = detector.timing['detection']
    times['IMPIC'] = detector.timing['impic']
    
    if results is not None and len(results) > 0:
        results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
        
        print(f"\n{'='*70}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"Detected {len(results_sorted)} candidates:")
        print(f"{'Rank':<6} {'TOA (s)':<12} {'DM (pc/cm³)':<15} {'Confidence':<12}")
        print(f"{'-'*70}")
        for i, (toa, dm, score) in enumerate(results_sorted, 1):
            print(f"{i:<6} {toa:<12.5f} {dm:<15.2f} {score:<12.4f}")
        print(f"{'='*70}\n")
        
        # Plot
        if rank_to_display > len(results_sorted):
            print(f"Error: Rank #{rank_to_display} exceeds {len(results_sorted)} candidates")
            return None
            
        toa, dm, score = results_sorted[rank_to_display - 1]
        print(f"Plotting Rank #{rank_to_display}: ToA={toa:.5f}s, DM={dm:.2f}, Conf={score:.4f}\n")
        
        # Stage 5: Plot
        t1 = time.time()
        preprocess = detector.preprocess
        rawdatafile = FilterbankFile(file_path) if file_path.endswith('.fil') else PsrfitsFile(file_path)
        pulsar_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Use nsubint for plotting duration
        plot_nsubint = preprocess.get('nsubint', 4)
        
        img = waterfall_jupyter(
            rawdatafile=rawdatafile, start=toa, duration=plot_nsubint,
            dm=dm, fd=preprocess.downsample_freq, downsamp=preprocess.downsample_time,
            tboxwindow=preprocess.tbox, scaling=preprocess.scaling,
            basebandStd=preprocess.basebandStd, pulsar=pulsar_name, confidence=score
        )
        times['Plot'] = time.time() - t1
        times['Total'] = time.time() - t0
        
        # Print timing summary
        print("Timing:")
        for stage in ['Model load and build', 'Preprocess', 'Detection', 'IMPIC', 'Plot', 'Total']:
            if stage in times:
                v = times[stage]
                pct = f"({v/times['Total']*100:>5.1f}%)" if stage != 'Total' else ""
                print(f"  {stage:<22}: {v:>6.2f}s  {pct}")
        print()
        
        # Save image
        save_name = os.path.splitext(os.path.basename(file_path))[0] + f'_rank{rank_to_display}.png'
        save_path = os.path.join('test_sample', save_name)
        if img is not None:
            with open(save_path, 'wb') as f:
                f.write(img.data)
            print(f"Image saved to: {save_path}\n")
        
        return save_path
    else:
        print("No candidates detected.")
        times['Total'] = time.time() - t0
        print(f"Total time: {times['Total']:.2f}s")
        return None


def main():
    """Main function with example test cases."""
    parser = argparse.ArgumentParser(description='FRTSearch Detection Test Script')
    parser.add_argument('--file', type=str, help='Path to observation file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--slide-size', type=int, default=8, help='Sliding window size')
    parser.add_argument('--rank', type=int, default=1, help='Rank to display')
    parser.add_argument('--freq-min', type=float, help='Minimum frequency (MHz)')
    parser.add_argument('--freq-max', type=float, help='Maximum frequency (MHz)')
    parser.add_argument('--nsubint', type=int, help='Number of subints for plotting')
    parser.add_argument('--example', type=str, choices=[
        'FRB20121102', 'FRB20201124', 'FRB20180301', 
        'FRB20180119', 'FRB20180212', 'FRB20220610A'
    ], help='Run predefined example')
    
    args = parser.parse_args()
    
    # If example is specified, use predefined parameters
    if args.example:
        examples = {
            'FRB20121102': {
                'file_path': './test_sample/FRB20121102_0038.fits',
                'config_file': './configs/detector_FAST.py',
                'slide_size': 128,
                'rank_to_display': 3,
            },
            'FRB20201124': {
                'file_path': './test_sample/FRB20201124_0016.fits',
                'config_file': './configs/detector_FAST.py',
                'slide_size': 128,
                'rank_to_display': 3,
            },
            'FRB20180301': {
                'file_path': './test_sample/FRB20180301_0004.fits',
                'config_file': './configs/detector_FAST.py',
                'slide_size': 128,
                'rank_to_display': 2,
            },
            'FRB20180119': {
                'file_path': './test_sample/FRB20180119_SKA_1660_1710.fil',
                'config_file': './configs/detector_SKA.py',
                'slide_size': 8,
                'rank_to_display': 1,
            },
            'FRB20180212': {
                'file_path': './test_sample/FRB20180212_SKA_1820_1870.fil',
                'config_file': './configs/detector_SKA.py',
                'slide_size': 8,
                'rank_to_display': 1,
            },
           
        }
        
        params = examples[args.example]
        detect_and_plot(**params)
        
    elif args.file and args.config:
        # Use command line arguments
        freq_range = None
        if args.freq_min and args.freq_max:
            freq_range = (args.freq_min, args.freq_max)
        
        detect_and_plot(
            file_path=args.file,
            config_file=args.config,
            slide_size=args.slide_size,
            rank_to_display=args.rank,
            freq_range=freq_range,
            nsubint=args.nsubint,
        )
    else:
        # Default: Run FRB20121102 example
        print("No arguments provided. Running default example: FRB20121102\n")
        print("Usage examples:")
        print("  python test_sample/test_samples.py --example FRB20121102")
        print("  python test_sample/test_samples.py --file ./test_sample/FRB20121102_0038.fits --config ./configs/detector_FAST.py --slide-size 128 --rank 3\n")
        
        detect_and_plot(
            file_path='./test_sample/FRB20121102_0038.fits',
            config_file='./configs/detector_FAST.py',
            slide_size=128,
            rank_to_display=3,
        )


if __name__ == '__main__':
    main()
