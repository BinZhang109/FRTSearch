#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRTSearch CLI entry point.

Usage:
    frtsearch data.fits config.py --slide-size 128
    frtsearch data.fil config.py --slide-size 64 --no-plots
"""

import sys
import os

# Add the package root to path so that 'utils' can be imported
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)


def main():
    """CLI entry point — delegates to FRTSearch.main()"""
    # Import here to avoid heavy loading on module import
    import argparse
    import time
    import torch

    from frtsearch._core import main as frtsearch_main

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
  FRTSearch data.fits config.py --slide-size 64 --no-plots

  # Detection with diagnostic plots
  FRTSearch data.fil config.py --slide-size 128

  # Limited plotting with sorting
  FRTSearch data.fits config.py --slide-size 64 --max-plots 10 --sort-by-score
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
    print("\n" + "=" * 60)
    print("FRTSearch Pipeline")
    print("=" * 60)

    tic = time.time()
    frtsearch_main(args)
    toc = time.time()

    print(f"Program finished. Total elapsed time: {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
