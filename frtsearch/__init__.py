"""
FRTSearch - Fast Radio Transients Search Tool

A deep learning-based detection tool for fast radio transients
in pulsar search data (PSRFITS/Filterbank).

Author: Bin Zhang
"""

__version__ = "0.1.3"

from utils import FRTDetector, waterfall
from utils import PsrfitsFile, SpectraInfo
from utils import FilterbankFile
from utils import Spectra

__all__ = [
    "FRTDetector",
    "waterfall",
    "PsrfitsFile",
    "SpectraInfo",
    "FilterbankFile",
    "Spectra",
]
