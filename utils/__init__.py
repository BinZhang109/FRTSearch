"""
Author: Bin Zhang
Date: 2025.8.11
"""
from .loader import LoadImageFromNumpy, FastData
from .detector import FRTDetector
from .psrfits import PsrfitsFile,SpectraInfo
from .filterbank import FilterbankFile
from .spectra import Spectra 
from .waterfaller_zhang import waterfall, waterfall_jupyter, show_waterfall_in_jupyter
