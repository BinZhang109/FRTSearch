"""
Modified from PRESTO's waterfaller.py module
Original: https://github.com/scottransom/presto
Original Authors: Scott Ransom and contributors
Modified by: Bin Zhang
Date: 2024.06.23

This module is based on PRESTO's waterfaller and has been extensively
modified to support FRTSearch's waterfall plot generation.

PRESTO is licensed under the GNU General Public License v2.0.
See LICENSE file for details.
"""

import sys
import optparse
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy import stats
from astropy.time import Time
from presto import psr_utils
from presto import rfifind
from . import psrfits
from . import filterbank
import astropy.io.fits as pyfits
import datetime as dt
import os.path as osp
import os 
import re
from pylab import *
from decimal import Decimal
import matplotlib
matplotlib.use("Agg")
from matplotlib.pyplot import Polygon
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy import integrate
from scipy import signal
import math
import traceback

input_snr = 1.0
ra_dec = None
start_subint_num = 1

starttime = dt.datetime.now()
secperday = 3600. * 24.
accel_patch = int(os.getenv("ACCEL", 0))
fits_id_matcher = re.compile(r"M[\d]{2}([\d]{4})")


def extract_pulsar_name(filename):
    """Extract pulsar name from filename."""
    basename = os.path.basename(filename)
    basename = re.sub(r'\.(fits|fil|npy)$', '', basename)
    
    if 'FRB' in basename:
        frb_match = re.search(r'FRB\d+[A-Z]?', basename)
        if frb_match:
            return frb_match.group(0)
    
    parts = basename.split('_')
    if len(parts) > 0:
        source_name = parts[0]
        return f"PSR_{source_name}"
    
    return "PSR_Unknown"


def fit_single_gaussian(profile, profile_y, profile_std, section, py_max, tsamp, downsamp, ax, tboxwindow, sampperbox):
    """Single Gaussian fitting function."""
    max_idx = np.argmax(profile_y)
    max_height = profile_y[max_idx]
    
    half_max = max_height / 2.0
    left_idx = max_idx
    right_idx = max_idx
    
    while left_idx > 0 and profile_y[left_idx] > half_max:
        left_idx -= 1
    while right_idx < len(profile_y)-1 and profile_y[right_idx] > half_max:
        right_idx += 1
    
    estimated_sigma = (right_idx - left_idx) / 2.355
    
    def single_gaussian(x, a, mu, sigma, d):
        return a * np.exp(-(x-mu)**2/(2*sigma**2)) + d
    
    try:
        bounds_lower = [0, max_idx - estimated_sigma*3, estimated_sigma/3, 0]
        bounds_upper = [max_height*2, max_idx + estimated_sigma*3, estimated_sigma*3, profile_std]
        
        initial_guess = [max_height, max_idx, estimated_sigma, 0]
        
        params, pcov = curve_fit(single_gaussian, section, profile_y, 
                               p0=initial_guess, bounds=(bounds_lower, bounds_upper), maxfev=10000)
        
        y_fit = single_gaussian(section, *params)
        
        y_mean = np.mean(profile_y)
        ss_total = np.sum((profile_y - y_mean) ** 2)
        ss_residual = np.sum((profile_y - y_fit) ** 2)
        goodness = 1 - (ss_residual / ss_total)
        
        a, mu, sigma, d = params
        
        if ax is not None:
            ax.axvline(x=pulse_left, linestyle='--', color='blue', alpha=0.6, linewidth=2, label='On Pulse')
            ax.axvline(x=pulse_right, linestyle='--', color='blue', alpha=0.6, linewidth=2)
        
        snr = abs(a * py_max / profile_std)
        width_ms = abs(2.0 * math.sqrt(2 * math.log(2)) * sigma * tsamp * downsamp * 1e3)
        
        pulse_left = mu - 3*sigma
        pulse_right = mu + 3*sigma
        
        return params, pulse_left, pulse_right, snr, width_ms
        
    except Exception as e:
        print(f"Gaussian fitting failed: {e}")
        return None, None, None, None, None


def add_ticks_all_sides(ax, exclude_raw=False):
    """Add tick marks on all sides of the subplot."""
    if exclude_raw:
        return
    
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    ax.tick_params(axis='both', which='major', direction='in', 
                   top=True, right=True, bottom=True, left=True,
                   length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in',
                   top=True, right=True, bottom=True, left=True, 
                   length=3, width=0.5)
    
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    
def plot_profile_with_precomputed(ax, profile, section, profile_std, py_max, normalized_data,
                                 fit_results, pulse_left, pulse_right, snr, width_ms):
    """Plot pulse profile with precomputed fitting results."""
    ax.plot(profile, '-k', linewidth=2)
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.fill_between(section, profile+1.5*profile_std, profile-1.5*profile_std,
                   alpha=0.5, facecolor="gray", color="white")
    
    if pulse_left is not None and pulse_right is not None:
        ax.axvline(x=pulse_left, linestyle='--', color='blue', alpha=0.6, linewidth=2, label='On Pulse')
        ax.axvline(x=pulse_right, linestyle='--', color='blue', alpha=0.6, linewidth=2)
    
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0)
    ax.yaxis.set_visible(False)
    max_min_delta = np.max(profile) - np.min(profile)
    ax.set_xlim([np.min(section), np.max(section)])
    ax.set_ylim([np.min(profile)-max_min_delta/7., np.max(profile)+max_min_delta/7.])
    
    add_ticks_all_sides(ax)
    
    if pulse_left is not None:
        ax.legend(loc='upper right', fontsize=14, prop={'family': 'Times New Roman'})
    
    return snr, width_ms, pulse_left, pulse_right


def plot_profile(ax, profile, section, profile_std, py_max, tsamp, downsamp, tboxwindow, sampperbox, normalized_data):
    """Plot pulse profile."""
    ax.plot(profile, '-k', linewidth=2)
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.fill_between(section, profile+1.5*profile_std, profile-1.5*profile_std, 
                   alpha=0.5, facecolor="gray", color="white")

    profile_y = profile - np.min(profile)
    profile_y = profile_y / py_max
    
    fit_results, pulse_left, pulse_right, snr, width_ms = fit_single_gaussian(
        profile, profile_y, profile_std, section, py_max, tsamp, downsamp, ax, tboxwindow, sampperbox)
    
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0)
    ax.yaxis.set_visible(False)
    max_min_delta = np.max(profile) - np.min(profile)
    ax.set_xlim([np.min(section), np.max(section)])
    ax.set_ylim([np.min(profile)-max_min_delta/7., np.max(profile)+max_min_delta/7.])
    
    add_ticks_all_sides(ax)
    
    ax.legend(loc='upper right', fontsize=10, prop={'family': 'Times New Roman'})
    
    return snr, width_ms, pulse_left, pulse_right


def plot_dedispersed_waterfall(ax, start, exp_normalized_data, startf, endf, dm, tboxwindow, 
                              masked_channels, scaling, plot_vmin, plot_vmax, pulse_left=None, pulse_right=None, section=None):
    """Plot dedispersed waterfall."""
    im = ax.imshow(exp_normalized_data, aspect='auto', cmap=plt.cm.viridis, 
                   origin="lower", vmin=plot_vmin, vmax=plot_vmax)
    
    ax.set_xlabel('Time (ms) ', fontsize = 20, fontweight='regular')
    ax.set_ylabel('Frequency (MHz)', fontsize = 20, fontweight='regular')
    
    l_tick, m_tick = exp_normalized_data.shape
    
    floattimelabelddm = np.linspace(0, m_tick-1, 11)
    strtimelabelddmtop = [str(timepoint) for timepoint in np.linspace(-tboxwindow, tboxwindow, 11).astype('int64')]
    floatfreqlabel = np.linspace(0, l_tick-1, 5)
    strfreqlabel = [str(int(freqpoint)) for freqpoint in np.linspace(startf, endf, 5)]
    
    ax.set_yticks(floatfreqlabel)
    ax.set_yticklabels(strfreqlabel, fontsize=16, family='Times New Roman')
    ax.set_xticks(floattimelabelddm)
    ax.set_xticklabels(strtimelabelddmtop, fontsize=16, family='Times New Roman')
    
    if pulse_left is not None and pulse_right is not None and section is not None:
        pulse_left_img = pulse_left * m_tick / len(section)
        pulse_right_img = pulse_right * m_tick / len(section)
        
        ax.axvline(x=pulse_left_img, linestyle='--', color='blue', alpha=0.6, linewidth=2)
        ax.axvline(x=pulse_right_img, linestyle='--', color='blue', alpha=0.6, linewidth=2)
    
    ax.text(
            0.70, 0.90,
            f'Dedispersed\nDM={dm:.2f} pc cm⁻³',
            transform=ax.transAxes,
            fontsize=20, fontweight='semibold', color='white', family='Times New Roman')
    
    line_length = int(0.01 * m_tick)
    a = np.linspace(0, line_length, 100)
    if len(masked_channels) > 0:
        for ii in range(len(masked_channels)):
            if 0 <= masked_channels[ii][0] < l_tick:
                ax.plot(a, np.ones_like(a) * masked_channels[ii][0], color='red', linewidth=2)
    
    add_ticks_all_sides(ax)
    
    ax.set_xlim(0, m_tick-1)
    ax.set_ylim(0, l_tick-1)
    
    ax.tick_params(axis='y', direction='out', length=6, width=1.5)


def plot_bandpass(ax, exp_normalized_data, startf, endf, centerf):
    """Plot bandpass power curve."""
    try:
        bandpass_ddm = np.sum(exp_normalized_data, axis=1)
        ysize = int(bandpass_ddm.size)
        zero = min(np.mean(bandpass_ddm[0:ysize//5]), np.mean(bandpass_ddm[ysize-ysize//5:ysize]))
        y = bandpass_ddm - zero
        a = np.linspace(0, 1, len(y))
        ax.plot(y, a, color='black', label="Data", alpha=0.7, linewidth=1.5)
    except:
        y = np.sum(exp_normalized_data, axis=1)
        a = np.linspace(0, 1, len(y))
        ax.plot(y, a, color='black', label="Data", alpha=0.7, linewidth=1.5)
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    max_min_delta = np.max(y) - np.min(y)
    ax.set_xlim([np.min(y)-max_min_delta/7., np.max(y)+max_min_delta/7.])
    ax.set_ylim([0, 1])
    
    add_ticks_all_sides(ax)


def plot_raw_waterfall(ax, exp_normalized_raw_data, mjdstart, start, samppersubint, 
                      tsamp, downsamp, startf, endf, masked_channels, dm, 
                      startn, endn, tboxwindow, delay_second, plot_vmin, plot_vmax):
    """Plot raw data waterfall."""
    im = ax.imshow(exp_normalized_raw_data, aspect='auto', alpha=1.0, cmap=plt.cm.viridis,
                   origin="lower", vmin=plot_vmin, vmax=plot_vmax)
    
    l_tick, m_tick = exp_normalized_raw_data.shape
    
    ax.set_xlabel(f'Time offset (s) + MJD {mjdstart}', fontsize=20, fontweight='regular', family='Times New Roman')
    ax.set_ylabel('Frequency (MHz)', fontsize=20, fontweight='regular', family='Times New Roman')
    
    floattimelabel = np.linspace(0, m_tick-1, 7)
    time_offset_seconds = np.linspace(0, (endn - startn)*samppersubint*tsamp, 7)
    strtimelabel = [f"{t:.2f}" for t in time_offset_seconds]
    
    floatfreqlabel = np.linspace(0, l_tick-1, 5)
    strfreqlabel = [str(int(freqpoint)) for freqpoint in np.linspace(startf, endf, 5)]
    
    ax.set_xticks(floattimelabel)
    ax.set_xticklabels(strtimelabel, fontsize=16, family='Times New Roman')
    ax.set_yticks(floatfreqlabel)
    ax.set_yticklabels(strfreqlabel, fontsize=16, family='Times New Roman')
    ax.tick_params(which='major', width=1, length=6)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    
    ax.axvline(x=(float(start/samppersubint/tsamp)-int(start/samppersubint/tsamp)+start_subint_num)*samppersubint/downsamp, 
               linestyle='--', color='white', linewidth=1.5)
    ax.text(0.85, 0.9, 'Waterfall', transform=ax.transAxes, fontsize=26, fontweight='semibold', 
            color='white', family='Times New Roman')
    
    line_length = int(0.01 * m_tick)
    a = np.linspace(0, line_length, 100)
    if len(masked_channels) > 0:
        masked_channels_meshgrid, b = np.meshgrid(masked_channels, a)
        for ii in range(len(masked_channels)):
            ax.plot(a, masked_channels_meshgrid[:, ii], color='red')
    
    try:
        me, le = exp_normalized_raw_data.shape
        a = np.linspace(0, me-1, 10)
        dytime_l = -startn*samppersubint*tsamp + endn*samppersubint*tsamp
        if dytime_l > 0:
            dytime_raw = 4148808.0 * dm * ((1. / (startf + a/(me-1)*(endf-startf))**2 - 1./(endf**2)) / 1000.) / dytime_l * (le-1)
            b = (float(start/tsamp/samppersubint) - int(start/tsamp/samppersubint) + start_subint_num) * samppersubint/downsamp + dytime_raw
            
            valid_indices = (b >= 0) & (b <= le-1)
            if np.any(valid_indices):
                ax.plot(b[valid_indices] - tboxwindow/dytime_l*(le-1)/1000, a[valid_indices], 
                       linestyle='-', color='white', linewidth=1., alpha=0.3)
                ax.plot(b[valid_indices] + tboxwindow/dytime_l*(le-1)/1000, a[valid_indices], 
                       linestyle='-', color='white', linewidth=1., alpha=0.3)
    except Exception as e:
        print(f"Error plotting frequency sweep: {e}")
    
    ax.set_xlim([0, m_tick-1])
    ax.set_ylim([0, l_tick-1])


def plot_file_info(fig, pulsar, mjd_toa, dm):
    """Add file information to figure."""
    mjd_toa_float = float(mjd_toa) if isinstance(mjd_toa, str) else mjd_toa
    dm_float = float(dm) if isinstance(dm, str) else dm
    
    name = f'Filename: {pulsar}_ToA{mjd_toa_float:.10f}_DM{dm_float:.2f}.png'
    
    fig.text(0.42, 0.94, name, fontsize=22, fontweight='bold',
             family='Times New Roman', ha='center', va='top')


def waterfall(
        rawdatafile,
        start: float,
        duration: float,
        dm: float,
        fd: int,
        downsamp: int,
        tboxwindow: int,
        scaling: float,
        basebandStd: float,
        plot_dir: str = "./",
        ra_dec: str = None,
        pulsar: str = None,
        confidence: float = None,
        compute_only: bool = False,
        plot_only: bool = False,
        ):
    """Main waterfall plotting function."""
    
    try:
        fchannel = rawdatafile.freqs
        nsub = int(rawdatafile.nchan/fd)
        fch1 = rawdatafile.freqs[0]
        telescope = rawdatafile.telescope
        
        if rawdatafile.frequencies[0] < rawdatafile.frequencies[-1]:
            startf = rawdatafile.frequencies[0]
            endf = rawdatafile.frequencies[-1] + (rawdatafile.frequencies[1] - rawdatafile.frequencies[0])
            df = rawdatafile.frequencies[1] - rawdatafile.frequencies[0]
            start = start - 4148808.0 * dm * (1. / (fchannel[-1])**2 - 1. / (endf)**2) / 1000.
        else:
            startf = rawdatafile.frequencies[-1]
            endf = rawdatafile.frequencies[0] - (rawdatafile.frequencies[-1] - rawdatafile.frequencies[-2])
            df = rawdatafile.frequencies[-2] - rawdatafile.frequencies[-1]
            start = start - 4148808.0 * dm * (1. / (fchannel[0])**2 - 1. / (endf)**2) / 1000.
        
        centerf = int((endf + startf) / 2.)
        fn = rawdatafile.filename
        tsamp = rawdatafile.tsamp
        nsubint = rawdatafile.nsubints
        samppersubint = rawdatafile.nsamp_per_subint
        
        startn = int(start/tsamp/samppersubint) - start_subint_num
        if startn < 0:
            startn = int(start/tsamp/samppersubint)
        
        dytime = 4148808.0 * dm * (1./(startf)**2 - 1./(endf)**2) / 1000.
        endn = int(dytime/tsamp/samppersubint + duration) + startn
        endn = int(min(endn, int(nsubint) - 1))
        
        sampperbox = int(tboxwindow/1000./tsamp)
        ref_freq = rawdatafile.freqs.max()
        fn = rawdatafile.filename
        
        mjdstart = "%.11f" % (Decimal(float(rawdatafile.mjd)) + Decimal(float(start)/secperday))
        starttime_float = float(mjdstart)
        t_header = Time(starttime_float, format='mjd', scale='utc')
        tmp_t_header = str(t_header.datetime).split('.')
        
        if len(tmp_t_header) == 1:
            tmp_t_header_str = tmp_t_header[0] + '.000001'
        else:
            tmp_t_header_str = tmp_t_header[0] + '.' + tmp_t_header[1]
        
        starttime_utc = dt.datetime.strptime(tmp_t_header_str, "%Y-%m-%d %H:%M:%S.%f")
        starttime_bj = starttime_utc + dt.timedelta(hours=8)
        
        start_bin = startn * samppersubint
        nbinsextra = endn * samppersubint - startn * samppersubint
        if (start_bin + nbinsextra) > rawdatafile.nspec - 1:
            nbinsextra = rawdatafile.nspec - 1 - start_bin
        
        original_data = rawdatafile.get_spectra(start_bin, nbinsextra)
        original_data.data = original_data.data[::-1, :]
        original_data.freqs = original_data.freqs[::-1]
        
        original_data_copy = cp.deepcopy(original_data)
        data_raw_copy = cp.deepcopy(original_data)
        
        if (nsub is not None) and (dm is not None):
            data_raw_copy.subband(nsub)
            data_raw_copy.downsample(downsamp)
        downsampled_data = data_raw_copy.data
        
        data_raw_copy_temp = cp.deepcopy(data_raw_copy)
        data_raw_copy = data_raw_copy.scaled_mean_std()
        normalized_raw_data = data_raw_copy.data
        data_raw_copy_temp.mask_baseband(basebandStd=basebandStd)
        
        if dm:
            original_data.dedisperse(dm, padval='mean')
        if (nsub is not None) and (dm is not None):
            original_data.subband(nsub)
            original_data.downsample(downsamp)
        dedispersed_data = original_data.data
        
        original_data = original_data.scaled_mean_std()
        normalized_data = original_data.data
        
        masked_channels = data_raw_copy_temp.Maskchannel
        masked_channels = masked_channels.reshape(-1, 1)
        rfi_zap = data_raw_copy_temp.RFIzap
        delay_second = original_data.delay_second
        
        for ii in range(len(masked_channels)):
            normalized_data[masked_channels[ii], :] = 0.
            normalized_raw_data[masked_channels[ii], :] = 0.
        
        exp_normalized_raw_data = np.exp(normalized_raw_data/np.max(normalized_data) + 1)
        
        startbox = int((start/tsamp - startn*samppersubint - sampperbox)/downsamp)
        endbox = int((start/tsamp - startn*samppersubint + sampperbox)/downsamp)
        ny, nx = exp_normalized_raw_data.shape
        
        if startbox < 0:
            startbox = 0
        if nx - startbox >= int(2*sampperbox/downsamp):
            startbox = startbox
        else:
            startbox = nx - int(2*sampperbox/downsamp)
        
        exp_normalized_data = np.exp(normalized_data[:, startbox:endbox]/np.max(normalized_data[:, startbox:endbox]) + 1)
        
        profile = np.sum(normalized_data[:, startbox:endbox], axis=0)
        section = np.arange(0, profile.shape[0], 1)
        profile = profile[0:len(section)]
        profile_std = np.std(profile[0:int(0.3*len(profile))])
        py_max = np.max(profile - np.min(profile))
        
        profile_y = profile - np.min(profile)
        profile_y = profile_y / py_max
        
        fit_results, pulse_left, pulse_right, snr, width_ms = fit_single_gaussian(
            profile, profile_y, profile_std, section, py_max, tsamp, downsamp,
            None, tboxwindow, sampperbox)
        
        if compute_only:
            if snr is not None and width_ms is not None:
                return (snr, width_ms, float(mjdstart))
            else:
                return (0.0, 0.0, float(mjdstart))
        
        exp_normalized_data_avg = np.mean(exp_normalized_data)
        exp_normalized_data_std = np.std(exp_normalized_data)
        plot_vmin = exp_normalized_data_avg - 1.5 * exp_normalized_data_std * scaling
        plot_vmax = exp_normalized_data_avg + 3. * exp_normalized_data_std * scaling
        
        fig = plt.figure(figsize=(14, 18))
        
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        
        ax_ts = plt.axes((0.1, 0.82, 0.53, 0.10))
        ax_im = plt.axes((0.1, 0.40, 0.53, 0.42))
        ax_spec = plt.axes((0.63, 0.40, 0.18, 0.42))
        ax_orig = plt.axes((0.1, 0.08, 0.75, 0.28))
        
        plot_profile_with_precomputed(ax_ts, profile, section, profile_std, py_max,
                                     normalized_data[:, startbox:endbox], fit_results,
                                     pulse_left, pulse_right, snr, width_ms)
        
        plot_dedispersed_waterfall(ax_im, start, exp_normalized_data, startf, endf, dm, tboxwindow,
                                  masked_channels, scaling, plot_vmin, plot_vmax, pulse_left, pulse_right, section)
        
        plot_bandpass(ax_spec, exp_normalized_data, startf, endf, centerf)
        
        plot_raw_waterfall(ax_orig, exp_normalized_raw_data, mjdstart, start, samppersubint,
                          tsamp, downsamp, startf, endf, masked_channels, dm,
                          startn, endn, tboxwindow, delay_second, plot_vmin, plot_vmax)
        
        if snr is not None and width_ms is not None:
            ax_spec.text(1.2, 0.8, f"SNR: {snr:.1f}", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
            ax_spec.text(1.2, 0.55, f"Width: {width_ms:.2f}ms", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
        
        if confidence is not None:
            ax_spec.text(1.2, 0.3, f"Confidence: {confidence*100:.2f}%", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
        
        if pulsar is None:
            pulsar = extract_pulsar_name(fn)
        
        plot_file_info(fig, pulsar, mjdstart, dm)
        
        base_fn = fn.split('/')[-1]
        if rawdatafile.ext == ".fits":
            prefix = base_fn[:-5]
        elif rawdatafile.ext == ".fil":
            prefix = base_fn[:-4]
        else:
            raise ValueError(f"Unsupported file extension: {rawdatafile.ext}")
        imgfilename = f'{prefix}_{pulsar}_toa{start:.5f}_DM{dm:.2f}.png'
        
        # 保存图片
        if plot_dir is not None:
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            save_path = os.path.join(plot_dir, imgfilename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Image saved: {save_path}")
        
        # 保存后关闭图像
        plt.close(fig)
        
        return imgfilename if plot_dir is not None else None
        
    except Exception as e:
        print(f"Error in waterfall: {e}")
        traceback.print_exc()
        if compute_only:
            return None
        return None


def waterfall_jupyter(
        rawdatafile,
        start: float,
        duration: float,
        dm: float,
        fd: int,
        downsamp: int,
        tboxwindow: int,
        scaling: float,
        basebandStd: float,
        pulsar: str = None,
        confidence: float = None,
        display_width: int = 525,
        ):
    """
    在 Jupyter 中直接显示 waterfall 图，不保存文件。
    
    参数:
        display_width: 显示宽度（像素），默认525px约为原图的1/4
                      设置为None则显示原始大小
    
    返回 IPython.display.Image 对象，单元格最后一行调用此函数会自动显示图片。
    """
    import io
    from PIL import Image as PILImage
    import matplotlib.pyplot as plt
    
    try:
        fchannel = rawdatafile.freqs
        nsub = int(rawdatafile.nchan/fd)
        fch1 = rawdatafile.freqs[0]
        telescope = rawdatafile.telescope
        
        if rawdatafile.frequencies[0] < rawdatafile.frequencies[-1]:
            startf = rawdatafile.frequencies[0]
            endf = rawdatafile.frequencies[-1] + (rawdatafile.frequencies[1] - rawdatafile.frequencies[0])
            df = rawdatafile.frequencies[1] - rawdatafile.frequencies[0]
            start = start - 4148808.0 * dm * (1. / (fchannel[-1])**2 - 1. / (endf)**2) / 1000.
        else:
            startf = rawdatafile.frequencies[-1]
            endf = rawdatafile.frequencies[0] - (rawdatafile.frequencies[-1] - rawdatafile.frequencies[-2])
            df = rawdatafile.frequencies[-2] - rawdatafile.frequencies[-1]
            start = start - 4148808.0 * dm * (1. / (fchannel[0])**2 - 1. / (endf)**2) / 1000.
        
        centerf = int((endf + startf) / 2.)
        fn = rawdatafile.filename
        tsamp = rawdatafile.tsamp
        nsubint = rawdatafile.nsubints
        samppersubint = rawdatafile.nsamp_per_subint
        
        startn = int(start/tsamp/samppersubint) - start_subint_num
        if startn < 0:
            startn = int(start/tsamp/samppersubint)
        
        dytime = 4148808.0 * dm * (1./(startf)**2 - 1./(endf)**2) / 1000.
        endn = int(dytime/tsamp/samppersubint + duration) + startn
        endn = int(min(endn, int(nsubint) - 1))
        
        sampperbox = int(tboxwindow/1000./tsamp)
        ref_freq = rawdatafile.freqs.max()
        fn = rawdatafile.filename
        
        mjdstart = "%.11f" % (Decimal(float(rawdatafile.mjd)) + Decimal(float(start)/secperday))
        starttime_float = float(mjdstart)
        t_header = Time(starttime_float, format='mjd', scale='utc')
        tmp_t_header = str(t_header.datetime).split('.')
        
        if len(tmp_t_header) == 1:
            tmp_t_header_str = tmp_t_header[0] + '.000001'
        else:
            tmp_t_header_str = tmp_t_header[0] + '.' + tmp_t_header[1]
        
        starttime_utc = dt.datetime.strptime(tmp_t_header_str, "%Y-%m-%d %H:%M:%S.%f")
        starttime_bj = starttime_utc + dt.timedelta(hours=8)
        
        start_bin = startn * samppersubint
        nbinsextra = endn * samppersubint - startn * samppersubint
        if (start_bin + nbinsextra) > rawdatafile.nspec - 1:
            nbinsextra = rawdatafile.nspec - 1 - start_bin
        
        original_data = rawdatafile.get_spectra(start_bin, nbinsextra)
        original_data.data = original_data.data[::-1, :]
        original_data.freqs = original_data.freqs[::-1]
        
        original_data_copy = cp.deepcopy(original_data)
        data_raw_copy = cp.deepcopy(original_data)
        
        if (nsub is not None) and (dm is not None):
            data_raw_copy.subband(nsub)
            data_raw_copy.downsample(downsamp)
        downsampled_data = data_raw_copy.data
        
        data_raw_copy_temp = cp.deepcopy(data_raw_copy)
        data_raw_copy = data_raw_copy.scaled_mean_std()
        normalized_raw_data = data_raw_copy.data
        data_raw_copy_temp.mask_baseband(basebandStd=basebandStd)
        
        if dm:
            original_data.dedisperse(dm, padval='mean')
        if (nsub is not None) and (dm is not None):
            original_data.subband(nsub)
            original_data.downsample(downsamp)
        dedispersed_data = original_data.data
        
        original_data = original_data.scaled_mean_std()
        normalized_data = original_data.data
        
        masked_channels = data_raw_copy_temp.Maskchannel
        masked_channels = masked_channels.reshape(-1, 1)
        rfi_zap = data_raw_copy_temp.RFIzap
        delay_second = original_data.delay_second
        
        for ii in range(len(masked_channels)):
            normalized_data[masked_channels[ii], :] = 0.
            normalized_raw_data[masked_channels[ii], :] = 0.
        
        exp_normalized_raw_data = np.exp(normalized_raw_data/np.max(normalized_data) + 1)
        
        startbox = int((start/tsamp - startn*samppersubint - sampperbox)/downsamp)
        endbox = int((start/tsamp - startn*samppersubint + sampperbox)/downsamp)
        ny, nx = exp_normalized_raw_data.shape
        
        if startbox < 0:
            startbox = 0
        if nx - startbox >= int(2*sampperbox/downsamp):
            startbox = startbox
        else:
            startbox = nx - int(2*sampperbox/downsamp)
        
        exp_normalized_data = np.exp(normalized_data[:, startbox:endbox]/np.max(normalized_data[:, startbox:endbox]) + 1)
        
        profile = np.sum(normalized_data[:, startbox:endbox], axis=0)
        section = np.arange(0, profile.shape[0], 1)
        profile = profile[0:len(section)]
        profile_std = np.std(profile[0:int(0.3*len(profile))])
        py_max = np.max(profile - np.min(profile))
        
        profile_y = profile - np.min(profile)
        profile_y = profile_y / py_max
        
        fit_results, pulse_left, pulse_right, snr, width_ms = fit_single_gaussian(
            profile, profile_y, profile_std, section, py_max, tsamp, downsamp,
            None, tboxwindow, sampperbox)
        
        exp_normalized_data_avg = np.mean(exp_normalized_data)
        exp_normalized_data_std = np.std(exp_normalized_data)
        plot_vmin = exp_normalized_data_avg - 1.5 * exp_normalized_data_std * scaling
        plot_vmax = exp_normalized_data_avg + 3. * exp_normalized_data_std * scaling
        
        fig = plt.figure(figsize=(14, 18))
        
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        
        ax_ts = plt.axes((0.1, 0.82, 0.53, 0.10))
        ax_im = plt.axes((0.1, 0.40, 0.53, 0.42))
        ax_spec = plt.axes((0.63, 0.40, 0.18, 0.42))
        ax_orig = plt.axes((0.1, 0.08, 0.75, 0.28))
        
        plot_profile_with_precomputed(ax_ts, profile, section, profile_std, py_max,
                                     normalized_data[:, startbox:endbox], fit_results,
                                     pulse_left, pulse_right, snr, width_ms)
        
        plot_dedispersed_waterfall(ax_im, start, exp_normalized_data, startf, endf, dm, tboxwindow,
                                  masked_channels, scaling, plot_vmin, plot_vmax, pulse_left, pulse_right, section)
        
        plot_bandpass(ax_spec, exp_normalized_data, startf, endf, centerf)
        
        plot_raw_waterfall(ax_orig, exp_normalized_raw_data, mjdstart, start, samppersubint,
                          tsamp, downsamp, startf, endf, masked_channels, dm,
                          startn, endn, tboxwindow, delay_second, plot_vmin, plot_vmax)
        
        if snr is not None and width_ms is not None:
            ax_spec.text(1.2, 0.8, f"SNR: {snr:.1f}", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
            ax_spec.text(1.2, 0.55, f"Width: {width_ms:.2f}ms", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
        
        if confidence is not None:
            ax_spec.text(1.2, 0.3, f"Confidence: {confidence*100:.2f}%", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
        
        if pulsar is None:
            pulsar = extract_pulsar_name(fn)
        
        plot_file_info(fig, pulsar, mjdstart, dm)
        
        # 保存到内存缓冲区
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        # 返回 IPython Image 对象，设置宽度
        try:
            from IPython.display import Image
            img_data = buf.getvalue()
            if display_width is not None:
                return Image(data=img_data, width=display_width)
            else:
                return Image(data=img_data)
        except ImportError:
            print("Error: Not in Jupyter environment")
            return None
            
    except Exception as e:
        print(f"❌ Error in waterfall_jupyter: {e}")
        traceback.print_exc()
        return None
        return None


def show_waterfall_in_jupyter(image_path):
    """
    在 Jupyter Notebook 中显示已保存的 waterfall 图片。
    直接返回 IPython Image 对象，作为单元格输出。
    
    Parameters:
    -----------
    image_path : str
        图片文件路径
        
    Returns:
    --------
    IPython.display.Image or None
        图片对象，可直接在 Jupyter 中显示
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    try:
        from IPython.display import Image
        return Image(filename=image_path)
    except ImportError:
        print("Error: Not in Jupyter environment")
        return None
    except Exception as e:
        print(f"Error: Failed to load image: {e}")
        return None



def main():
    """Main function."""
    fn = sys.argv[1]
    if fn.endswith(".fil"):
        filetype = "filterbank"
        rawdatafile = filterbank.FilterbankFile(fn)
    elif fn.endswith(".fits"):
        filetype = "psrfits"
        rawdatafile = psrfits.PsrfitsFile(fn)
    else:
        raise ValueError("Cannot recognize data file type from "
                         "extension. (Only '.fits' and '.fil' "
                         "are supported.)")
    

if __name__ == '__main__':
    parser = optparse.OptionParser(prog="waterfaller.py",
                        version="v0.9 Patrick Lazarus (Aug. 19, 2011)",
                        usage="%prog [OPTIONS] INFILE",
                        description="Create a waterfall plot to show the "
                                    "frequency sweep of a single pulse "
                                    "in psrFits data.")
    
    parser.add_option('-T', '--start-time', dest='start', type='float',
                        help="Time into observation (in seconds) at which to start plot.")
    parser.add_option('-t', '--duration', dest='duration', type='float',
                        help="Duration (in seconds) of plot.")
    parser.add_option('-s', '--nsub', dest='nsub', type='int',
                        help="Number of subbands to use. Must be a factor of number of channels. "
                             "(Default: number of channels)", default=None)
    parser.add_option('-d', '--dm', dest='dm', type='float',
                        help="DM to use when dedispersing data for plot. (Default: 0 pc/cm^3)", default=0.0)
    parser.add_option('--downsamp', dest='downsamp', type='int',
                        help="Factor to downsample data by. (Default: 1).", default=1)
    parser.add_option('--width-bins', dest='width_bins', type='int',
                        help="Smooth each channel/subband with a boxcar this many bins wide. "
                             "(Default: Don't smooth)", default=1)
    parser.add_option('--tboxwindow', dest='tboxwindow', type='int',
                        help='Time window parameter', default=1)
    parser.add_option('--basebandStd', dest='basebandStd', type=float, default=1.0, help='basebandStd')
    parser.add_option('--scaling', dest='scaling', type=float, default=0.8, help='scaling')
    parser.add_option('--plot-dir', dest='plot_dir',
                        help="The directory where plots will be saved. (Default: ./plots/)",
                        default='./plots/')
    parser.add_option('--ra_dec', dest='ra_dec', type='str', default="",
                        help='Input RA and DEC coordinates')
    parser.add_option('--pulsar', dest='pulsar', type='str', default=None,
                        help='Pulsar name for output filename')
   
    options, args = parser.parse_args()
    
    if not hasattr(options, 'start'):
        raise ValueError("Start time (-T/--start-time) must be given on command line!")
    if (not hasattr(options, 'duration')) and (not hasattr(options, 'nbins')):
        raise ValueError("One of duration (-t/--duration) and num bins (-n/--nbins) "
                        "must be given on command line!")
    
    main()