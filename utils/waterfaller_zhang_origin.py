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
#---------------------------------------------------------------------------------------------------------------------------------------
# python  ../dyn/waterfaller_zhang_250521.py 2022-06-10-2204260000000000000000.000000.03.fil  -T 1299.49999  -t 500   -d 1458.15  --tboxwindow 200  --width-bins 4  -s 336 --downsamp 1  --plot-dir res_plot11/  --delta-dm-max 100.0 --delta-time 50 --n-dm 10
# python  ../dyn/waterfaller_zhang_250521.py Dec+5757_00_05_arcdrift-M11_0221_2bit.fits  -T 3.195  -t 4   -d 118.6  --tboxwindow 100  --width-bins 4  -s 256 --downsamp 16  --plot-dir res_plot12/  --delta-dm-max 30 --delta-time 30 --n-dm 5
#----------------------------------------------------------------------------------------------------------------------------------------
input_snr = 1.0
ra_dec = None
start_subint_num = 1

starttime = dt.datetime.now()
secperday = 3600. * 24.
accel_patch = int(os.getenv("ACCEL", 0))
fits_id_matcher = re.compile(r"M[\d]{2}([\d]{4})")

def extract_pulsar_name(filename):
    """从文件名提取pulsar名称"""
    # 去掉路径
    basename = os.path.basename(filename)
    # print("basename:",basename)
    # 使用正则表达式移除所有常见扩展名
    basename = re.sub(r'\.(fits|fil|npy)$', '', basename)
    
    # 检查是否包含FRB
    if 'FRB' in basename:
        # 提取FRB名称 (如 FRB20201124A, FRB121102)
        frb_match = re.search(r'FRB\d+[A-Z]?', basename)
        if frb_match:
            return frb_match.group(0)
    
    # 否则提取源名称（第一个下划线前的部分）
    # 例如：J1843-0459_Dec-0452_arcdrift-M10_1240_2bit -> J1843-0459
    parts = basename.split('_')
    # print(parts)
    if len(parts) > 0:
        source_name = parts[0]
        # # 如果源名称看起来像pulsar名称（J开头或B开头），直接返回
        # if source_name.startswith(('J', 'B', 'PSR')):
        #     return source_name
        # 否则添加PSR_前缀
        return f"PSR_{source_name}"
    
    return "PSR_Unknown"

def fit_single_gaussian(profile, profile_y, profile_std, section, py_max, tsamp, downsamp, ax, tboxwindow, sampperbox):
    """单高斯拟合函数 - 支持不绘图模式"""
    # 找到最高峰
    max_idx = np.argmax(profile_y)
    max_height = profile_y[max_idx]
    
    # 估算初始参数
    # 找到半高宽度来估算sigma
    half_max = max_height / 2.0
    left_idx = max_idx
    right_idx = max_idx
    
    # 向左找到半高点
    while left_idx > 0 and profile_y[left_idx] > half_max:
        left_idx -= 1
    # 向右找到半高点  
    while right_idx < len(profile_y)-1 and profile_y[right_idx] > half_max:
        right_idx += 1
    
    estimated_sigma = (right_idx - left_idx) / 2.355  # FWHM to sigma conversion
    
    def single_gaussian(x, a, mu, sigma, d):
        return a * np.exp(-(x-mu)**2/(2*sigma**2)) + d
    
    try:
        # 设置拟合边界
        bounds_lower = [0, max_idx - estimated_sigma*3, estimated_sigma/3, 0]
        bounds_upper = [max_height*2, max_idx + estimated_sigma*3, estimated_sigma*3, profile_std]
        
        # 初始猜测参数
        initial_guess = [max_height, max_idx, estimated_sigma, 0]
        
        params, pcov = curve_fit(single_gaussian, section, profile_y, 
                               p0=initial_guess, bounds=(bounds_lower, bounds_upper), maxfev=10000)
        
        y_fit = single_gaussian(section, *params)
        
        # 计算拟合优度
        y_mean = np.mean(profile_y)
        ss_total = np.sum((profile_y - y_mean) ** 2)
        ss_residual = np.sum((profile_y - y_fit) ** 2)
        goodness = 1 - (ss_residual / ss_total)
        
        a, mu, sigma, d = params
        
        # 只有当ax不为None时才绘图
        if ax is not None:
            # 绘制垂直线标记脉冲边界
            ax.axvline(x=pulse_left, linestyle='--', color='blue', alpha=0.6, linewidth=2, label='On Pulse')
            ax.axvline(x=pulse_right, linestyle='--', color='blue', alpha=0.6, linewidth=2)
        
        # 计算SNR和脉冲宽度
        snr = abs(a * py_max / profile_std)
        width_ms = abs(2.0 * math.sqrt(2 * math.log(2)) * sigma * tsamp * downsamp * 1e3)  # FWHM in ms
        
        # 计算脉冲边界 (基于3σ)
        pulse_left = mu - 3*sigma
        pulse_right = mu + 3*sigma
        
        return params, pulse_left, pulse_right, snr, width_ms
        
    except Exception as e:
        print(f"高斯拟合失败: {e}")
        return None, None, None, None, None



def add_ticks_all_sides(ax, exclude_raw=False):
    """为所有子图四周添加刻度线"""
    if exclude_raw:
        return
    
    # 获取当前刻度位置
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    # 设置刻度参数 - 向内
    ax.tick_params(axis='both', which='major', direction='in', 
                   top=True, right=True, bottom=True, left=True,
                   length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in',
                   top=True, right=True, bottom=True, left=True, 
                   length=3, width=0.5)
    
    # 启用所有边的刻度
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
def plot_profile_with_precomputed(ax, profile, section, profile_std, py_max, normalized_data,
                                 fit_results, pulse_left, pulse_right, snr, width_ms):

    ax.plot(profile, '-k', linewidth=2)
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.fill_between(section, profile+1.5*profile_std, profile-1.5*profile_std,
                   alpha=0.5, facecolor="gray", color="white")
    
    # 如果有拟合结果，绘制脉冲边界
    if pulse_left is not None and pulse_right is not None:
        ax.axvline(x=pulse_left, linestyle='--', color='blue', alpha=0.6, linewidth=2, label='On Pulse')
        ax.axvline(x=pulse_right, linestyle='--', color='blue', alpha=0.6, linewidth=2)
    
    # 隐藏x轴刻度和标签
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
    """绘制脉冲轮廓图"""
    ax.plot(profile, '-k', linewidth=2)
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    ax.fill_between(section, profile+1.5*profile_std, profile-1.5*profile_std, 
                   alpha=0.5, facecolor="gray", color="white")

    # 进行单高斯拟合
    profile_y = profile - np.min(profile)
    profile_y = profile_y / py_max
    
    fit_results, pulse_left, pulse_right, snr, width_ms = fit_single_gaussian(
        profile, profile_y, profile_std, section, py_max, tsamp, downsamp, ax, tboxwindow, sampperbox)
    
    # 隐藏x轴刻度和标签
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0)
    ax.yaxis.set_visible(False)
    max_min_delta = np.max(profile) - np.min(profile)
    ax.set_xlim([np.min(section), np.max(section)])
    ax.set_ylim([np.min(profile)-max_min_delta/7., np.max(profile)+max_min_delta/7.])
    
    # 添加四周刻度
    add_ticks_all_sides(ax)
    
    ax.legend(loc='upper right', fontsize=10, prop={'family': 'Times New Roman'})
    
    return snr, width_ms, pulse_left, pulse_right  # 新增返回值

def plot_dedispersed_waterfall(ax, start, exp_normalized_data, startf, endf, dm, tboxwindow, 
                              masked_channels, scaling, plot_vmin, plot_vmax, pulse_left=None, pulse_right=None, section=None):
    """绘制消色散瀑布图"""
    im = ax.imshow(exp_normalized_data, aspect='auto', cmap=plt.cm.viridis, 
                   origin="lower", vmin=plot_vmin, vmax=plot_vmax)
    
    ax.set_xlabel('Time (ms) ', fontsize = 20, fontweight='regular')
    ax.set_ylabel('Frequency (MHz)', fontsize = 20, fontweight='regular')
    
    l_tick, m_tick = exp_normalized_data.shape
    
    # 设置频率刻度
    floattimelabelddm = np.linspace(0, m_tick-1, 11)
    strtimelabelddmtop = [str(timepoint) for timepoint in np.linspace(-tboxwindow, tboxwindow, 11).astype('int64')]
    floatfreqlabel = np.linspace(0, l_tick-1, 5)
    strfreqlabel = [str(int(freqpoint)) for freqpoint in np.linspace(startf, endf, 5)]
    
    ax.set_yticks(floatfreqlabel)
    ax.set_yticklabels(strfreqlabel, fontsize=16, family='Times New Roman')
    ax.set_xticks(floattimelabelddm)
    ax.set_xticklabels(strtimelabelddmtop, fontsize=16, family='Times New Roman')
    
    # 绘制脉冲边界垂直线
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
    
    # 绘制掩码通道
    line_length = int(0.01 * m_tick)
    a = np.linspace(0, line_length, 100)
    if len(masked_channels) > 0:
        for ii in range(len(masked_channels)):
            if 0 <= masked_channels[ii][0] < l_tick:  # 确保索引有效
                ax.plot(a, np.ones_like(a) * masked_channels[ii][0], color='red', linewidth=2)
    
    # 添加四周刻度
    add_ticks_all_sides(ax)
    
    # 明确设置轴范围，这是关键！
    ax.set_xlim(0, m_tick-1)
    ax.set_ylim(0, l_tick-1)
    
    # 不要使用 axis('tight')，因为它可能会覆盖我们的设置
    # 设置Y轴刻度线向外
    ax.tick_params(axis='y', direction='out', length=6, width=1.5)

def plot_bandpass(ax, exp_normalized_data, startf, endf, centerf):
    """绘制功率抛物线拟合"""
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
    
    # 添加四周刻度
    add_ticks_all_sides(ax)

def plot_raw_waterfall(ax, exp_normalized_raw_data, mjdstart, start, samppersubint, 
                      tsamp, downsamp, startf, endf, masked_channels, dm, 
                      startn, endn, tboxwindow, delay_second, plot_vmin, plot_vmax):
    """绘制原始数据瀑布图"""
    
    
    im = ax.imshow(exp_normalized_raw_data, aspect='auto', alpha=1.0, cmap=plt.cm.viridis,
                   origin="lower", vmin=plot_vmin, vmax=plot_vmax)
    
    l_tick, m_tick = exp_normalized_raw_data.shape
    
    # 使用偏移量显示
    ax.set_xlabel(f'Time offset (s) + MJD {mjdstart}', fontsize=20, fontweight='regular', family='Times New Roman')
    ax.set_ylabel('Frequency (MHz)', fontsize=20, fontweight='regular', family='Times New Roman')
    
    # 设置刻度 - 显示相对时间（秒）
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
    
    
    # 添加colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    
    ax.axvline(x=(float(start/samppersubint/tsamp)-int(start/samppersubint/tsamp)+start_subint_num)*samppersubint/downsamp, 
               linestyle='--', color='white', linewidth=1.5)
    ax.text(0.85, 0.9, 'Waterfall', transform=ax.transAxes, fontsize=26, fontweight='semibold', 
            color='white', family='Times New Roman')
    
    # 绘制掩码通道
    line_length = int(0.01 * m_tick)
    a = np.linspace(0, line_length, 100)
    if len(masked_channels) > 0:
        masked_channels_meshgrid, b = np.meshgrid(masked_channels, a)
        for ii in range(len(masked_channels)):
            ax.plot(a, masked_channels_meshgrid[:, ii], color='red')
    
    # 绘制频率扫描线
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
        print(f"绘制频率扫描线时出错: {e}")
    
    ax.set_xlim([0, m_tick-1])
    ax.set_ylim([0, l_tick-1])

def plot_file_info(fig, pulsar, mjd_toa, dm):
    """添加文件信息 - 加粗版本"""
    # 确保 mjd_toa 是浮点数
    mjd_toa_float = float(mjd_toa) if isinstance(mjd_toa, str) else mjd_toa
    dm_float = float(dm) if isinstance(dm, str) else dm
    
    name = f'Filename: {pulsar}_ToA{mjd_toa_float:.10f}_DM{dm_float:.2f}.png'
    
    # 文件名居中显示在顶部，加粗
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
        compute_only: bool = False,  # 新增：只计算参数 # type: ignore
        plot_only: bool = False,      # 新增：只绘图 # type: ignore
        ):
    """主要的瀑布图绘制函数
    
    Parameters:
    -----------
    rawdatafile : psrfits.PsrfitsFile or filterbank.FilterbankFile
        原始数据文件对象
    start : float
        开始时间 (秒)
    duration : float
        持续时间 (秒)
    dm : float
        色散量 (pc/cm³)
    fd : int
        频率降采样因子
    downsamp : int
        时间降采样因子
    tboxwindow : int
        时间窗口 (微秒)
    scaling : float
        缩放因子
    basebandStd : float
        基带标准差阈值
    plot_dir : str
        输出目录
    ra_dec : str
        RA/DEC坐标（可选）
    pulsar : str
        脉冲星名称（可选）
    confidence : float
        检测置信度（可选）
    compute_only : bool
        如果为True，只计算参数不绘图，返回(snr, width_ms, mjd_toa)
    plot_only : bool
        如果为True，只绘图不重复计算（用于多进程绘图）
    
    Returns:
    --------
    如果compute_only=True，返回 (snr, width_ms, mjd_toa)
    否则返回None
    """
    
    try:
        # 频率信息处理
        fchannel = rawdatafile.freqs
        nsub = int(rawdatafile.nchan/fd)
        fch1 = rawdatafile.freqs[0]
        telescope = rawdatafile.telescope
        
        # 判断频率排序方向并校正TOA
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
        
        # 时间参数
        tsamp = rawdatafile.tsamp
        nsubint = rawdatafile.nsubints
        samppersubint = rawdatafile.nsamp_per_subint
        
        # 时间范围设置
        startn = int(start/tsamp/samppersubint) - start_subint_num
        if startn < 0:
            startn = int(start/tsamp/samppersubint)
        
        dytime = 4148808.0 * dm * (1./(startf)**2 - 1./(endf)**2) / 1000.
        endn = int(dytime/tsamp/samppersubint + duration) + startn
        endn = int(min(endn, int(nsubint) - 1))
        
        sampperbox = int(tboxwindow/1000./tsamp)
        ref_freq = rawdatafile.freqs.max()
        fn = rawdatafile.filename
        
        # 时间信息处理
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
        
        # 数据读取和处理
        start_bin = startn * samppersubint
        nbinsextra = endn * samppersubint - startn * samppersubint
        if (start_bin + nbinsextra) > rawdatafile.nspec - 1:
            nbinsextra = rawdatafile.nspec - 1 - start_bin
        
        original_data = rawdatafile.get_spectra(start_bin, nbinsextra)
        original_data.data = original_data.data[::-1, :]
        original_data.freqs = original_data.freqs[::-1]
        
        # 创建各种数据副本
        original_data_copy = cp.deepcopy(original_data)
        data_raw_copy = cp.deepcopy(original_data)
        
        # 处理原始数据
        if (nsub is not None) and (dm is not None):
            data_raw_copy.subband(nsub)
            data_raw_copy.downsample(downsamp)
        downsampled_data = data_raw_copy.data
        
        data_raw_copy_temp = cp.deepcopy(data_raw_copy)
        data_raw_copy = data_raw_copy.scaled_mean_std()
        normalized_raw_data = data_raw_copy.data
        data_raw_copy_temp.mask_baseband(basebandStd=basebandStd)
        
        # 消色散处理
        if dm:
            original_data.dedisperse(dm, padval='mean')
        if (nsub is not None) and (dm is not None):
            original_data.subband(nsub)
            original_data.downsample(downsamp)
        dedispersed_data = original_data.data
        
        original_data = original_data.scaled_mean_std()
        normalized_data = original_data.data
        
        # 掩码处理
        masked_channels = data_raw_copy_temp.Maskchannel
        masked_channels = masked_channels.reshape(-1, 1)
        rfi_zap = data_raw_copy_temp.RFIzap
        delay_second = original_data.delay_second
        
        # 应用掩码
        for ii in range(len(masked_channels)):
            normalized_data[masked_channels[ii], :] = 0.
            normalized_raw_data[masked_channels[ii], :] = 0.
        
        # 指数归一化
        exp_normalized_raw_data = np.exp(normalized_raw_data/np.max(normalized_data) + 1)
        
        # 计算绘图范围
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
        
        # 脉冲轮廓处理
        profile = np.sum(normalized_data[:, startbox:endbox], axis=0)
        section = np.arange(0, profile.shape[0], 1)
        profile = profile[0:len(section)]
        profile_std = np.std(profile[0:int(0.3*len(profile))])
        py_max = np.max(profile - np.min(profile))
        
        # 高斯拟合计算
        profile_y = profile - np.min(profile)
        profile_y = profile_y / py_max
        
        fit_results, pulse_left, pulse_right, snr, width_ms = fit_single_gaussian(
            profile, profile_y, profile_std, section, py_max, tsamp, downsamp,
            None, tboxwindow, sampperbox)
        
        # 如果只计算参数，返回结果
        if compute_only:
            if snr is not None and width_ms is not None:
                return (snr, width_ms, float(mjdstart))
            else:
                # 如果拟合失败，返回默认值
                return (0.0, 0.0, float(mjdstart))
        
        # ========== 以下是绘图部分 ==========
        
        # 计算绘图参数
        exp_normalized_data_avg = np.mean(exp_normalized_data)
        exp_normalized_data_std = np.std(exp_normalized_data)
        plot_vmin = exp_normalized_data_avg - 1.5 * exp_normalized_data_std * scaling
        plot_vmax = exp_normalized_data_avg + 3. * exp_normalized_data_std * scaling
        
        # 开始绘图
        fig = plt.figure(figsize=(14, 18))
        
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        
        # 子图布局
        ax_ts = plt.axes((0.1, 0.82, 0.53, 0.10))
        ax_im = plt.axes((0.1, 0.40, 0.53, 0.42))
        ax_spec = plt.axes((0.63, 0.40, 0.18, 0.42))
        ax_orig = plt.axes((0.1, 0.08, 0.75, 0.28))
        
        # 绘制各个子图
        plot_profile_with_precomputed(ax_ts, profile, section, profile_std, py_max,
                                     normalized_data[:, startbox:endbox], fit_results,
                                     pulse_left, pulse_right, snr, width_ms)
        
        plot_dedispersed_waterfall(ax_im, start, exp_normalized_data, startf, endf, dm, tboxwindow,
                                  masked_channels, scaling, plot_vmin, plot_vmax, pulse_left, pulse_right, section)
        
        plot_bandpass(ax_spec, exp_normalized_data, startf, endf, centerf)
        
        plot_raw_waterfall(ax_orig, exp_normalized_raw_data, mjdstart, start, samppersubint,
                          tsamp, downsamp, startf, endf, masked_channels, dm,
                          startn, endn, tboxwindow, delay_second, plot_vmin, plot_vmax)
        
        # 添加参数信息（包括置信度）
        if snr is not None and width_ms is not None:
            ax_spec.text(1.2, 0.8, f"SNR: {snr:.1f}", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
            ax_spec.text(1.2, 0.55, f"Width: {width_ms:.2f}ms", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
        
        # 显示置信度
        if confidence is not None:
            ax_spec.text(1.2, 0.3, f"Confidence: {confidence*100:.2f}%", fontsize=22, ha='center', va='center',
                        fontweight='semibold', transform=ax_ts.transAxes, family='Times New Roman')
        
        # 处理pulsar名称
        if pulsar is None:
            pulsar = extract_pulsar_name(fn)
        
        # 添加文件信息
        plot_file_info(fig, pulsar, mjdstart, dm)
        
        # 保存图像
        imgfilename = f'{pulsar}_toa{start:.5f}_DM{dm:.2f}.png'
        
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        plt.savefig(os.path.join(plot_dir, imgfilename), dpi=80, bbox_inches='tight')
        plt.close(fig)
        
        return None
        
    except Exception as e:
        print(f"Error in waterfall: {e}")
        traceback.print_exc()
        if compute_only:
            return None
        return None

def main():
    """主函数"""
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
    
    # 添加所有选项参数
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
    parser.add_option('--basebandStd',dest='basebandStd', type=float,default=1.0, help='basebandStd')
    parser.add_option('--scaling',dest='scaling', type=float, default=0.8, help='scaling')
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



