"""
Author: Bin Zhang
Date: 2025.8.11
"""

from builtins import str
from builtins import range
from builtins import object
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from presto import psr_utils
from scipy import stats

class Spectra(object):
    """A class to store spectra. This is mainly to provide
        reusable functionality.
    """
    def __init__(self, freqs, dt, data, starttime=0, dm=0):
        """Spectra constructor.
            
            Inputs:
                freqs: Observing frequencies for each channel.
                dt: Sample time (in seconds).
                data: A 2D numpy array containing pulsar data.
                        Axis 0 should contain channels. (e.g. data[0,:])
                        Axis 1 should contain spectra. (e.g. data[:,0])
                starttime: Start time (in seconds) of the spectra
                        with respect to the start of the observation.
                        (Default: 0).
                dm: Dispersion measure (in pc/cm^3). (Default: 0)

            Output:
                spectra_obj: Spectrum object.
        """
        self.numchans, self.numspectra = data.shape
        assert len(freqs)==self.numchans

        self.freqs = freqs
        self.data = data.astype('float')
        self.dt = dt
        self.starttime = starttime
        self.dm = 0

    def __str__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def get_chan(self, channum):
        return self.data[channum,:]

    def get_spectrum(self, specnum):
        return self.data[:,specnum]

    def shift_channels(self, bins, padval=0):
        """Shift each channel to the left by the corresponding 
            value in bins, an array.

            Inputs:
                bins: An array containing the number of bins
                    to shift each channel by.
                padval: Value to use when shifting near the edge
                    of a channel. This can be a numeric value,
                    'median', 'mean', or 'rotate'. 
                    
                    The values 'median' and 'mean' refer to the 
                    median and mean of the channel. The value 
                    'rotate' takes values from one end of the 
                    channel and shifts them to the other.

            Outputs:
                None

            *** Shifting happens in-place ***
        """
        assert self.numchans == len(bins)
        for ii in range(self.numchans):
            chan = self.get_chan(ii)
            # Use 'chan[:]' so update happens in-place
            # this way the change effects self.data
            chan[:] = psr_utils.rotate(chan, bins[ii])
            if padval!='rotate':
                # Get padding value
                if padval=='mean':
                    pad = np.mean(chan)
                elif padval=='median':
                    pad = np.median(chan)
                else:
                    pad = padval
                
                # Replace rotated values with padval
                if bins[ii]>0:
                    chan[-bins[ii]:] = pad
                elif bins[ii]<0:
                    chan[:-bins[ii]] = pad

    def subband(self, nsub, subdm=None, padval=0):
        """Reduce the number of channels to 'nsub' by subbanding.
            The channels within a subband are combined using the
            DM 'subdm'. 'padval' is passed to the call to
            'Spectra.shift_channels'.

            Inputs:
                nsub: Number of subbands. Must be a factor of 
                    the number of channels.
                subdm: The DM with which to combine channels within
                    each subband (Default: don't shift channels 
                    within each subband)
                padval: The padding value to use when shifting
                    channels during dedispersion. See documentation
                    of Spectra.shift_channels. (Default: 0)

            Outputs:
                None

            *** Subbanding happens in-place ***
        """
        assert (self.numchans % nsub) == 0
        assert (subdm is None) or (subdm >= 0)
        nchan_per_sub = self.numchans // nsub
        sub_hifreqs = self.freqs[np.arange(nsub)*nchan_per_sub]
        sub_lofreqs = self.freqs[(1+np.arange(nsub))*nchan_per_sub-1]
        sub_ctrfreqs = 0.5*(sub_hifreqs+sub_lofreqs)
        
        if subdm is not None:
            # Compute delays
            ref_delays = psr_utils.delay_from_DM(subdm-self.dm, sub_ctrfreqs)
            delays = psr_utils.delay_from_DM(subdm-self.dm, self.freqs)
            rel_delays = delays-ref_delays.repeat(nchan_per_sub) # Relative delay
            rel_bindelays = np.round(rel_delays/self.dt).astype('int')
            # Shift channels
            self.shift_channels(rel_bindelays, padval)

        # Subband
        self.data = np.array([np.sum(sub, axis=0) for sub in \
                                np.vsplit(self.data, nsub)])
        self.freqs = sub_ctrfreqs
        self.numchans = nsub

    def scaled_median_std(self, indep=False):
        """Return a scaled version of the Spectra object.
            When scaling subtract the median from each channel,
            and divide by global std deviation (if indep==False), or
            divide by std deviation of each row (if indep==True).

            Input:
                indep: Boolean. If True, scale each row
                    independantly (Default: False).

            Output:
                scaled_spectra: A scaled version of the
                    Spectra object.
        """
        other = copy.deepcopy(self)
        if not indep:
            std = other.data.std()
        for ii in range(other.numchans):
            chan = other.get_chan(ii)
            median = np.median(chan)
            if indep:
                std = chan.std()
            chan[:] = (chan-median)/std
        return other
    
    def scaled_min_max(self, indep=False):
        """Return a scaled version of the Spectra object.
            When scaling subtract the min from each channel,
            and divide by global max (if indep==False), or
            divide by max of each row (if indep==True).

            Input:
                indep: Boolean. If True, scale each row
                    independantly (Default: False).

            Output:
                scaled_spectra: A scaled version of the
                    Spectra object.
        """
        other = copy.deepcopy(self)
        if not indep:
            max = other.data.max()
        for ii in range(other.numchans):
            chan = other.get_chan(ii)
            min = chan.min()
            if indep:
                max = chan.max()
            chan[:] = (chan-min)/max
        return other
    
    def mask_baseband_and_scale(self, basebandStd=1.0, indep=False):
        """
        Mask the baseband of the Spectra object and scale the remaining channels.
        
        This function combines mask_baseband and scaled_mean_std operations:
        1. Identifies channels to be masked based on their sum exceeding threshold
        2. Scales unmasked channels by subtracting mean and dividing by standard deviation
        3. Ensures masked channels remain zero
        
        Input:
            basebandStd: Float. The number of standard deviations to use for the mask (Default: 1.0).
            indep: Boolean. If True, scale each row independently (Default: False).
            
        Output:
            self: Returns the modified Spectra object (operation happens in-place).
        """
        # Step 1: Identify channels to mask (from mask_baseband)
        baseline_dataraw = np.sum(self.data, axis=1)
        data_bandpass = baseline_dataraw
        std_band, mean_band = np.std(data_bandpass), np.mean(data_bandpass)
        
        upper = basebandStd * std_band + mean_band
        lower = -basebandStd * std_band + mean_band
        
        # Create mask array (True for channels to keep, False for channels to mask)
        preserve_array = (data_bandpass <= upper) & (data_bandpass >= lower)
        preserve_array = preserve_array.flatten()  # Flatten to 1D
        
        # Step 2: Scale data (from scaled_mean_std)
        if not indep:
            # Calculate global standard deviation using only unmasked channels
            valid_data = self.data[preserve_array, :]
            if valid_data.size > 0:  # Check if we have any valid data
                global_std = valid_data.std()
            else:
                global_std = 1.0  # Fallback if all channels are masked
        
        # Process each channel
        for ii in range(self.numchans):
            chan = self.get_chan(ii)
            
            if preserve_array[ii]:  # Only scale channels we're keeping
                mean = np.mean(chan)
                
                if indep:
                    std = chan.std()
                else:
                    std = global_std
                    
                # Scale by subtracting mean and dividing by std
                chan[:] = (chan - mean) / std
            else:
                # Mask channels that didn't pass the threshold test
                chan[:] = 0
        
        return self
    
    # modify by zhang
    def mask_baseband(self, basebandStd=1.0):
        """Mask the baseband of the Spectra object.

        Input:
            axis: Integer. The axis along which to sum the data (Default: 0).
            basebandStd: Float. The number of standard deviations to use for the mask (Default: 1.0).

        Output:
            output: A masked version of the Spectra object.
        """
      
        baseline_dataraw = np.sum(self.data, axis=1)
        # print("self.data.shape:",self.data.shape)
        # print(len(baseline_dataraw))
        # print(baseline_dataraw)
        # data_bandpass = baseline_dataraw / np.max(baseline_dataraw)
        
        data_bandpass = baseline_dataraw / np.max(baseline_dataraw)
        # print("data_bandpass：",data_bandpass[1])
        std, mean = np.std(data_bandpass), np.mean(data_bandpass)

        upper = basebandStd * std + mean
        lower = -basebandStd * std + mean

        preserve_array = (data_bandpass < upper) & (data_bandpass > lower)
        preserve_array = preserve_array.flatten()  # Flatten the array to 1D
       
        self.Maskchannel = np.argwhere((data_bandpass >= upper) | (data_bandpass <= lower)).flatten()
        self.RFIzap = len(self.Maskchannel) / float(len(data_bandpass)) * 100.

        # for ii in range(self.numchans):
        #     chan = self.get_chan(ii)
        #     if not preserve_array[ii]:  # If the ii-th index is not in the preserve_array
        #         chan[:]=0

        return self
    
    # modify by zhang
    def scaled_mean_std(self, indep=False):
        """Return a scaled version of the Spectra object.
            When scaling subtract the mean from each channel,
            and divide by global std deviation (if indep==False), or
            divide by std deviation of each row (if indep==True).

            Input:
                indep: Boolean. If True, scale each row
                    independantly (Default: False).

            Output:
                scaled_spectra: A scaled version of the
                    Spectra object.
        """
        other = copy.deepcopy(self)
        # means = np.mean(other.data, axis=1).astype(np.uint64)
        # stds=np.std(other.data, axis=1)
        if not indep:
            std = other.data.std()
        for ii in range(other.numchans):
            chan = other.get_chan(ii)
            mean = np.mean(chan)
            if indep:
                std = chan.std()
            # chan[:] = (chan-means[ii])/stds[ii]
            # chan[:] = (chan-mean)/std
            chan[:] = (chan-mean)
        return other
    
     # modify by zhang
    def scaled_chan_mean(self):
        """Return a scaled version of the Spectra object.
            When scaling subtract the mean from each channel,
            and divide by global std deviation (if indep==False), or
            divide by std deviation of each row (if indep==True).

            Input:
                indep: Boolean. If True, scale each row
                    independantly (Default: False).

            Output:
                scaled_spectra: A scaled version of the
                    Spectra object.
        """
        other = copy.deepcopy(self)
        means = np.mean(other.data, axis=1)  # shape: (numspecs,)
        for ii in range(other.numchans):
            chan = other.get_chan(ii)
            # mean = np.mean(chan)
            chan[:] = (chan-means[ii])
        return other
    
    def scaled_spectrum_mean(self):
        """Return a scaled version of the Spectra object.
        When scaling subtract the mean from each spectrum (time sample),
        i.e., subtract the mean of each column.

        This performs: data - data.mean(axis=0) for each spectrum

        Output:
            scaled_spectra: A scaled version of the
                Spectra object with each spectrum's mean removed.
        """
        other = copy.deepcopy(self)
        means = np.mean(other.data, axis=0)  # shape: (numspecs,)
        for specnum in range(other.numspectra):
            spec = other.get_spectrum(specnum)  
            spec[:] = spec - means[specnum]     
        
        return other

    def masked(self, mask, maskval='median-mid80'):
        """Replace masked data with 'maskval'. Returns
            a masked copy of the Spectra object.
            
            Inputs:
                mask: An array of boolean values of the same size and shape
                    as self.data. True represents an entry to be masked.
                maskval: Value to use when masking. This can be a numeric
                    value, 'median', 'mean', or 'median-mid80'.

                    The values 'median' and 'mean' refer to the median and
                    mean of the channel, respectively. The value 'median-mid80'
                    refers to the median of the channel after the top and bottom
                    10% of the sorted channel is removed.
                    (Default: 'median-mid80')

            Output:
                maskedspec: A masked version of the Spectra object.
        """
        assert self.data.shape == mask.shape
        maskvals = np.ones(self.numchans)
        for ii in range(self.numchans):
            chan = self.get_chan(ii)
            # Use 'chan[:]' so update happens in-place
            if maskval=='mean':
                maskvals[ii]=np.mean(chan)
            elif maskval=='median':
                maskvals[ii]=np.median(chan)
            elif maskval=='median-mid80':
                n = int(np.round(0.1*self.numspectra))
                maskvals[ii]=np.median(sorted(chan)[n:-n])
            else:
                maskvals[ii]=maskval
            if np.all(mask[ii]):
                self.data[ii] = np.ones_like(self.data[ii])*(maskvals[:,np.newaxis][ii])
        return self
 
    def dedisperse(self, dm=0, padval=0):
        """Shift channels according to the delays predicted by
            the given DM.
            
            Inputs:
                dm: The DM (in pc/cm^3) to use.
                padval: The padding value to use when shifting
                    channels during dedispersion. See documentation
                    of Spectra.shift_channels. (Default: 0)

            Outputs:
                None

            *** Dedispersion happens in place ***
        """
        assert dm >= 0
        ref_delay = psr_utils.delay_from_DM(dm-self.dm, np.max(self.freqs))
        delays = psr_utils.delay_from_DM(dm-self.dm, self.freqs)
        rel_delays = delays-ref_delay # Relative delay
        rel_bindelays = np.round(rel_delays/self.dt).astype('int')
        self.delay_second = np.max(delays) - np.min(delays)
        # Shift channels
        self.shift_channels(rel_bindelays, padval)

        self.dm=dm

    def smooth(self, width=1, padval=0):
        """Smooth each channel by convolving with a top hat
            of given width. The height of the top had is
            chosen shuch that RMS=1 after smoothing. 
            Overlap values are determined by 'padval'.

            Inputs:
                width: Number of bins to smooth by (Default: no smoothing)
                padval: Padding value to use. Possible values are
                    float-value, 'mean', 'median', 'wrap'.
                    (Default: 0).

            Ouputs:
                None

            This bit of code is taken from Scott Ransom's
            PRESTO's single_pulse_search.py (line ~ 423).
            
            *** Smoothing is done in place. ***
        """
        if width > 1:
            kernel = np.ones(width, dtype='float32')/np.sqrt(width)
            for ii in range(self.numchans):
                chan = self.get_chan(ii)
                if padval=='wrap':
                    tosmooth = np.concatenate([chan[-width:], \
                                chan, chan[:width]])
                elif padval=='mean':
                    tosmooth = np.ones(self.numspectra+width*2) * \
                                np.mean(chan)
                    tosmooth[width:-width] = chan
                elif padval=='median':
                    tosmooth = np.ones(self.numspectra+width*2) * \
                                np.median(chan)
                    tosmooth[width:-width] = chan
                else: # padval is a float
                    tosmooth = np.ones(self.numspectra+width*2) * \
                                padval
                    tosmooth[width:-width] = chan
                    
                smoothed = scipy.signal.convolve(tosmooth, kernel, 'same')
                chan[:] = smoothed[width:-width]
                    
    def trim(self, bins=0):
        """Trim the end of the data by 'bins' spectra.
            
            Input:
                bins: Number of spectra to trim off the end of the observation.
                    If bins is negative trim spectra off the beginning of the
                    observation.

            Outputs:
                None

            *** Trimming is irreversible ***
        """
        assert bins < self.numspectra
        if bins == 0:
            return
        elif bins > 0:
            self.data = self.data[:,:-bins]
            self.numspectra = self.numspectra-bins
        elif bins < 0:
            self.data = self.data[:,bins:]
            self.numspectra = self.numspectra-bins
            self.starttime = self.starttime+bins*self.dt

    def downsample(self, factor=1, trim=True):
        """Downsample (in-place) the spectra by co-adding
            'factor' adjacent bins.

            Inputs:
                factor: Reduce the number of spectra by this
                    factor. Must be a factor of the number of
                    spectra if 'trim' is False.
                trim: Trim off excess bins.

            Ouputs:
                None

            *** Downsampling is done in place ***
        """
        assert trim or not (self.numspectra % factor)
        new_num_spectra = self.numspectra // factor
        num_to_trim = self.numspectra%factor
        self.trim(num_to_trim)
        self.data = np.array(np.column_stack([np.sum(subint, axis=1) for \
                        subint in np.hsplit(self.data,new_num_spectra)]))
        self.numspectra = new_num_spectra
        self.dt = self.dt*factor
        
    
    def exp_normalization_plus(self):
        """
        Normalize the data using exponential normalization.

        Inputs:
            None

        Outputs:
            data: The normalized data.
        """
        self.data = np.exp(self.data / np.max(self.data) + 1)
        return self.data
    
    def exp_normalization_subtract(self):
        """
        Normalize the data using exponential normalization.

        Inputs:
            None

        Outputs:
            data: The normalized data.
        """
        self.data = np.exp(self.data / np.max(self.data) - 1)
        return self.data
    
    def correct_data(self, scaling=0.8, lower_size=1.5, upper_size=3.0):
        """
        Apply dynamic correction to the data.

        Inputs:
            scaling: The scaling factor for the standard deviation.
            lower_size: The lower size for the standard deviation.
            upper_size: The upper size for the standard deviation.

        Outputs:
            output: The corrected data.
        """
        mean= np.mean(self.data)
        std = np.std(self.data)
        plotVmin = mean - lower_size * std * scaling
        plotVmax = mean + upper_size * std * scaling

        self.data = np.clip(self.data, a_min=plotVmin, a_max=plotVmax)
        self.data = (self.data - plotVmin) / (plotVmax - plotVmin)
        
    def plot_data(self, title, ax):
        img = ax.imshow(self.data, aspect='auto', cmap=plt.get_cmap("gist_yarg"),
                        interpolation='nearest', origin='upper',
                        extent=(self.starttime, self.starttime + len(self.data[0])*self.dt,
                                self.freqs.min(), self.freqs.max()))
        ax.set_title(title)
        fig = ax.figure  
        fig.colorbar(img, ax=ax) 
        return img
    
    def plot_data_single(self, title, plot_dir, fn, start, dm, downsamp, fd):
        """
        Plot the data with labels and save as a png file.

        Parameters:
        title (str): The title of the plot.
        plot_dir (str): The directory to save the plot.
        fn (str): The filename of the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(self.data, aspect='auto', cmap='hot',
                        interpolation='nearest', origin='upper',
                        extent=(self.starttime, self.starttime + len(self.data[0])*self.dt,
                                self.freqs.min(), self.freqs.max()))
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{fn}_{title}_TOA{start:.3f}_DM{dm}_Td{downsamp}_Fd{fd}.png')
        plt.close()
    
    def create_dmvstm_array(self, dm, nbinlim):
        dmvstm_array = []
        dm_2 = 150
        lodm = max(dm - dm_2, 0)
        hidm = dm + dm_2
        dmstep = (hidm - lodm) // 50

        for ii in np.arange(lodm, hidm, dmstep):
            self.dedisperse(0, padval='rotate')
            self.dedisperse(ii, padval='rotate')
            Data = np.array(self.data[..., :nbinlim])
            Dedisp_ts = Data.sum(axis=0)
            dmvstm_array.append(Dedisp_ts)
        
        return np.array(dmvstm_array)
    
    def plot_dmvstm(self, ax, dmvstm_array, nbinlim, dm):
        dm_2 = 150
        lodm = max(dm - dm_2, 0)
        hidm = dm + dm_2
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("DM")
        extent = (self.starttime, self.starttime+nbinlim*self.dt, lodm, hidm)
        ax.imshow(dmvstm_array, cmap=plt.get_cmap("gist_yarg"), aspect='auto', origin='lower', extent=extent)
        
        ax.set_ylim(lodm, hidm)

    def plot_freqtime(self, ax, nbinlim):
        ax.imshow(self.data[..., :nbinlim], aspect='auto',
                        cmap=plt.get_cmap("gist_yarg"),
                        interpolation='nearest', origin='upper',
                        extent=(self.starttime, self.starttime + nbinlim * self.dt,
                                self.freqs.min(), self.freqs.max()))
        
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.set_ylabel("Frequency (MHz)")
        
    def plot_timeseries(self, ax, duration, nbinlim, start, dm, fn):
        spectrum_window = 0.02*duration
        window_width = int(spectrum_window/self.dt)
        burst_bin = nbinlim//2
        Data = np.array(self.data[..., :nbinlim])
        Dedisp_ts = Data.sum(axis=0)
        times = (np.arange(self.numspectra) * self.dt + start)[..., :nbinlim]
        
        ax.plot(times, Dedisp_ts, "k")
        ax.set_xlim([times.min(), times.max()])
        
        text1 = f"DM: {self.dm:.2f}"
        ax.text(1.1, 0.9, text1, fontsize=15, ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(fn, fontsize=14)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axvline(times[burst_bin]-spectrum_window,ls="--",c="grey")
        ax.axvline(times[burst_bin]+spectrum_window,ls="--",c="grey")
    
    def plot_spectrum_dmsnr(self, ax_spec, ax_dmsnr, ax_ts, dmvstm_array, burst_bin, window_width, nbinlim):
        on_spec = np.array(self.data[..., burst_bin-window_width:burst_bin+window_width])
        on_dmsnr = np.array(dmvstm_array[..., burst_bin-window_width:burst_bin+window_width])
        Dedisp_spec = np.mean(on_spec, axis=1)
        Dedisp_dmsnr = np.mean(on_dmsnr, axis=1)

        off_spec1 = np.array(self.data[..., 0:burst_bin-window_width])
        off_spec2 = np.array(self.data[..., burst_bin+window_width:nbinlim])
        off_spec = (np.mean(off_spec1, axis=1) + np.mean(off_spec2, axis=1)) / 2.0
        
        off_dmsnr1 = np.array(dmvstm_array[..., 0:burst_bin-window_width])
        off_dmsnr2 = np.array(dmvstm_array[..., burst_bin+window_width:nbinlim])
        off_dmsnr = (np.mean(off_dmsnr1, axis=1) + np.mean(off_dmsnr2, axis=1)) / 2.0

        dm_2 = 150
        lodm = max(self.dm - dm_2, 0)
        hidm = self.dm + dm_2
        dms = np.linspace(lodm, hidm, len(Dedisp_dmsnr))
        freqs = np.linspace(self.freqs.max(), self.freqs.min(), len(Dedisp_spec))

        ax_spec.plot(Dedisp_spec, freqs, color="grey",ls="--", lw=2)
        ax_spec.plot(off_spec, freqs, color="grey", ls="-",alpha=0.5, lw=1)
        
        ttest = float(stats.ttest_ind(Dedisp_spec, off_spec)[0])
        ttestprob = float(stats.ttest_ind(Dedisp_spec, off_spec)[1])
        ax_spec.text(1.1, 0.45, "t-test", fontsize=12, ha='center', va='center', transform=ax_ts.transAxes)
        ax_spec.text(1.1, 0.3, f"{ttest:.2f}({(1-ttestprob)*100:.2f}%)", fontsize=12, ha='center', va='center', transform=ax_ts.transAxes)

        ax_dmsnr.plot(Dedisp_dmsnr, dms, color="grey", ls="--",lw=2)
        ax_dmsnr.plot(off_dmsnr, dms, color="grey", ls="-",alpha=0.5, lw=1)

        plt.setp(ax_spec.get_xticklabels(), visible=True)
        plt.setp(ax_dmsnr.get_xticklabels(), visible=False)
        plt.setp(ax_spec.get_yticklabels(), visible=False)
        plt.setp(ax_dmsnr.get_yticklabels(), visible=False)
        ax_spec.set_ylim([self.freqs.min(), self.freqs.max()])
        ax_dmsnr.set_ylim(lodm, hidm)

    def plot_original_data(self, ax):
        self.dedisperse(0, padval='rotate')
        ax.set_ylabel("Frequency (MHz)")
        ax.set_xlabel("Time (sec)")
        # #For FRB20220610A
        # data=np.transpose(self.data)
        # img_orig = ax.imshow(data, aspect='auto', cmap=plt.get_cmap("plasma"),
        #                      interpolation='nearest', origin='upper',
        #                      extent=(self.starttime, self.starttime + len(self.data[0])*self.dt,
        #                              self.freqs.min(), self.freqs.max()))
        img_orig = ax.imshow(self.data, aspect='auto', cmap=plt.get_cmap("plasma"),
                             interpolation='nearest', origin='upper',
                             extent=(self.starttime, self.starttime + len(self.data[0])*self.dt,
                                     self.freqs.min(), self.freqs.max()))
        cb = ax.get_figure().colorbar(img_orig)
        cb.set_label("Scaled signal intensity (arbitrary units)")
