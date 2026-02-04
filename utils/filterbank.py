"""
Modified from PRESTO's filterbank.py module
Original: https://github.com/scottransom/presto
Original Authors: Scott Ransom and contributors
  - Original version (June 26, 2012)
Modified by: Bin Zhang (June 23, 2024)

A module for reading filterbank files.
This version adds support for 1-bit, 2-bit, and 4-bit data formats.

PRESTO is licensed under the GNU General Public License v2.0.
See LICENSE file for details.
"""
from builtins import object

import sys
import os
import os.path
import numpy as np
from presto import sigproc
from .spectra import Spectra


DEBUG = False

def create_filterbank_file(outfn, header, spectra=None, nbits=8, \
                           verbose=False, mode='append'):
    """Write filterbank header and spectra to file.

        Input:
            outfn: The outfile filterbank file's name.
            header: A dictionary of header paramters and values.
            spectra: Spectra to write to file. (Default: don't write
                any spectra - i.e. write out header only)
            nbits: The number of bits per sample of the filterbank file.
                This value always overrides the value in the header dictionary.
                (Default: 8 - i.e. each sample is an 8-bit integer)
            verbose: If True, be verbose (Default: be quiet)
            mode: Mode for writing (can be 'append' or 'write')

        Output:
            fbfile: The resulting FilterbankFile object opened
                in read-write mode.
    """
    dtype = get_dtype(nbits) # Get dtype. This will check to ensure
                             # 'nbits' is valid.
    header['nbits'] = nbits
    outfile = open(outfn, 'wb')
    outfile.write(sigproc.addto_hdr("HEADER_START", None))
    for paramname in list(header.keys()):
        if paramname not in sigproc.header_params:
            # Only add recognized parameters
            continue
        if verbose:
            print("Writing header param (%s)" % paramname)
        value = header[paramname]
        outfile.write(sigproc.addto_hdr(paramname, value))
    outfile.write(sigproc.addto_hdr("HEADER_END", None))
    if spectra is not None:
        spectra.flatten().astype(dtype).tofile(outfile)
    outfile.close()
    return FilterbankFile(outfn, mode=mode)


def is_float(nbits):
    """For a given number of bits per sample return
        true if it corresponds to floating-point samples
        in filterbank files.

        Input:
            nbits: Number of bits per sample, as recorded in the filterbank
                file's header.

        Output:
            isfloat: True, if 'nbits' indicates the data in the file
                are encoded as floats.
    """
    check_nbits(nbits)
    if nbits == 32:
        return True
    else:
        return False


def check_nbits(nbits):
    """Given a number of bits per sample check to make
        sure 'filterbank.py' can cope with it.

        An exception is raise if 'filterbank.py' cannot cope.

        Input:
            nbits: Number of bits per sample, as recorded in the filterbank
                file's header.

        Output:
            None
    """
    if nbits not in [32, 16, 8, 4, 2, 1]:  # 添加 4
        raise ValueError("'filterbank.py' only supports " \
                                    "files with 1-, 2-, 4-, 8- or 16-bit " \
                                    "integers, or 32-bit floats " \
                                    "(nbits provided: %g)!" % nbits)


def get_dtype(nbits):
    """For a given number of bits per sample return
        a numpy-recognized dtype.

        Input:
            nbits: Number of bits per sample, as recorded in the filterbank
                file's header.

        Output:
            dtype: A numpy-recognized dtype string.
    """
    check_nbits(nbits)
    if is_float(nbits):
        dtype = 'float%d' % nbits
    elif nbits in [1, 2, 4]:  # 添加 4
        dtype = 'uint8'  # 1-bit, 2-bit and 4-bit data will be packed into 8-bit integers.
    else:
        dtype = 'uint%d' % nbits
    return dtype


def read_header(filename, verbose=False):
    """Read the header of a filterbank file, and return
        a dictionary of header paramters and the header's
        size in bytes.

        Inputs:
            filename: Name of the filterbank file.
            verbose: If True, be verbose. (Default: be quiet)

        Outputs:
            header: A dictionary of header paramters.
            header_size: The size of the header in bytes.
    """
    header = {}
    filfile = open(filename, 'rb')
    filfile.seek(0)
    paramname = ""
    while (paramname != 'HEADER_END'):
        if verbose:
            print("File location: %d" % filfile.tell())
        paramname, val = sigproc.read_hdr_val(filfile, stdout=verbose)
        if verbose:
            print("Read param %s (value: %s)" % (paramname, val))
        if paramname not in ["HEADER_START", "HEADER_END"]:
            header[paramname] = val
    header_size = filfile.tell()
    filfile.close()
    return header, header_size


class FilterbankFile(object):
    def __init__(self, filfn, mode='readonly'):
        # self.filename =  os.path.basename(filfn)
        self.filename =  filfn
        #modify zhang
        self.fn, self.ext = os.path.splitext(os.path.basename(filfn))
        self.filfile = None
        if not os.path.isfile(filfn):
            raise ValueError("ERROR: File does not exist!\n\t(%s)" % filfn)
        self.header, self.header_size = read_header(self.filename)
        self.frequencies = self.fch1 + self.foff*np.arange(self.nchans)
        self.is_hifreq_first = (self.foff < 0)
        self.bytes_per_spectrum = self.nchans*self.nbits // 8
        data_size = os.path.getsize(self.filename)-self.header_size
        self.nspec = data_size // self.bytes_per_spectrum
        self.nsamp_per_subint = 1024  # modify zhang fix the per subint nsamp
        self.nsubints = self.nspec // self.nsamp_per_subint
        self.df = abs(self.foff)
        # print("load fil sucess!!!!")
       
        # Check if this file is a folded-filterbank file
        if 'npuls' in self.header and 'period' in self.header and \
                'nbins' in self.header and 'tsamp' not in self.header:
            # Foleded file
            self.isfold = True
            self.dt = self.period/self.nbins
        else:
            self.isfold = False
            self.dt = self.tsamp
        # telescope_id = self.header.get('telescope_id', 0)
        # print(f"Telescope ID found: {telescope_id}")
        # self.telescope = get_telescope_name(telescope_id)
        # print(f"Telescope name: {self.telescope}")

        telescope_id = self.header.get('telescope_id', 0)
        print(f"Telescope ID found: {telescope_id}")
        telescope_names = {0: "Fake", 1: "Arecibo", 4: "Parkes", 6: "GBT", 
                          20: "FAST", 64: "MeerKAT", 7: "SKA"}
        self.telescope = telescope_names.get(telescope_id, f"Unknown_ID_{telescope_id}")
        
        if 'src_raj' in self.header:
            ra_val = self.header['src_raj']
            ra_hr = int(ra_val // 10000)
            ra_min = int((ra_val - ra_hr*10000) // 100)
            ra_sec = ra_val - ra_hr*10000 - ra_min*100
            self.ra_str = f"{ra_hr:02d}:{ra_min:02d}:{ra_sec:07.4f}"
        else:
            self.ra_str = "00:00:00.0000"
        
        
        if 'src_dej' in self.header:
            dec_val = self.header['src_dej']
            sign = '-' if dec_val < 0 else '+'
            dec_val = abs(dec_val)
            dec_deg = int(dec_val // 10000)
            dec_min = int((dec_val - dec_deg*10000) // 100)
            dec_sec = dec_val - dec_deg*10000 - dec_min*100
            self.dec_str = f"{sign}{dec_deg:02d}:{dec_min:02d}:{dec_sec:07.4f}"
        else:
            self.dec_str = "+00:00:00.0000"
        
     
        if 'tstart' in self.header:
            self.mjd = self.header['tstart']

        # Get info about dtype
        self.dtype = get_dtype(self.nbits)
        if is_float(self.nbits):
            tinfo = np.finfo(self.dtype)
        else:
            tinfo = np.iinfo(self.dtype)
        self.dtype_min = tinfo.min
        self.dtype_max = tinfo.max

        if mode.lower() in ('read', 'readonly'):
            self.filfile = open(self.filename, 'rb')
        elif mode.lower() in ('write', 'readwrite'):
            self.filfile = open(self.filename, 'r+b')
        elif mode.lower() == 'append':
            self.filfile = open(self.filename, 'a+b')
        else:
            raise ValueError("Unrecognized mode (%s)!" % mode)
        
    # modify by zhang
    def unpack_1bit(self, data):
        """Unpack 1-bit data that has been read in as bytes.

            Input:
                data: array of bits packed into an array of bytes.

            Output:
                outdata: unpacked array. The size of this array will
                    be eight times the size of the input data.
        """
        b0 = np.bitwise_and(data >> 0x07, 0x01)
        b1 = np.bitwise_and(data >> 0x06, 0x01)
        b2 = np.bitwise_and(data >> 0x05, 0x01)
        b3 = np.bitwise_and(data >> 0x04, 0x01)
        b4 = np.bitwise_and(data >> 0x03, 0x01)
        b5 = np.bitwise_and(data >> 0x02, 0x01)
        b6 = np.bitwise_and(data >> 0x01, 0x01)
        b7 = np.bitwise_and(data, 0x01)
        return np.dstack([b0, b1, b2, b3, b4, b5, b6, b7]).flatten()

    # modify by zhang
    def unpack_2bit(self, data):
        """Unpack 2-bit data that has been read in as bytes.

            Input: 
                data: array of unsigned 2-bit ints packed into
                    an array of bytes.

            Output: 
                outdata: unpacked array. The size of this array will 
                    be four times the size of the input data.
        """
        piece0 = np.bitwise_and(data >> 0x06, 0x03)
        piece1 = np.bitwise_and(data >> 0x04, 0x03)
        piece2 = np.bitwise_and(data >> 0x02, 0x03)
        piece3 = np.bitwise_and(data, 0x03)
        return np.dstack([piece0, piece1, piece2, piece3]).flatten()

    def unpack_4bit(self, data):
        """Unpack 4-bit data that has been read in as bytes.

            Input:
                data: array of 4-bit values packed into an array of bytes.

            Output:
                outdata: unpacked array. The size of this array will
                    be two times the size of the input data.
        """
        # 高 4 位
        piece0 = np.bitwise_and(data >> 0x04, 0x0F)
        # 低 4 位
        piece1 = np.bitwise_and(data, 0x0F)
        return np.dstack([piece0, piece1]).flatten()

    @property
    def freqs(self):
        # Alias for frequencies
        return self.frequencies

    @property
    def nchan(self):
        # more aliases..
        return self.nchans

    def close(self):
        if self.filfile is not None:
            self.filfile.close()

    def get_timeslice(self, start, stop):
        startspec = int(np.round(start/self.tsamp))
        stopspec = int(np.round(stop/self.tsamp))
        return self.get_spectra(startspec, stopspec-startspec)

    def get_spectra(self, start, nspec):
        stop = min(start+nspec, self.nspec)
        pos = self.header_size+start*self.bytes_per_spectrum
        # Compute number of elements to read
        nspec = int(stop) - int(start)
        
        if self.nbits == 2:
            num_to_read = (nspec*self.nchans+3)//4 
        elif self.nbits == 1:
            num_to_read = (nspec*self.nchans+7)//8
        elif self.nbits == 4:  # 添加 4bit 支持
            num_to_read = (nspec*self.nchans+1)//2
        else:
            num_to_read = nspec*self.nchans
            
        num_to_read = max(0, num_to_read)
        self.filfile.seek(pos, os.SEEK_SET)
        spectra_dat = np.fromfile(self.filfile, dtype=self.dtype, 
                              count=num_to_read)
        
        if self.nbits == 2:
            spectra_dat = self.unpack_2bit(spectra_dat) 
        elif self.nbits == 1:
            spectra_dat = self.unpack_1bit(spectra_dat)
        elif self.nbits == 4:  # 添加 4bit 支持
            spectra_dat = self.unpack_4bit(spectra_dat)
            
        #Debugging information
        print(f"Read {len(spectra_dat)} elements from file.")
        print(f"Expected shape after unpacking: ({nspec}, {self.nchans})")
        spectra_dat.shape = nspec, self.nchans
   
        if not self.is_hifreq_first:
            spectra_dat = spectra_dat[:, ::-1]
            freqs = self.frequencies[::-1]
        else:
            freqs = self.freqs 
        spec = Spectra(freqs, self.tsamp, spectra_dat.T,
                starttime=start*self.tsamp, dm=0.0)
        return spec
    
    # modify by zhang 
    def get_spectra_slide(self, start_time, start_samp, end_samp):
        """Return data from filterbank file for the specified sample range.
    
        Args:
            start_time: Starting time in seconds
            start_samp: Starting sample index
            end_samp: Ending sample index
            
        Returns:
            Spectra object containing the requested data
        """
        if not self.ext.endswith('.fil'):
            raise ValueError("This method is for filterbank files only")
        
        # Calculate number of samples to read
        nspec = end_samp - start_samp
        
        # Get the spectra data using the existing method
        spec = self.get_spectra(start_samp, nspec)
        
        # Update the start time if needed
        if start_time != spec.starttime:
            spec.starttime = start_time
        
        return spec
    
    # modify by zhang 
    def get_spectra_all(self):
        """Return all spectra from filterbank file.
 
        Output:
            spec: Spectra object containing all data from the file
        """
        return self.get_spectra(0, self.nspec)
    
    # modify by zhang
    def get_spectra_subint(self, subints, samples_per_subint):
        """Return spectra from specified subints in filterbank file.
    
        Inputs:
            subints: list or array of subint indices to read
            samples_per_subint: number of time samples in each subint
    
        Output:
            spec: Spectra object containing data from specified subints
        """
        # Validate subint indices
        subints = np.array(subints)
        max_subint = self.nspec // samples_per_subint
        if np.any(subints < 0) or np.any(subints >= max_subint):
            raise ValueError(f"Subint indices must be between 0 and {max_subint-1}")
        
        # Calculate start sample and number of samples to read
        start = subints[0] * samples_per_subint
        nspec = len(subints) * samples_per_subint
        
        # Check if subints are continuous
        if np.all(np.diff(subints) == 1):
            # If subints are continuous, read all at once
            return self.get_spectra(start, nspec)
        else:
            # If subints are not continuous, read each subint separately and combine
            specs = []
            for isub in subints:
                start = isub * samples_per_subint
                spec = self.get_spectra(start, samples_per_subint)
                specs.append(spec.data)
            
            # Combine all spectra
            combined_data = np.hstack(specs)
            
            # Create new Spectra object
            spec = Spectra(self.freqs, self.tsamp, combined_data,
                                starttime=subints[0]*samples_per_subint*self.tsamp,
                                dm=0.0)
            return spec
    
    # modify by zhang
    def pack_1bit(self, data):
        """Pack 1-bit data.
        
            Input:
                data: unpacked array of 1-bit values
                
            Output:
                packed_data: packed array where each byte contains eight 1-bit values
        """
        if len(data) % 8 != 0:
            raise ValueError("Data length must be a multiple of 8 to pack into 1-bit format")
        data = data.reshape(-1, 8)
        packed_data = np.zeros(data.shape[0], dtype=np.uint8)
        for i in range(8):
            packed_data |= (data[:, i] << (7-i))
        return packed_data
    
    # modify by zhang
    def pack_2bit(self, data):
        """Pack 2-bit data.
        
            Input:
                data: unpacked array of 2-bit values
                
            Output:
                packed_data: packed array where each byte contains four 2-bit values
        """
        if len(data) % 4 != 0:
            raise ValueError("Data length must be a multiple of 4 to pack into 2-bit format")
        data = data.reshape(-1, 4)
        packed_data = (data[:, 0] << 6) | (data[:, 1] << 4) | (data[:, 2] << 2) | data[:, 3]
        return packed_data.astype(np.uint8)

    def pack_4bit(self, data):
        """Pack 4-bit data.
        
            Input:
                data: unpacked array of 4-bit values
                
            Output:
                packed_data: packed array where each byte contains two 4-bit values
        """
        if len(data) % 2 != 0:
            raise ValueError("Data length must be a multiple of 2 to pack into 4-bit format")
        data = data.reshape(-1, 2)
        packed_data = (data[:, 0] << 4) | data[:, 1]
        return packed_data.astype(np.uint8)

    def append_spectra(self, spectra):
        """Append spectra to the file if is not read-only.
            
            Input:
                spectra: The spectra to append. The new spectra
                    must have the correct number of channels (ie
                    dimension of axis=1.

            Outputs:
                None
        """
        if self.filfile.mode.lower() in ('r', 'rb'):
            raise ValueError("FilterbankFile object for '%s' is read-only." % \
                        self.filename)
        nspec, nchans = spectra.shape
        if nchans != self.nchans:
            raise ValueError("Cannot append spectra. Incorrect shape. " \
                        "Number of channels in file: %d; Number of " \
                        "channels in spectra to append: %d" % \
                        (self.nchans, nchans))
        data = spectra.flatten()
        np.clip(data, self.dtype_min, self.dtype_max, out=data)
        # Move to end of file
        self.filfile.seek(0, os.SEEK_END)
        # modify by zhang
        if self.nbits == 2:
            data = self.pack_2bit(data) 
        elif self.nbits == 1:
            data = self.pack_1bit(data)
        elif self.nbits == 4:  # 添加 4bit 支持
            data = self.pack_4bit(data)
            
        self.filfile.write(data.astype(self.dtype))
        # modify by zhang
        self.nspec += nspec
        #self.filfile.flush()
        #os.fsync(self.filfile)

    def write_spectra(self, spectra, ispec):
        """Write spectra to the file if is writable.
            
            Input:
                spectra: The spectra to append. The new spectra
                    must have the correct number of channels (ie
                    dimension of axis=1.
                ispec: The index of the spectrum of where to start writing.

            Outputs:
                None
        """
        if 'r+' not in self.filfile.mode.lower():
            raise ValueError("FilterbankFile object for '%s' is not writable." % \
                        self.filename)
        nspec, nchans = spectra.shape
        if nchans != self.nchans:
            raise ValueError("Cannot write spectra. Incorrect shape. " \
                        "Number of channels in file: %d; Number of " \
                        "channels in spectra to write: %d" % \
                        (self.nchans, nchans))
        if ispec > self.nspec:
            raise ValueError("Cannot write past end of file! " \
                             "Present number of spectra: %d; " \
                             "Requested index of write: %d" % \
                             (self.nspec, ispec))
        data = spectra.flatten()
        np.clip(data, self.dtype_min, self.dtype_max, out=data)
        # modify by zhang
        if self.nbits == 2:
            data = self.pack_2bit(data) 
        elif self.nbits == 1:
            data = self.pack_1bit(data)
        elif self.nbits == 4:  # 添加 4bit 支持
            data = self.pack_4bit(data)
            
        # Move to requested position
        pos = self.header_size + ispec*self.bytes_per_spectrum
        self.filfile.seek(pos, os.SEEK_SET)
        self.filfile.write(data.astype(self.dtype))
        if nspec+ispec > self.nspec:
            self.nspec = nspec+ispec

    def __getattr__(self, name):
        if name in self.header:
            if DEBUG:
                print("Fetching header param (%s)" % name)
            val = self.header[name]
        else:
            raise ValueError("No FilterbankFile attribute called '%s'" % name)
        return val

    def print_header(self):
        """Print header parameters and values.
        """
        for param in sorted(self.header.keys()):
            if param in ("HEADER_START", "HEADER_END"):
                continue
            print("%s: %s" % (param, self.header[param]))


def main():
    fil = FilterbankFile(sys.argv[1])
    fil.print_header()


if __name__ == '__main__':
    main()