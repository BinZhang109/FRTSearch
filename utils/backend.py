from mmengine.fileio.file_client import FileClient
from mmengine.fileio.file_client import BaseStorageBackend
from .psrfits import PsrfitsFile
from .filterbank import FilterbankFile

@FileClient.register_backend("fits_and_fil")
class FitsBackend(BaseStorageBackend):

    def get(self, filepath):
        filepath = str(filepath)
        output = None
        if filepath.endswith(".fil"):
            # Filterbank file
            try:
                output = FilterbankFile(filepath)
            except Exception:
                output = None
        elif filepath.endswith(".fits"):
            # PSRFITS file
            try:
                output = PsrfitsFile(filepath)
            except Exception:
                output = None
        else:
            raise ValueError("Cannot recognize data file type from "
                             "extension. (Only '.fits' and '.fil' "
                             "are supported.)")
        return output

    def get_text(self, filepath):
        raise NotImplementedError