from .shift_kerchunk import kerchunk_shift_rfl
import s3fs
import fsspec
import ujson
from kerchunk.combine import MultiZarrToZarr
import os
import re
import datetime
import numpy as np
from shift_python_utilities.envi_kerchunk.utils import *


def make_shift_multi_kerchunk(input_directory, outfile):
    """
    Reformats a directory of shift data into a single zarr using kerchunk

    Parameters
    ----------
    input_directory : str
        Path to input directory
        
    outfile : str
        Write path
    
    Returns
    -------
        None (writes zarr)
    """
    
    assert os.path.splitext(output_file)[-1] == ".json", "Output file needs to be a json"
    
    if 's3' in input_directory:
        s3 = s3fs.S3FileSystem(anon=False)
        flist_all = s3.ls(input_directory)
        flist = sorted([f"s3://{f}" for f in flist_all if f.endswith("rfl_phase.hdr")])
    else:
        base_path, _ = os.path.split(input_directory)
        flist_all = os.listdir(input_directory)
        flist = sorted([os.path.join(base_path, f) for f in flist_all if f.endswith("rfl_phase.hdr")])

    json_list = [kerchunk_shift_rfl(of) for of in flist]


    new_dims = np.array([parse_date(f) for f in json_list])
    combined = MultiZarrToZarr(
        json_list,
        remote_protocol="s3",
        remote_options={'anon': True},
        coo_map={'time': new_dims},
        concat_dims=['time'],
        identical_dims=['x', 'y', 'wavelength']
    )

    combined_t = combined.translate()
    with fsspec.open(outfile, "w") as of:
        of.write(ujson.dumps(combined_t, indent=2))