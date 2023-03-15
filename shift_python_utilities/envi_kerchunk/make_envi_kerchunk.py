import numpy as np
import fsspec
import ujson
import base64
import os
from shift_python_utilities.envi_kerchunk.utils import *

def make_envi_kerchunk(rdn_path, loc_path, obs_path, output_file):
    """
    Reformats envi files as a zarr

    Parameters
    ----------
    rdn_path : str
        Path radiance data
    
    loc_path : str
        Path to location data
    
    obs_path : str
        Path to obs data
        
    output_file : str
        Write path
    
    Returns
    -------
        None (writes zarr)
    """

    assert rdn_path.endswith(".hdr"), f"Need path to radiance HDR file, not binary. Got {rdn_path}."
    assert loc_path.endswith(".hdr"), f"Need path to location HDR file, not binary. Got {loc_path}."
    assert os.path.splitext(output_file)[-1] == ".json", "Output file needs to be a json"
    
    with fsspec.open(rdn_path, "r") as f:
        rdn_meta = read_envi_header(f)

    nsamp = int(rdn_meta["samples"])
    nlines = int(rdn_meta["lines"])

    waves = np.array(rdn_meta["wavelength"], np.float32)
    waves_b64 = string_encode(waves)
    waves_dict = format_dict(name="wavelength", data=waves_b64, dims=["wavelength"], chunks=[len(waves)], shape=[len(waves)], dtype="<f4")
    
    lines = np.arange(nlines, dtype="<i4")
    lines_b64 = string_encode(lines)
    lines_dict = format_dict(name="line", data=lines_b64, dims=["line"], chunks=[nlines], shape=[nlines], dtype="<i4")

    samps = np.arange(nsamp, dtype="<i4")
    samps_b64 = string_encode(samps)
    samps_dict = format_dict(name="sample", data=samps_b64, dims=["sample"], chunks=[nsamp], shape=[nsamp], dtype="<i4")

    rdn_data = rdn_path.rstrip(".hdr")
    rdn_byte_order = {"0": "<", "1": ">"}[rdn_meta["byte order"]]
    rdn_dtype = envi_dtypes[rdn_meta["data type"]].newbyteorder(rdn_byte_order)
    rdn_interleave = rdn_meta["interleave"]
    assert rdn_interleave == "bil", f"Interleave {rdn_interleave} unsupported. Only BIL interleave currently supported."
    ra = rdn_dtype.alignment
    radiance_chunks = {
        f"radiance/{i}.0.0": [rdn_data, i*nsamp*len(waves)*ra, nsamp*len(waves)*ra] for i in range(nlines)
    }
    radiance_dict = {
        "radiance/.zarray": ujson.dumps({
            **zarray_common,
            "chunks": [1, len(waves), nsamp],
            "dtype": rdn_dtype.str,  # < = Byte order 0; f4 = data type 4
            "shape": [nlines, len(waves), nsamp],
        }),
        "radiance/.zattrs": ujson.dumps({
            "_ARRAY_DIMENSIONS": ["line", "wavelength", "sample"]
        }),
        **radiance_chunks
    }

    # Location file
    with fsspec.open(loc_path, "r") as f:
        loc_meta = read_envi_header(f)
    loc_data = loc_path.rstrip(".hdr")
    loc_byte_order = {"0": "<", "1": ">"}[loc_meta["byte order"]]
    loc_dtype = envi_dtypes[loc_meta["data type"]].newbyteorder(loc_byte_order)
    loc_interleave = loc_meta["interleave"]
    assert loc_interleave == "bil", f"Interleave {loc_interleave} unsupported. Only BIL interleave currently supported."
    la = loc_dtype.alignment
    lat_chunks = {
        f"lat/{i}.0": [loc_data, (i*nsamp*3)*la, nsamp*la] for i in range(nlines)
    }
    lat_dict = {
        "lat/.zarray": ujson.dumps({
            **zarray_common,
            "chunks": [1, nsamp],
            "dtype": loc_dtype.str,
            "shape": [nlines, nsamp],
        }),
        "lat/.zattrs": ujson.dumps({
            "_ARRAY_DIMENSIONS": ["line", "sample"],
        }),
        **lat_chunks
    }

    lon_chunks = {
        f"lon/{i}.0": [loc_data, (i*nsamp*3 + nsamp)*la, nsamp*la] for i in range(nlines)
    }
    lon_dict = {
        "lon/.zarray": ujson.dumps({
            **zarray_common,
            "chunks": [1, nsamp],
            "dtype": loc_dtype.str,
            "shape": [nlines, nsamp],
        }),
        "lon/.zattrs": ujson.dumps({
            "_ARRAY_DIMENSIONS": ["line", "sample"]
        }),
        **lon_chunks
    }

    output = {
        "version": 1,
        "refs": {
            ".zgroup": ujson.dumps({"zarr_format": 2}),
            ".zattrs": ujson.dumps({"loc": {**loc_meta}, "rdn": {**rdn_meta}}),
            **waves_dict, **samps_dict, **lines_dict,
            **radiance_dict, **lat_dict, **lon_dict
        }
    }

    with fsspec.open(output_file, "w") as of:
        of.write(ujson.dumps(output, indent=2))

    return output_file

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description = "Create Kerchunk metadata for ENVI binary files")
#     parser.add_argument("rdn_path", metavar="Radiance HDR path", type=str,
#                         help = "Path to radiance HDR file.")
#     parser.add_argument("output_file", metavar="Outut JSON path", type=str,
#                         help = "Path to target output JSON file.")
#     parser.add_argument("--loc_path", metavar="Location HDR path", type=str,
#                         help = "Path to location HDR file.")
#     parser.add_argument("--obs_path", metavar="Observation HDR path", type=str,
#                         help = "Path to observation HDR file.")

#     args = parser.parse_args()
#     rdn_path = args.rdn_path
#     loc_path = args.loc_path
#     obs_path = args.obs_path
#     if loc_path is None:
#         loc_path = rdn_path.replace("rdn", "loc")
#         print(f"Loc path not set. Assuming {loc_path}.")
#     if obs_path is None:
#         obs_path = rdn_path.replace("rdn", "obs")
#         print(f"Obs path not set. Assuming {obs_path}.")
#     output_file = make_envi_kerchunk(rdn_path, loc_path, obs_path, args.output_file)
#     print(f"Successfully created output file {output_file}.")
