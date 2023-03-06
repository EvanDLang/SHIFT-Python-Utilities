import numpy as np
import fsspec
import ujson
import os
import re
import datetime
import s3fs
import pyproj
# from shift_utilities.envi_kerchunk.utils import string_encode, read_envi_header, zarray_common, envi_dtypes, parse_date ,format_dict

def make_shift_multi(input_directory, output_file):
    """
    Reformats a directory of shift data into a single zarr

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

    dates = np.array([parse_date(f) for f in flist])
    dates_dtype = np.dtype(dates[0])
    time_dict = format_dict(name="time", data=string_encode(dates), dims=["time"], chunks=[len(dates)], shape=[len(dates)], dtype=dates_dtype.str)

    # ENVI metadata for all of these should be the same
    rfl_path = flist[0]
    with fsspec.open(rfl_path, "r") as f:
        rfl_meta = read_envi_header(f)

    nsamp = int(rfl_meta["samples"])
    nlines = int(rfl_meta["lines"])

    waves = np.array(rfl_meta["wavelength"], np.float32)
    waves_b64 = string_encode(waves)
    waves_dict = format_dict(name="wavelength", data=waves_b64, dims=["wavelength"], chunks=[len(waves)], shape=[len(waves)], dtype="<f4")

    fwhm = np.array(rfl_meta["fwhm"], np.float32)
    fwhm_b64 = string_encode(fwhm)
    fwhm_dict = format_dict(name="fwhm", data=fwhm_b64, dims=["wavelength"], chunks=[len(waves)], shape=[len(waves)], dtype="<f4")

    # Parse map information to generate X,Y coordinates
    metadata = parse_map_info(rfl_meta)

    # Project line/sample into X/Y using projection information
    lines = np.arange(nlines)
    ycoords = metadata['px_northing'] - metadata['y_size'] * (lines + 0.5)
    samples = np.arange(nsamp)
    xcoords = metadata['px_easting'] + metadata['x_size'] * (samples + 0.5)
    
    # format y
    y_b64 = string_encode(ycoords)
    y_dict = format_dict(name="y", data=y_b64, dims=["y"], chunks=[nlines], shape=[nlines], dtype=ycoords.dtype.str)
    
    #format x
    x_b64 = string_encode(xcoords)
    x_dict = format_dict(name="x", data=x_b64, dims=["x"], chunks=[nsamp], shape=[nsamp], dtype=xcoords.dtype.str)

    crs_string = ",".join(rfl_meta["coordinate system string"])
    crs = pyproj.CRS(crs_string)
    spref_dict = {
        "spatial_ref/.zarray": ujson.dumps({
            **zarray_common,
            "chunks": [],
            "dtype": np.dtype(np.int64).str,
            "shape": []
        }),
        "spatial_ref/.zattrs": ujson.dumps({
            **crs.to_cf(),
            "spatial_ref": crs_string,
            "GeoTransform": f"{px_easting} {x_size:.1f} -0.0 {px_northing} -0.0 -{y_size:.1f}",
            "_ARRAY_DIMENSIONS": []
        }),
        "spatial_ref/0": string_encode(np.int64(0))
    }

    byte_order = {"0": "<", "1": ">"}[rfl_meta["byte order"]]
    rfl_dtype = envi_dtypes[rfl_meta["data type"]].newbyteorder(byte_order)
    rfl_interleave = rfl_meta["interleave"]
    assert rfl_interleave == "bil", f"Interleave {rfl_interleave} unsupported. Only BIL interleave currently supported."
    ra = rfl_dtype.alignment
    reflectance_chunks = {}
    for t in range(len(dates)):
        rfl_data = flist[t].rstrip(".hdr")
        reflectance_chunks_t = {
            f"reflectance/{t}.{i}.0.0": [rfl_data, i*nsamp*len(waves)*ra, nsamp*len(waves)*ra] for i in range(nlines)
        }
        reflectance_chunks = {**reflectance_chunks, **reflectance_chunks_t}

    reflectance_dict = {
        "reflectance/.zarray": ujson.dumps({
            **zarray_common,
            "chunks": [1, 1, len(waves), nsamp],
            "dtype": rfl_dtype.str,
            "shape": [len(dates), nlines, len(waves), nsamp],
        }),
        "reflectance/.zattrs": ujson.dumps({
            "_ARRAY_DIMENSIONS": ["time", "y", "wavelength", "x"]
        }),
        **reflectance_chunks
    }

    output = {
        "version": 1,
        "refs": {
            ".zgroup": ujson.dumps({"zarr_format": 2}),
            ".zattrs": ujson.dumps({**rfl_meta}),
            **waves_dict, **x_dict, **y_dict, **time_dict,
            **spref_dict,
            **reflectance_dict
        }
    }

    with fsspec.open(output_file, "w") as of:
        of.write(ujson.dumps(output, indent=2))
