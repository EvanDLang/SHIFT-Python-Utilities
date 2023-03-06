import numpy as np
import fsspec
import ujson

# from shift_utilities.envi_kerchunk.utils import string_encode, read_envi_header, zarray_common, envi_dtypes, parse_date ,format_dict


def gleon_kerchunk(rfl_hdr_path, output_file):
    """
    Reformats Gleon data as a zarr

    Parameters
    ----------
    rfl_hdr_path : str
        Path to reflectance header file
        
    output_file : str
        Write path
    
    Returns
    -------
        None (writes zarr)
    """
    assert os.path.splitext(output_file)[-1] == ".json", "Output file needs to be a json"
    
    # Read the header
    with fsspec.open(rfl_hdr_path, "r") as f:
        rfl_meta = read_envi_header(f)
    
    # Parse meta-data
    nsamp = int(rfl_meta["samples"])
    nlines = int(rfl_meta["lines"])
    
    # Get wavelengths and convert them to floats
    waves = np.array(rfl_meta["wavelength"], np.float32)
    waves_b64 = string_encode(waves)
    # format wavelength dict
    waves_dict = format_dict(name="wavelength", data=waves_b64, dims=["wavelength"], chunks=[len(waves)], shape=[len(waves)], dtype="<f4")
    
    # format lines dict
    lines = np.arange(nlines, dtype="<i4")
    lines_b64 = string_encode(lines)
    lines_dict = format_dict(name="line", data=samps_b64, dims=["line"], chunks=[nlines], shape=[nlines], dtype="<i4")

    # format sample dict
    samps = np.arange(nsamp, dtype="<i4")
    samps_b64 = string_encode(samps)
    sample_dict = format_dict(name="sample", data=samps_b64, dims=["sample"], chunks=[nsamp], shape=[nsamp], dtype="<i4")
    
    # format reflectance data
    rfl_data = rfl_hdr_path.rstrip(".hdr")
    rfl_byte_order = {"0": "<", "1": ">"}[rfl_meta["byte order"]]
    rfl_dtype = envi_dtypes[rfl_meta["data type"]].newbyteorder(rfl_byte_order)
    rdn_interleave = rfl_meta["interleave"]
    assert rdn_interleave == "bil", f"Interleave {rdn_interleave} unsupported. Only BIL interleave currently supported."
    ra = rfl_dtype.alignment
    reflectance_chunks = {
        f"reflectance/{i}.0.0": [rfl_data, i*nsamp*len(waves)*ra, nsamp*len(waves)*ra] for i in range(nlines)
    }
    reflectance_dict = {
        "reflectance/.zarray": ujson.dumps({
            **zarray_common,
            "chunks": [1, len(waves), nsamp],
            "dtype": rfl_dtype.str,  # < = Byte order 0; f4 = data type 4
            "shape": [nlines, len(waves), nsamp],
        }),
        "reflectance/.zattrs": ujson.dumps({
            "_ARRAY_DIMENSIONS": ["line", "wavelength", "sample"]
        }),
        **reflectance_chunks
    }
    
    # format output
    output = {
        "version": 1,
        "refs": {
            ".zgroup": ujson.dumps({"zarr_format": 2}),
            ".zattrs": ujson.dumps({**rfl_meta}),
            **waves_dict, **samps_dict, **lines_dict,
            **reflectance_dict
        }
    }

    # write data
    with fsspec.open(output_file, "w") as of:
        of.write(ujson.dumps(output, indent=2))


    ## Test that the output can be read
#     import xarray as xr
#     dat = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
#         "consolidated": False,
#         "storage_options": {"fo": output_file}
#     })

#     dat.isel(sample=300, line=3500).reflectance.values
