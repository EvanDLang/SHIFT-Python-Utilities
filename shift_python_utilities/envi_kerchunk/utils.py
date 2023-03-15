import fsspec
import warnings
import base64
import numpy as np
import ujson

def string_encode(x):
    """
    Encodes a string as base64

    Parameters
    ----------
    x : str
        String to be encoded
        
    Returns
    -------
        encoded_string: str
            encoded_string converted to bits then formated as a string
    """
    bits = base64.b64encode(x)
    s = str(bits).lstrip("b'").rstrip("'")
    return f"base64:{s}"

def parse_date(fname):
    """
    Parses the date from a file name and returns it as an np.datetime64

    Parameters
    ----------
    fname : str
        File name to be parsed
        
    Returns
    -------
        date: np.dateime64
            parsed date
    """
    basename = os.path.basename(fname)
    dstring = re.match(r'\d{8}', basename)
    assert dstring
    date = datetime.datetime.strptime(dstring.group(), "%Y%m%d")
    return np.datetime64(date)


def read_envi_header(f):
    """
    Reads an ENVI ".hdr" file header and returns the parameters in a
    dictionary as strings.  Header field names are treated as case
    insensitive and all keys in the dictionary are lowercase.
    
    ENVI header description: 
    https://www.l3harrisgeospatial.com/docs/enviheaderfiles.html

    Parameters
    ----------
    f : str
        Path to envi header file
        
    Returns
    -------
        parsed_header: dict
    """

 

    try:
        starts_with_ENVI = f.readline().strip().startswith('ENVI')
    except UnicodeDecodeError:
        msg = 'File does not appear to be an ENVI header (appears to be a ' \
          'binary file).'
        raise Exception(msg)
    else:
        if not starts_with_ENVI:
            msg = 'File does not appear to be an ENVI header (missing "ENVI" \
              at beginning of first line).'
            raise Exception(msg)

    lines = f.readlines()

    dict = {}
    have_nonlowercase_param = False
    # support_nonlowercase_params = spy.settings.envi_support_nonlowercase_params
    support_nonlowercase_params = False
    try:
        while lines:
            line = lines.pop(0)
            if line.find('=') == -1: continue
            if line[0] == ';': continue

            (key, sep, val) = line.partition('=')
            key = key.strip()
            if not key.islower():
                have_nonlowercase_param = True
                if not support_nonlowercase_params:
                    key = key.lower()
            val = val.strip()
            if val and val[0] == '{':
                str = val.strip()
                while str[-1] != '}':
                    line = lines.pop(0)
                    if line[0] == ';': continue

                    str += '\n' + line.strip()
                if key == 'description':
                    dict[key] = str.strip('{}').strip()
                else:
                    vals = str[1:-1].split(',')
                    for j in range(len(vals)):
                        vals[j] = vals[j].strip()
                    dict[key] = vals
            else:
                dict[key] = val

        if have_nonlowercase_param and not support_nonlowercase_params:
            msg = 'Parameters with non-lowercase names encountered ' \
                  'and converted to lowercase. To retain source file ' \
                  'parameter name capitalization, set ' \
                  'spectral.settings.envi_support_nonlowercase_params to ' \
                  'True.'
            warnings.warn(msg)
            # logger.debug('ENVI header parameter names converted to lower case.')
        return dict
    except:
        raise Exception()

zarray_common = {
    "compressor": None,
    "fill_value": None,
    "filters": None,
    "order": "C",
    "zarr_format": 2
}

# ENVI header description: 
# https://www.l3harrisgeospatial.com/docs/enviheaderfiles.html

envi_dtypes = {
    "1": np.dtype("int8"),
    "2": np.dtype("int16"),
    "3": np.dtype("int32"),
    "4": np.dtype("float32"),     # float32
    "5": np.dtype("float64"),     # float64
    "6": np.dtype("complex64"),     # complex64
    "9": np.dtype("complex128"),   # complex128
    "12": np.dtype("uint16"),
    "13": np.dtype("uint32"),
    "14": np.dtype("int64"),
    "15": np.dtype("uint64")
}

def format_dict(name, data, dims, chunks, shape, dtype):
    """
    Formats zarr metadata given the name ,data, dims, chunks, shape and dtype

    Parameters
    ----------
        
    Returns
    -------
        formated_dict: dict
           dictionary formated as zarr metadata
    """
    formated_dict = {
        f"{name}/.zarray": ujson.dumps({
            **zarray_common,
            "chunks": chunks,
            "dtype": dtype,
            "shape": shape,
        }),
        f"{name}/.zattrs": ujson.dumps({
            "_ARRAY_DIMENSIONS": dims
        }),
        f"{name}/0": data
    }
    
    return formated_dict

def parse_map_info(metadata):
    """
    Parses map info from envi mapinfo metadata

    Parameters
    ----------
    metadata : dict
        metadata to be parsed
        
    Returns
    -------
        metadata_dict: dict
            parsed map info
    """
    metadata_dict = { 
        "proj_name": metadata["map info"][0],
        "ref_x": int(metadata["map info"][1]),
        "ref_y": int(metadata["map info"][2]),
        "px_easting": float(metadata["map info"][3]),
        "px_northing": float(metadata["map info"][4]),
        "x_size": float(metadata["map info"][5]),
        "y_size": float(metadata["map info"][6]),
        "utm_zone": metadata["map info"][7],
        "north_south": rfl_meta["map info"][8],
        "datum": metadata["map info"][9]
    }
        
    return metadata_dict