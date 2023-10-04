"""
Add ``.shift.`` extension to :py:class:`xr.Dataset` and :class:`xr.DataArray`.
"""
import functools

from typing import (
    Any,
    Callable,
    TypeVar
)

import numpy as np
import xarray as xr
import rioxarray as rxr
import math as m
from rasterio.crs import CRS
from ._dask_orthorectification import _dask_orthorectification
from ._dask_ipca import Dask_IPCA_SK, Dask_IPCA_DS
from ._clip import clip
from ._utils import _retrieve_crs_from_igm, reformat_path

XrT = TypeVar("XrT", xr.DataArray, xr.Dataset)
F = TypeVar("F", bound=Callable)

def _subset_glt(glt, x_sub, y_sub):
    y_inds = xr.DataArray(list(range(y_sub[0], y_sub[1])))
    x_inds = xr.DataArray(list(range(x_sub[0], x_sub[1])))
    return glt.where(glt[1, :, :].compute().isin(y_inds) & (glt[0, :, :].compute().isin(x_inds)), drop=True)

def _create_valid_glt(glt, x_sub, y_sub):
    glt_array = glt.fillna(-9999).astype(int).values
    valid_glt = np.all(glt_array != -9999, axis=0)
    
    if not x_sub is None:
        glt_array[0, valid_glt] -= min(x_sub)
    else:
        glt_array[0, valid_glt] -= 1
    
    if not y_sub is None:
        glt_array[1, valid_glt] -= min(y_sub)
    else:
        glt_array[1, valid_glt] -= 1
    
    return glt_array, valid_glt

def _load_file(path):
    return rxr.open_rasterio(path)

def _apply_transform(glt):
    GT = glt.rio.transform()
    dim_x = glt.x.shape[0]
    dim_y = glt.y.shape[0]
    
    lon = np.zeros(dim_x)
    lat = np.zeros(dim_y)

    for x in np.arange(dim_x):
        x_geo = GT[2] + x * GT[0]
        lon[x] = x_geo
    for y in np.arange(dim_y):
        y_geo = GT[5] + y * GT[4]
        lat[y] = y_geo
    
    return lat, lon

def xr_orthorectify(src: XrT, url=None) -> XrT:
    """
    Orthorectify raster
    """
    assert 'y' in src.coords and 'x' in src.coords
    
    if isinstance(src, xr.DataArray):
        if url is not None:
            assert isinstance(url, str), "The GLT url must be a string when orthorectifying a dataset"
        
        src_dims = {dim: i for i, dim in enumerate(src.dims)}
        
        return _xr_orthorectify_da(src, src_dims, url)      
    
    if isinstance(url, list):
        assert len(url) == len(src.data_vars), "Then number of urls must match the number of data vars. You can also pass a single url as a string to be used for all data vars"
    elif isinstance(url, str):
        url = [url for i in range(len(src.data_vars))]
    elif url is None:
        url = [None for i in range(len(src.data_vars))]
        
    assert isinstance(url, list), "When passing glt urls for a dataset you must pass a list of urls equal to the number of datavars or a single url to be used for all data vars"
    
    url = {name: url[i] for i, (name, array) in enumerate(src.data_vars.items())}
    
    return _xr_orthorectify_ds(src, url)


def _xr_orthorectify_ds(
    src: Any,
    url: dict
) -> xr.Dataset:
    
    assert isinstance(src, xr.Dataset)
    
    
    def _orthorectify_data_var(dv: xr.DataArray, urls):
        src_dims = {dim: i for i, dim in enumerate(dv.dims)}
        
        if "y" not in src_dims and "x" not in src_dims:
            return dv
        
        url = urls[dv.name]
        
        return _xr_orthorectify_da(dv, src_dims, url)
    
   
    return src.map(_orthorectify_data_var, urls=url)


# adjust to allow passing file path
def _xr_orthorectify_da(
    src: Any,
    src_dims: dict,
    url
) -> xr.DataArray:
    """
    Orthorectify raster
    """
    if url is not None:
        path = url
    else:
        if 'source' in src.encoding:
            path = src.encoding['source']
        else:
            raise Exception("A url to the glt must be provided or having the original data path stored in source")
    
    x_sub = (m.floor(min(src.x.values)), m.ceil(max(src.x.values)))
    y_sub = (m.floor(min(src.y.values)), m.ceil(max(src.y.values)))
    
    glt = _load_file(reformat_path('glt', path))
    crs = _retrieve_crs_from_igm(_load_file(reformat_path('igm', path)))
    glt = _subset_glt(glt, x_sub, y_sub)
    glt_array, v_glt = _create_valid_glt(glt, x_sub, y_sub)
    glt_dims = {dim: i for i, dim in enumerate(glt.dims)}
    
    lat, lon = _apply_transform(glt)
    
    nodata = -9999 if src.rio.nodata is None else src.rio.nodata

    dst = _dask_orthorectification(src.data, src_dims, glt_array, v_glt, glt_dims, nodata)
    
    coords =  {k: v if v.chunks is None else v.compute() for k, v in src.coords.items() if k != 'spatial_ref'}
    
    coords['x'] = lon
    coords['y'] = lat
    
    out = xr.DataArray(dst, coords=coords, dims=src.dims, attrs=src.attrs).rio.write_crs(crs, inplace=True)
    out = out.rio.write_nodata(nodata)
    out.encoding = src.encoding
    
    return out

def _wrap_op(method: F) -> F:
    @functools.wraps(method, assigned=("__doc__",))
    def wrapped(*args, **kw):
        # pylint: disable=protected-access
        _self, *rest = args
        return method(_self._xx, *rest, **kw)

    return wrapped  # type: ignore


@xr.register_dataarray_accessor("SHIFT")
class SHIFTExtensionDa:
    """
    ODC extension for :py:class:`xr.DataArray`.
    """

    def __init__(self, xx: xr.DataArray):
        self._xx = xx
    
    def init_ipca(self, feature_dim, base='dask', *args, **kwargs):
        if base == 'dask':
            self.ipca = Dask_IPCA_DS(self._xx, feature_dim, *args, **kwargs)
        elif base == 'sklearn':
            self.ipca = Dask_IPCA_SK(self._xx, feature_dim, *args, **kwargs)
        return self.ipca
        
        
    orthorectify = _wrap_op(xr_orthorectify)
    clip = _wrap_op(clip)



@xr.register_dataset_accessor("SHIFT")
class  SHIFTExtensionDs:
    """
    ODC extension for :py:class:`xr.Dataset`.
    """

    def __init__(self, ds: xr.Dataset):
        self._xx = ds

    orthorectify = _wrap_op(xr_orthorectify)
    
