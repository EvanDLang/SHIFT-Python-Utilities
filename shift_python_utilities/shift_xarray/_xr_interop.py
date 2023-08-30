"""
Add ``.shift.`` extension to :py:class:`xr.Dataset` and :class:`xr.DataArray`.
"""
import functools
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import xarray as xr
import rioxarray as rxr
import math as m
from rasterio.crs import CRS
from ._dask_orthorectification import _dask_orthorectification

# XarrayObject = Union[xarray.DataArray, xarray.Dataset]
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

def _retrieve_crs(url):
    igm = _load_file(url)
    description = igm.attrs['description']
    ind = description.find("UTM")
    coord_system, _, zone, direction = description[57:].split(" ")
    direction = False if direction == 'North' else True
    epsg_code = 32600
    epsg_code += int(zone)
    if direction is True:
        epsg_code += 100

    return CRS.from_epsg(epsg_code)


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

def xr_orthorectify(src: XrT) -> XrT:
    """
    Orthorectify raster
    """
    assert 'y' in src.coords and 'x' in src.coords
    
    if isinstance(src, xr.Dataset):
        for dv in src.data_vars:
            if 'source' in src.data_vars[dv].encoding:
                path = src.data_vars[dv].encoding['source']
                break
    else:
        path = src.encoding['source']
        
    assert path is not None, "Unable to find path to GLT"

    
    x_sub = (m.floor(min(src.x.values)), m.ceil(max(src.x.values)))
    y_sub = (m.floor(min(src.y.values)), m.ceil(max(src.y.values)))

    glt = _load_file(path.replace('L2a', 'L1/glt')[:-4] + '_glt')
    crs = _retrieve_crs(path.replace('L2a', 'L1/igm')[:-4] + '_igm')
    glt = _subset_glt(glt, x_sub, y_sub)
    glt_array, v_glt = _create_valid_glt(glt, x_sub, y_sub)
    glt_dims = {dim: i for i, dim in enumerate(glt.dims)}
    
    lat, lon = _apply_transform(glt)
    
    if isinstance(src, xr.DataArray):
        src_dims = {dim: i for i, dim in enumerate(src.dims)}
        
        dst, coords, attrs, dims = _xr_orthorectify_da(src, src_dims, glt_array, v_glt, glt_dims)
        
        coords['x'] = lon
        coords['y'] = lat
     
        out = xr.DataArray(dst, coords=coords, dims=dims, attrs=attrs)
        
        return out
        
    
    out = _xr_orthorectify_ds(src, glt_array, v_glt, glt_dims, lat, lon)
    
    return out


def _xr_orthorectify_ds(
    src: Any,
    glt_array: XrT,
    v_glt: np.ndarray,
    glt_dims: dict,
    lat: np.ndarray,
    lon: np.ndarray
) -> xr.Dataset:
    
    assert isinstance(src, xr.Dataset)

    
    def _orthorectify_data_var(dv: xr.DataArray):
        src_dims = {dim: i for i, dim in enumerate(dv.dims)}
        
        if "y" not in src_dims and "x" not in src_dims:
            return dv
        
        dst, coords, attrs, dims =  _xr_orthorectify_da(dv, src_dims, glt_array, v_glt, glt_dims)
        
                
        coords['x'] = lon
        coords['y'] = lat
     
        out = xr.DataArray(dst, coords=coords, dims=dims, attrs=attrs)
        
        return out
   
    return src.map(_orthorectify_data_var)



def _xr_orthorectify_da(
    src: Any,
    src_dims: dict,
    glt_array: XrT,
    v_glt: np.ndarray,
    glt_dims: dict
) -> xr.DataArray:
    """
    Orthorectify raster
    """
    
    dst = _dask_orthorectification(src.data, src_dims, glt_array, v_glt, glt_dims)
    
    coords =  {k: v if v.chunks is None else v.compute() for k, v in src.coords.items() if k != 'spatial_ref'}
    
    if src.attrs is not None:
        attrs = src.attrs
    else:
        attrs = {}
        
    dims = src.dims
    # out = xr.DataArray(dst, coords=coords, dims=dims, attrs=attrs)
    
    return dst, coords, attrs, dims

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

    orthorectify = _wrap_op(xr_orthorectify)



@xr.register_dataset_accessor("SHIFT")
class  SHIFTExtensionDs:
    """
    ODC extension for :py:class:`xr.Dataset`.
    """

    def __init__(self, ds: xr.Dataset):
        self._xx = ds

    orthorectify = _wrap_op(xr_orthorectify)
    


