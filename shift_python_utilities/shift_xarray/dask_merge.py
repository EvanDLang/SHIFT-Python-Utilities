from rioxarray.merge import merge_arrays
import dask.array as da
from dask.delayed import delayed
import numpy as np
import rioxarray as rxr
from affine import Affine
import numpy as np
import xarray as xr
from uuid import uuid4
from rasterio.crs import CRS
from rioxarray.rioxarray import _make_coords
from rasterio.enums import Resampling
from rasterio.merge import copy_first, copy_last, copy_min, copy_max, copy_sum, copy_count
import odc.geo.xr
from shift_python_utilities.shift_xarray._utils import _chunk_maker
from typing import Union, Callable



MERGE_METHODS = {
    "first": copy_first,
    "last": copy_last,
    "min": copy_min,
    "max": copy_max,
    "sum": copy_sum,
    "count": copy_count,
}

def dask_merge_arrays(srcs: list[xr.DataArray], 
                      res: float=None, 
                      crs: Union[int, CRS]=None, 
                      nodata: float=None, 
                      resampling: str='nearest', 
                      method: Union[str, Callable]='first'
                     ) -> xr.DataArray:
    """
    Dask lazily executed version of rasterio.merge for xr.DataArrays

    Parameters
    ----------
        srcs
            A list of xr.DataArrays to be merged
        res
            Output resolution. Default: None, uses the resolution of the first data array in the list
        crs
            Output crs. Default: None, uses the crs of the first data array in the list
        nodata
            Output no data value. Default: None, uses the no data value of the first data array in the list
        resampling
            resampling method, Default: nearest, See rasterios resampling methods for more options
        method
             Describes how to handle overlapping flightlines within the area of interest. Default: 'first'. Provided strategies : ['first', 'last', 'max', 'min', 'sum', count]. Additionally, a custom merge strategy can be provide in the form of a Callable. The merge methods are based on rasterio.merge. See rasterio.merge for more information and examples of how to format the custom Callable.
    
    Returns
    -------
        xr.DataArray
            Merged xr.DataArrays
    """
    assert isinstance(srcs, list) and len(srcs) > 1, 'srcs must be a list of at least length 2!'
    
    if isinstance(method, str):
        assert method in MERGE_METHODS, f"Invalid method, The method must be one of the following {list(MERGE_METHODS.keys())} or a custom callable"
        copyto = MERGE_METHODS[method]
    elif isinstance(method, callable):
        copyto = method
    else:
        raise ValueError('Unknown method {0}, must be one of {1} or callable'
                         .format(method, list(MERGE_METHODS.keys())))
    
    valid_resampling = [r for r in dir(Resampling) if '_' not in r]
    if resampling not in valid_resampling:
        raise ValueError('Unknown resampling method {0}, must be one of {1}'.format(resampling, valid_resampling))
    
    resampling = getattr(Resampling, resampling)
    
    if res is None:
        res = srcs[0].rio.resolution()
    elif isinstance(res, int) or isinstance(res, float):
        res = (float(res), -float(res))
    else:
        raise ValueError

    if crs is None:
        crs = srcs[0].rio.crs
    
    if nodata is None:
        nodata = srcs[0].rio.nodata if srcs[0].rio.nodata is not None else -9999.

    dt = srcs[0].dtype
    xs = []
    ys = []
    
    for i in range(len(srcs)):
        # determine if any resampling needs to take place
        res_diff = np.asarray(res) - np.asarray(srcs[i].rio.resolution())
        res_diff = res_diff[0] if res_diff[0] > 0 else res_diff[0] * -1

        if res_diff >= .01 or crs.to_epsg() != srcs[i].rio.crs.to_epsg():
            how = srcs[i].odc.output_geobox(crs, resolution=res[0])
            srcs[i] = srcs[i].odc.reproject(how, resampling=resampling, dst_nodata=nodata)
            if 'y' not in srcs[i].dims:
                srcs[i] = srcs[i].rename({srcs[i].rio.y_dim: 'y', srcs[i].rio.x_dim: 'x'})
               
         
        left, bottom, right, top = srcs[i].rio.bounds()
        xs.extend([left, right])
        ys.extend([bottom, top])
  
    dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    output_width = int(round((dst_e - dst_w) / res[0]))
    output_height = int(round((dst_n - dst_s) / -res[1]))
    
    output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], res[1])
   
    output_dims = {"y": output_height, "x": output_width}
    dim_map = {dim: i for i, dim in enumerate(srcs[0].dims)}
    dst_shape = tuple([output_dims[k] if k == 'x' or k == 'y' else srcs[0].shape[v] for k,v in dim_map.items()])
    dst_chunks = _chunk_maker(srcs[0].data.chunksize, dst_shape)
 
    out = delayed(merge_arrays)(srcs, nodata=nodata, method=method)
  
    out = da.from_delayed(out, shape=dst_shape, dtype=dt)
    name = 'merge'
    tk = uuid4().hex
    name = f"{name}-{tk}"
    
    coords = _make_coords(
        srcs[0],
        output_transform,
        output_dims['x'],
        output_dims['y'],
    )
    
    out = xr.DataArray(
        name=srcs[0].name,
        data=out,
        coords=coords,
        dims=tuple(srcs[0].dims),
        attrs=srcs[0].attrs,
    ).chunk(dst_chunks)
    
    out = out.rio.write_nodata(nodata, encoded=True)
    out = out.rio.write_nodata(nodata)
    out = out.rio.write_crs(crs)
    out = out.rio.write_transform(output_transform)
    
    return out


def dask_merge_datasets(srcs: list[xr.Dataset], res: float=None, crs: Union[int, CRS]=None, nodata: float=None, resampling: str='nearest', method: Union[str, Callable]='first') -> xr.Dataset:
    """
    Dask lazily executed version of rasterio.merge for xr.DataSets

    Parameters
    ----------
        srcs
            A list of xr.DataSets to be merged
        
        See dask_merge_arrays
    
    Returns
    -------
        xr.DataArray
            Merged xr.DataSets
    """
    representative_ds = srcs[0]
    merged_data = {}
    
    for data_var in representative_ds.data_vars:
        merged_data[data_var] = dask_merge_arrays(
            [src[data_var] for src in srcs],
            res=res,
            resampling=resampling,
            crs=crs,
            nodata=nodata if nodata is not None else representative_ds[data_var].rio.nodata,
            method=method       
        )
        
    data_var = list(representative_ds.data_vars)[0]
    
    xds = xr.Dataset(
        merged_data,
        coords=_make_coords(
            merged_data[data_var],
            merged_data[data_var].rio.transform(),
            merged_data[data_var].shape[-1],
            merged_data[data_var].shape[-2],
            force_generate=True,
        ),
        attrs=representative_ds.attrs,
    )
    xds.rio.write_crs(merged_data[data_var].rio.crs, inplace=True)
    return xds