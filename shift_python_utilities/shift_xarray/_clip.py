import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import numpy as np
from shapely.geometry import Point
import rasterio as rio
from typing import Union
import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from ._utils import reformat_path, _retrieve_crs_from_igm

"""
WIP
- orthorectified result has extra nodata values
- tests
- verify input handling
"""
def clip_array(src: xr.DataArray, gdf: gpd.GeoDataFrame, nodataval=-9999., crs: Union[int, rio.crs.CRS]=None, path: str=None) -> xr.DataArray:
    """
    A clipping function for unorthorectified data. An IGM file is required to use this or pixel latitude and longitude values along with the appropriate crs.
    
    Parameters
    ----------
        src
            Source data array to be clipped
        gdf
            Geodataframe to clip with
        crs
            CRS of the input data
        path
            Path to the igm file
    Returns
    -------
        xr.DataArray
            Clipped data array using geodataframe
    """
    
    coords = list(src.coords.keys())
    
    if ('lat' not in coords or 'lon' not in coords) and ('latitude' not in coords or 'longitude' not in coords):
        
        if path is None:
            path = src.encoding['source']
            path = reformat_path('igm', path)
            
        igm = rxr.open_rasterio(path)
        
        if crs is None:
            crs = _retrieve_crs_from_igm(igm)
        elif isinstance(crs, int):
            crs = rio.crs.CRS.from_epsg(crs)

        if gdf.crs.to_epsg() != crs.to_epsg():
            gdf = gdf.to_crs(crs)
        
        src = src.assign_coords({'lon': (('y', 'x'), igm.isel(band=0).values), 'lat': (('y', 'x'), igm.isel(band=1).values)})
    
    
    if crs is None:
        raise ValueError("A CRS cannot be determined, please provide one!")
    
    
    # get bounds of the geodataframe
    x_range = np.min(gdf.bounds['minx'].values), np.max(gdf.bounds['maxx'].values)
    y_range = np.min(gdf.bounds['miny'].values), np.max(gdf.bounds['maxy'].values)

    
    # create a mask using the bounds
    m1 = ((src.lon.values <= x_range[1]) & (src.lon.values >= x_range[0])) & ((src.lat.values <= y_range[1]) & (src.lat.values >= y_range[0]))
   
    if m1.sum() == 0:
        raise Exception("GeoDataFrame does not overlap source data!")
        
    # roughly clip data to bounds
    sub_shape = src.where(xr.DataArray(m1, dims=('y', 'x')), other=nodataval, drop=True)
    
    # using the remaining data, create a geodataframe of points
    points = gpd.GeoSeries(map(Point, zip(sub_shape.lon.values.flatten(), sub_shape.lat.values.flatten())))
    points = points.set_crs(crs.to_epsg())
    points = gpd.GeoDataFrame(geometry=points)
    
    # perform a spatial join with the points and shapes to find which points lie within the shapes
    points = points.sjoin(gdf, how='left')
    
    # get the indicies of the points that are within the shapes
    groups = points.groupby('index_right')
    
    #create a mask using the indicies
    m2 = np.zeros(len(points)).astype(bool)
    for group in groups.groups:
        m2[groups.groups[group]] = True
   
    
    dim_map = {d:i for i, d in enumerate(src.dims)}
    spatial_dims = [dim_map[src.rio.x_dim], dim_map[src.rio.y_dim]]
    spatial_dims.sort()
    shape = [sub_shape.shape[i] for i in spatial_dims]
    m2 = m2.reshape(shape)
  
    
    # use the mask to subset the roughly clipped data
    res = sub_shape.where(xr.DataArray(m2, dims=('y', 'x')), other=nodataval, drop=True).chunk()
    
    res.encoding = src.encoding
    
    return res


def clip_dataset(src: xr.Dataset, gdf: gpd.GeoDataFrame, crs: Union[int, rio.crs.CRS]=None, path: str=None) -> xr.Dataset:
    """
    A clipping function for unorthorectified data. An IGM file is required to use this or pixel latitude and longitude values along with the appropriate crs.
    
    Parameters
    ----------
        src
            Source dataset to be clipped
        gdf
            Geodataframe to clip with
        crs
            CRS of the input data
        path
            Path to the igm file
    Returns
    -------
        xr.DataSet
            Clipped dataset using geodataframe
    """
    
    def _clip(dv):
        return clip_array(dv, gdf, crs, path)
    
    return src.map(_clip)
        