import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import numpy as np
from shapely.geometry import Point

from ._utils import reformat_path, _retrieve_crs_from_igm

"""
WIP
- orthorectified result has extra nodata values
"""
def clip(src, gdf):
    
    coords = [k for k  in src.coords.keys()]
    
    path = src.encoding['source']
    igm_path = reformat_path('igm', path)
    igm = rxr.open_rasterio(igm_path)
    crs = _retrieve_crs_from_igm(igm)

    if gdf.crs.to_epsg() != crs.to_epsg():
        gdf = gdf.to_crs(crs)
        

    if 'lat' not in coords or 'lon' not in coords:
        src = src.assign_coords({'lon': (('y', 'x'), igm.isel(band=0).values), 'lat': (('y', 'x'), igm.isel(band=1).values)})
      
    # get bounds of the geodataframe
    x_range = gdf.bounds['minx'].values[0], gdf.bounds['maxx'].values[0]
    y_range = gdf.bounds['miny'].values[0], gdf.bounds['maxy'].values[0] 
    
    # create a mask using the bounds
    m1 = ((src.lon.values <= x_range[1]) & (src.lon.values >= x_range[0])) & ((src.lat.values <= y_range[1]) & (src.lat.values >= y_range[0]))
   
    if not np.sum(m1) > 0:
        Exception("GeoDataFrame does not overlap source data!")
        
    # roughly clip data to bounds
    sub_shape = src.where(xr.DataArray(m1, dims=('y', 'x')), drop=True)
    
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
    m2[groups.groups[0.]] = True
    m2 = m2.reshape(sub_shape.shape[1:])
    
    # use the mask to subset the roughly clipped data
    res = sub_shape.where(xr.DataArray(m2, dims=('y', 'x')), drop=True).chunk()
    
    res = res.drop(['lat', 'lon'])
    
    res.encoding = src.encoding
    
    return res
