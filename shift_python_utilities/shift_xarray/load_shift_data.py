import re
import numpy as np
import xarray as xr
import rioxarray as rxr
import shift_python_utilities.shift_xarray.xr
from shift_python_utilities.shift_xarray.dask_merge import dask_merge_datasets
# from shift_python_utilities.intake_shift import shift_catalog
# import rasterio as rio
import pandas as pd
import geopandas as gpd
# from shapely.geometry import Polygon, mapping, box
from itertools import groupby
import datetime 
import dask.array as dask_array
import odc.geo.xr
import re


# initialize datasets and filepaths, hardcoded for now
L1_data = ['rdn', 'glt', 'igm', 'obs']
L2_data = ['rfl']
L1_base_path = '/efs/efs-data-curated/v1/{date}/L1/{dataset}/ang{date}t{time}_{dataset}'
L2_base_path = '/efs/efs-data-curated/v1/{date}/L2a/ang{date}t{time}_{dataset}'

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def _prep_file_paths(gdf, datasets):
    file_paths = {}

    for ind, (date, time) in gdf[['date', 'time']].iterrows():
        
        for dataset in datasets:
            if dataset in L1_data:
                path = L1_base_path.format(date=date, time=time, dataset=dataset)
            elif dataset in L2_data:
                path = L2_base_path.format(date=date, time=time, dataset=dataset)
            if dataset == 'rdn':
                path += '_v2aa1_clip'
            
            if date in file_paths:
                file_paths[date] += [path]
            else:
                file_paths[date] =  [path]
   
    return file_paths

def _load_data(file_paths, gdf, merge_strategy, chunks, resampling, res, crs):
    out_data = {}
    for date, file_path in file_paths.items():
    
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:])
        date = datetime.datetime(year, month, day)
        
        exp = re.compile('ang\d{8}t\d{6}')
        groups = [[fp for fp in file_path if name in fp] for name in set([exp.findall(fp)[0] for fp in file_path])]
        
        datasets = []
        for group in groups:
            if len(group) > 1:
                datasets += [xr.merge([rxr.open_rasterio(t, chunks=chunks).to_dataset(name=t.split('_')[-1] if 'rdn' not in t else 'rdn') for t in group])]
            else:
                datasets += [rxr.open_rasterio(t, chunks=chunks).to_dataset(name=t.split('_')[-1] if 'rdn' not in t else 'rdn') for t in group]
       
        to_merge = [ds.SHIFT.orthorectify(shapefile=gdf) for ds in datasets]
        
        if not all_equal([ds.rio.crs.to_epsg() for ds in to_merge]):
            if crs is None:
                crs = to_merge[0].rio.crs
            for i in range(len(to_merge)):
                if to_merge[i].rio.crs.to_epsg != crs:
                    how = to_merge[i].odc.output_geobox(crs)
                    to_merge[i] = to_merge[i].odc.reproject(how)
      
        if len(to_merge) > 1:
            to_merge = dask_merge_datasets(to_merge, method=merge_strategy, resampling=resampling, res=res, crs=crs)
        else:
       
            to_merge = to_merge[0]
            how = None
           
            if res is not None and crs is not None:
                how = to_merge.odc.output_geobox(crs, resolution=res)
            elif res is not None and crs is None:
                how = to_merge.odc.output_geobox(to_merge.rio.crs, resolution=res)
            elif res is None and crs is not None:
                how = to_merge.odc.output_geobox(crs) 
            
            if how is not None:
                to_merge = to_merge.odc.reproject(how)
                to_merge = to_merge.rename({to_merge.rio.x_dim: 'x', to_merge.rio.y_dim: 'y'})
        
        out_data[date] = to_merge
    
    return out_data

def load_shift_data(datasets,
                    gdf, 
                    date_range,
                    flight_lines=gpd.read_file('/efs/efs-data-curated/v1/shift_aviris_ng_raw_v1_shape/shift_aviris_ng_raw_v1.shp'),
                    merge_strategy='first',
                    all_touched=False,
                    chunks={'y':100},
                    res=None,
                    crs=None,
                    resampling='nearest'
                   ):
   
    # validate data argument
    if isinstance(datasets, list):
        for dataset in datasets:
            assert dataset in ['rfl', 'rdn', 'glt', 'igm', 'obs']
    else:
        assert datasets in ['rfl', 'rdn', 'glt', 'igm', 'obs']
        datasets = [datasets]
    
    # filter by date range
    intersecting = flight_lines[(flight_lines['date'] >= date_range[0]) & (flight_lines['date'] <= date_range[1])]
    
    assert len(intersecting) > 0, "Invalid date range"
    
    
    # gdf = gdf[['geometry']]
    
    temp = []
    for date in intersecting.date.unique():
        temp_intersecting = intersecting.loc[intersecting.date==date]
        temp_intersecting = temp_intersecting.sjoin(gdf.to_crs(temp_intersecting.crs), how='inner', predicate='contains')
        if len(temp_intersecting) < 1:
            temp_intersecting = intersecting.loc[intersecting.date==date]
            temp_intersecting = temp_intersecting.sjoin(gdf.to_crs(temp_intersecting.crs), how='inner')
        
        temp += [temp_intersecting]
    if len(temp) > 0:
        intersecting = pd.concat(temp)
    else:
        raise Exception("The provided shapefile does not overlap with the data")
    
    # intersecting = intersecting.sjoin(gdf.to_crs(intersecting.crs), how='inner', predicate='contains')
    # if len(intersecting) < 1:
    #     intersecting.sjoin(gdf.to_crs(intersecting.crs), how='inner')
    
    assert len(intersecting) > 0, "Shapefile does not overlap with any of the flight lines"
    
    # generate the file paths for the data
    file_paths = _prep_file_paths(intersecting, datasets)
    
    # load and merge datasets
    out_data = _load_data(file_paths=file_paths, gdf=gdf, merge_strategy=merge_strategy, chunks=chunks, resampling=resampling, res=res, crs=crs)
    
    # sort data by date
    temp_keys = list(out_data.keys())
    temp_keys.sort()
    out_data = {k: out_data[k] for k in temp_keys}
    
    # merge datasets
    if len(out_data) > 1:
        dates, data = zip(*[(k,v) for k,v in out_data.items()])
        
        representative_ds = data[0]

        out_data = []
        for data_var in representative_ds.data_vars:
            to_concat = [d[data_var] for d in data]
            temp = dask_array.concatenate([dask_array.expand_dims(d.data, axis=0) for d in to_concat], axis=0)
            coords = dict(data[0].coords.items())
            coords['time'] = list(dates)
            out_data += [xr.DataArray(temp, dims=('time',) + to_concat[0].dims, coords=coords, name=data_var)]
        out_data = xr.merge(out_data)
    
    else:
        out_data = out_data[list(out_data.keys())[0]]

    return out_data
        