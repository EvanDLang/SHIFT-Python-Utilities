import re
import numpy as np
import xarray as xr
import rioxarray as rxr
import shift_python_utilities.shift_xarray.xr
from shift_python_utilities.shift_xarray.dask_merge import dask_merge_datasets
import pandas as pd
import geopandas as gpd
from itertools import groupby
import datetime 
import dask.array as dask_array
import odc.geo.xr
import os
import rasterio as rio

from typing import Union, Callable

# initialize datasets and filepaths, hardcoded for now
L1_data = ['rdn', 'glt', 'igm', 'obs']
L2_data = ['rfl']
L1_base_path = '/efs/efs-data-curated/v1/{date}/L1/{dataset}/ang{date}t{time}_{dataset}'
L2_base_path = '/efs/efs-data-curated/v1/{date}/L2a/ang{date}t{time}_{dataset}'

def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def _prep_file_paths(gdf: gpd.GeoDataFrame, datasets: list[str]) -> list[str]:
    """
    Generates all of the required filepaths based on the provided area of interest

    Parameters
    ----------
        gdf
            Shapefile of flight lines
        datasets
            Data types to retrieve
    
    Returns
    -------
        list
            A list of filepaths
    """
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
                if path not in file_paths[date]:
                    file_paths[date] += [path]
            else:
                file_paths[date] =  [path]
   
    return file_paths

def _open_data(file_paths: list, chunks: dict) -> list[xr.Dataset]:
    """
    Supporting function that retrieves data

    Parameters
    ----------
        file_paths
            Paths to requested data
        chunks
            How the data is chunked in each dimension
    
    Returns
    -------
        list
            list of successfuly loaded xr.Datasets
    """
    datasets = []
    for fp in file_paths:
        if os.path.exists(fp):
            ds = rxr.open_rasterio(fp, chunks=chunks)
            if 'source' not in ds.encoding:
                ds.encoding['source'] = fp
            datasets += [ds.to_dataset(name=fp.split('_')[-1] if 'rdn' not in fp else 'rdn')]
        else:
            fname = re.compile('ang\d{8}t\d{6}').findall(fp)[0] + '_' + fp.split('_')[-1] if 'rdn' not in fp else 'rdn'
            print(f"The requested data ({fname}) does not exist!")
    return datasets
       
def _load_data(file_paths: list, gdf: gpd.GeoDataFrame, merge_strategy: Union[str, Callable], chunks: dict, resampling: str, res: Union[float, int], crs: Union[int, rio.crs. CRS]):
    """
    Supporting fuction which retrieves, merges and resamples data

    Parameters
    ----------
    file_paths
        paths to requested data

    Returns
    -------
        xr.Dataset, dict
            Successfully processed data, dict of data which failed to orthorectify
        dict, dict
    """
    out_data = {}
    failed = {}
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
                datasets += [xr.merge(_open_data(group, chunks))]
            else:
                datasets += _open_data(group, chunks)
      
        f = []
        to_merge = []
        for ds in datasets:
            try:
                to_merge += [ds.SHIFT.orthorectify(subset=gdf)]
            
            except Exception as e:
                if 'glt' in str(e):
                    fname = re.compile('ang\d{8}t\d{6}').findall(str(e))[0]
                    print(f"The requested data ({fname}) is missing a GLT file and cannot be orthorectified. Returning clipped unorthorectified data.")
                    f += [ds.SHIFT.clip(gdf)]
                else:
                    print(e)
                    continue
               
            
        if not _all_equal([ds.rio.crs.to_epsg() for ds in to_merge]):
            if crs is None:
                crs = to_merge[0].rio.crs
            for i in range(len(to_merge)):
                if to_merge[i].rio.crs.to_epsg != crs:
                    how = to_merge[i].odc.output_geobox(crs)
                    to_merge[i] = to_merge[i].odc.reproject(how)
      
        if len(to_merge) > 1:
            to_merge = dask_merge_datasets(to_merge, method=merge_strategy, resampling=resampling, res=res, crs=crs)
        
        elif len(to_merge) == 1:
       
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
                to_merge = to_merge.rename({to_merge.rio.y_dim: 'y', to_merge.rio.x_dim: 'x'}) 
               
        
        out_data[date] = to_merge
        
        if len(f) > 0:
            failed[date] = f
    
    return out_data, failed

def load_shift_data(datasets: list[str],
                    gdf: gpd.GeoDataFrame, 
                    date_range : Union[list['str'], tuple['str']],
                    flight_lines: gpd.GeoDataFrame = gpd.read_file('/efs/efs-data-curated/v1/shift_aviris_ng_raw_v1_shape/shift_aviris_ng_raw_v1.shp'),
                    merge_strategy: Union[str, Callable] = 'first',
                    chunks: dict[str: int] = {'y':100},
                    res: Union[float, int] = None,
                    crs: Union[int, rio.crs.CRS] = None,
                    resampling: str ='nearest'
                   ) -> Union[tuple[xr.Dataset, dict], xr.Dataset, dict, None]:
    """
    Loads SHIFT data from the flight lines using dask lazy execution

    Parameters
    ----------
    datasets
        List of datasets to load. Any combination of ['rfl', 'rdn', 'obs', 'igm']
    date_range
        The date range (as strings) to retrieve data for. Valid range: ('20220224', '20220915')
    flight_lines
        Shapefile for the SHIFT flightlines, Provided by default
    merge_strategy
        Describes how to handle overlapping flightlines within the area of interest. Default: 'first'. Provided strategies : ['first', 'last', 'max', 'min', 'sum', count]. Additionally, a custom merge strategy can be provide in the form of a Callable. The merge methods are based on rasterio.merge. See rasterio.merge for more information and examples of how to format the custom Callable.
    chunks
        Describes how to chunk the data along each dimension. Default: {'y': 100}
    res
        Output resolution. Default: None, uses the resolution from the first raster being merged
    crs
        Output crs. Default: None, uses the crs from the first raster being merged
    resampling
        Resampling method. Default: 'nearest'. See rasterios resampling methods for more information.

    Returns
    -------
    xr.Dataset
        If all data is successfully retrieved a single xr.Dataset will be retrieved
    tuple[xr.Dataset, dict]
        If some data is missing a GLT file and cannot be orthorectified, a tuple will be returned. The first value will be an xr.Dataset of the successfully retrieved data and the second value will be a dictionary organized by date of the data that could not be orthorectified. This data will be clipped.
    dict
        A dictionary organized by date of the data that could not be orthorectified. This data will be clipped.
    None
        Nothing will be returned if the data requested does not exist. For example certain dates are missing data.

    """
    # validate data argument
    if isinstance(datasets, list):
        for dataset in datasets:
            assert dataset in ['rfl', 'rdn', 'igm', 'obs']
    else:
        assert datasets in ['rfl', 'rdn', 'igm', 'obs']
        datasets = [datasets]
    
    # filter by date range
    intersecting = flight_lines[(flight_lines['date'] >= date_range[0]) & (flight_lines['date'] <= date_range[1])]
    
    assert len(intersecting) > 0, "Invalid date range"
    
    # find the spatial overlap
    temp = []
    for date in intersecting.date.unique():
        # first use the contains predicate to try and minimize flightline overlap
        temp_intersecting = intersecting.loc[intersecting.date==date]
        temp_intersecting = temp_intersecting.sjoin(gdf.to_crs(temp_intersecting.crs), how='inner', predicate='contains')
        if len(temp_intersecting) < 1:
            # if shapefile is not contained to one flightline then recover all overlapping
            temp_intersecting = intersecting.loc[intersecting.date==date]
            temp_intersecting = temp_intersecting.sjoin(gdf.to_crs(temp_intersecting.crs), how='inner')
        
        temp += [temp_intersecting]
    if len(temp) > 0:
        intersecting = pd.concat(temp)
    else:
        raise Exception("The provided shapefile does not overlap with the data")

    
    assert len(intersecting) > 0, "Shapefile does not overlap with any of the flight lines"
    
    # generate the file paths for the data
    file_paths = _prep_file_paths(intersecting, datasets)
    
    # load and merge datasets
    out_data, failed = _load_data(file_paths=file_paths, gdf=gdf, merge_strategy=merge_strategy, chunks=chunks, resampling=resampling, res=res, crs=crs)
    
    # sort data by date
    temp_keys = list(out_data.keys())
    temp_keys.sort()
    out_data = {k: out_data[k] for k in temp_keys}
    
    # merge datasets
    if len(out_data) > 1:

        dates, data = zip(*[(k,v) for k,v in out_data.items()])
        inds_to_drop = [i for i in range(len(data)) if isinstance(data[i], list)]
        dates = [d for i, d in enumerate(dates) if i not in inds_to_drop]
        data = [d for i, d in enumerate(data) if i not in inds_to_drop]
        if len (data) > 0:
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
    
    # Determine what to return, successfully loaded data, data that failed to orthorectify, or a combination of the two
    if isinstance(out_data, xr.Dataset) and len(failed) > 0:  
        return out_data, failed
    elif not isinstance(out_data, xr.Dataset) and len(failed) > 0:  
         return failed
    elif isinstance(out_data, xr.Dataset) and len(failed) == 0:
        return out_data
    else:
        print('The requested data does not exist! Verify your inputs (datasets, gdf, daterange)!')
        