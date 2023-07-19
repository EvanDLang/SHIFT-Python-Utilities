import numpy as np
import fsspec
from intake_xarray.raster import RasterIOSource
from intake.source.utils import reverse_formats
from intake_xarray.base import Schema
import glob
import os
import xarray as xr
import rioxarray as rxr
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import affine
import dask
import warnings
import rasterio
from rasterio.crs import CRS
import osgeo.osr as osr

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
dask.config.set({"array.slicing.split_large_chunks": False})


def get_epsg_code(description):
    ind = description.find("UTM")
    coord_system, _, zone, direction = description[57:].split(" ")
    direction = False if direction == 'North' else True
    epsg_code = 32600
    epsg_code += int(zone)
    if direction is True:
        epsg_code += 100

    return CRS.from_epsg(epsg_code)

def get_map_info(map_info, crs):
    zone = crs.to_wkt().split(',')[0]
    ind = zone.find('zone')
    zone = zone[ind + 5:-2]
    direction = zone[ind + 7:-1] 
    
    map_info = map_info.split(",")
    map_info =[m.strip() for m in map_info]
    map_info[7] = zone
    map_info[8] = direction
    map_info = ", ".join(map_info)

    return map_info

class ShiftXarray(RasterIOSource):
    name = 'SHIFT_xarray'

    def __init__(self, urlpath, ortho=False, filter_bands=False, subset=None, chunks='auto', concat_dim='concat_dim',
                 xarray_kwargs=None, metadata=None, path_as_pattern=True,
                 storage_options=None, **kwargs):
        """
        url_path: str
        ortho: bool
        filter_bands: bool, np.ndarray, list
        subset: dict, str, gpd.GeoDataFrame
        chunks: dict, str
        """
        
        # initialize super class
        super().__init__(urlpath, chunks=chunks, concat_dim=concat_dim,
                 xarray_kwargs=xarray_kwargs, metadata=metadata, path_as_pattern=path_as_pattern,
                 storage_options=storage_options, **kwargs)
        
        # set class variables
        self.ortho = ortho
        self.filter_bands = filter_bands
        self.subset = subset
        
        # only allow subsetting with a dataframe or file path if ortho is set to true
        if isinstance(self.subset, gpd.GeoDataFrame) or isinstance(self.subset, str):
            assert self.ortho, "ortho must be set to True in order to subset with a shapefile"
        
        self.geodf = None

    def _get_supporting_file(self, path, name):
        # parse filepath
        base, file = os.path.split(path)
        base, directory = os.path.split(base)
        key = {'L2a': 'reflectance', 'rdn': 'radiance', 'obs': 'obs', 'igm': 'igm', 'glt': 'glt'}
        data_var = key[directory]     
        
        # edit filepath to point to glt file
        if directory != "L2a":
            directory = name
        else:
            directory = os.path.join("L1", name)
        
        file_base = file.split('_')[0]
        file = file_base + '_' + name
        s_path = os.path.join(base, directory, file)

        return s_path, data_var
      
    def _filter_bands(self):
        if isinstance(self.filter_bands, np.ndarray) or isinstance(self.filter_bands, list):
            mask = np.ones((int(self._ds.shape[-1])))
            mask[self.filter_bands] = 0
            mask = mask.astype(bool)
            self._ds = self._ds.isel(wavelength=mask)
        elif isinstance(self.filter_bands, bool):
            from .bad_bands import bad_bands
            self._ds = self._ds.isel(wavelength=bad_bands)
        
    
    def _open_files(self, files):
        """
        Not currently supported
        """
        das = [rxr.open_rasterio(f, chunks=self.chunks, **self._kwargs)
               for f in files]
        out = xr.concat(das, dim=self.dim)

        coords = {}
        if self.pattern:
            coords = {
                k: xr.concat(
                    [xr.DataArray(
                        np.full(das[i].sizes.get(self.dim, 1), v),
                        dims=self.dim
                    ) for i, v in enumerate(values)], dim=self.dim)
                for k, values in reverse_formats(self.pattern, files).items()
            }

        return out.assign_coords(**coords).chunk(self.chunks)
    
    def _open_dataset(self):

        if self._can_be_local:
            files = fsspec.open_local(self.urlpath, **self.storage_options)
        else:
            # pass URLs to delegate remote opening to rasterio library
            files = self.urlpath
            #files = fsspec.open(self.urlpath, **self.storage_options).open()
        if isinstance(files, list):
            # self._ds = self._open_files(files)
            print('Multi-file loading is currently not supported')
        else:
            self._ds = rxr.open_rasterio(files, chunks=self.chunks, **self._kwargs).swap_dims({"band":"wavelength"}).drop_vars("band").transpose('y', 'x', 'wavelength')
            
            # only filter bands for reflectance and radiance files
            if self.filter_bands is not False and (not "igm" in files and not "glt" in files and not "obs" in files):
                self._filter_bands()
            
            # load igm data for obs, reflectance and radiance files
            if not 'igm' in files:
                igm_path, _ = self._get_supporting_file(files, 'igm')
                igm = rxr.open_rasterio(igm_path, chunks=self.chunks, **self._kwargs).swap_dims({"band":"wavelength"}).drop_vars("band").transpose('y', 'x', 'wavelength')
                self.crs = get_epsg_code(igm.attrs['description'])
                if not "glt" in files:
                    lon = igm.isel(wavelength=0).values
                    lat = igm.isel(wavelength=1).values
                    elev = igm.isel(wavelength=2).values

                    self._ds = self._ds.assign_coords({'lat':(['y','x'], lat)})
                    self._ds = self._ds.assign_coords({'lon':(['y','x'], lon)})
                    self._ds = self._ds.assign_coords({'elevation':(['y','x'], elev)})
            
            # if we are loading an igm file get the crs for orthorectifcation
            elif 'igm' in files:
                 self.crs = get_epsg_code(self._ds.attrs['description'])
   
            
            # subset the dataset
            if self.subset is not None:
                self._get_subset()
            # orthorectify the dataset
            if self.ortho and not "glt" in files:
                self._orthorectify()
            elif self.ortho and "glt" in files:
                raise Exception("glt file cannot be orthorectified")
            else:
                # if ortho is false convert the datarray to a dataset
                if 'L2a' in files:
                    self._ds = self._ds.to_dataset(name='reflectance')
                elif 'rdn' in files:
                    self._ds = self._ds.to_dataset(name='radiance')
                elif 'obs' in files:
                    self._ds = self._ds.to_dataset(dim='wavelength').rename({
                        0: 'path_length',
                        1: 'to_sensor_azimuth', 
                        2:'to_sensor_zenith',
                        3:'to_sun_azimuth',
                        4:'to_sun_zenith',
                        5:'solar_phase',
                        6:'slope',
                        7:'aspect',
                        8:'cosine',
                        9:'utc_time',
                        10:'earth_sun_distance',
                    })
                elif "igm" in files:
                    self._ds = self._ds.to_dataset(dim='wavelength').rename({0: 'easting', 1: 'northing', 2:'elevation'})
    
    def _get_schema(self):
        self.urlpath, *_ = self._get_cache(self.urlpath)
        
        if self._ds is None:
            self._open_dataset()
            try:
                metadata = {
                    'dims': dict(self._ds.dims),
                    'data_vars': {k: list(self._ds[k].coords)
                              for k in self._ds.data_vars.keys()},
                    'coords': tuple(self._ds.coords.keys()),
                }
            except:
                metadata = {
                    'dims': None,
                    'data_vars': None,
                    'coords': None,
            }

        self._schema = Schema(
                datashape=None,
                dtype=None,
                shape=None,
                npartitions=None,
                extra_metadata=metadata)

        return self._schema
    
    def _get_subset(self):
        
        # check if the subset passed is a gdf or path to a gdf
        if isinstance(self.subset, gpd.GeoDataFrame):
            self.geodf = self.subset
        elif isinstance(self.subset, str):
            if os.path.exists(self.subset):
                self.geodf = gpd.read_file(self.subset)
            else:
                raise Exception("The provided Shapefile filepath does not exist!")
        
        # if a gdf is passed use the bounds of the shapefile as a bounding box
        if self.geodf is not None:
            assert not 'glt' in self.urlpath, "GLT files cannot be subset by shapefile"
            
            if self.crs.to_epsg() > 32000 and self.geodf.crs.to_epsg() < 32000:
                    self.geodf =  self.geodf.to_crs(self.geodf.estimate_utm_crs())
            if self.crs.to_epsg() != self.geodf.crs.to_epsg():
                print(f"The shapefile CRS ({self.geodf.crs.to_epsg()}) does not match the dataset ({self.crs.to_epsg()})! Attempting to convert the shapefile to the dataset's CRS.")
                self.geodf = self.geodf.to_crs(self.crs)
            
            lat_max = self.geodf.bounds['maxy'].max()
            lat_min = self.geodf.bounds['miny'].min()
            lon_max = self.geodf.bounds['maxx'].max()
            lon_min = self.geodf.bounds['minx'].min()
            self.subset ={'lat':(lat_min, lat_max), "lon": (lon_min, lon_max)}
        
        
        if "lat" in self.subset.keys() or "lon" in self.subset.keys():
            
            # get the minimum and maximum lat and lon values
            lat_min, lat_max = min(self.subset['lat']), max(self.subset['lat'])
            lon_min, lon_max = min(self.subset['lon']), max(self.subset['lon'])
            
            # create a mask using the min and max values
            if "igm" in self.urlpath:
                lon_mask = (self._ds.isel(wavelength=0)> lon_min) & (self._ds.isel(wavelength=0) < lon_max)
                lat_mask = (self._ds.isel(wavelength=1) > lat_min) & (self._ds.isel(wavelength=1) < lat_max)
            elif "glt" in self.urlpath:
                lon_mask = (self._ds.coords['x'] > lon_min) & (self._ds.coords['x'] < lon_max)
                lat_mask = (self._ds.coords['y'] > lat_min) & (self._ds.coords['y'] < lat_max)
            else:
                lon_mask = (self._ds.coords['lon'] > lon_min) & (self._ds.coords['lon'] < lon_max)
                lat_mask = (self._ds.coords['lat'] > lat_min) & (self._ds.coords['lat'] < lat_max)
            
            mask = (lon_mask) & (lat_mask)
            
            # make sure there are valid values in the mask
            assert np.count_nonzero(mask), "invalid lat/lon subset values, make sure you are querying the correct image"
            
            # use the mask to retrieve indicies
            inds = np.argwhere(mask.values)
            y_min, y_max, x_min, x_max = inds[:, 0].min(), inds[:, 0].max(), inds[:, 1].min(), inds[:, 1].max()
            self.subset = {'y':slice(y_min, y_max), "x": slice(x_min, x_max)}
        
        # subset the dataset using the indicies
        self._ds = self._ds.isel(self.subset)
    
    def _orthorectify(self):
        # get the path to the GLT file
        glt_path, data_var = self._get_supporting_file(self.urlpath, 'glt')
            
        
        if 'igm' in self.urlpath:
             elev = self._ds.isel(wavelength=2).values
        else:
            elev = self._ds.coords['elevation'].values
        # verify a glt file exists
        assert os.path.exists(glt_path), "No glt file exists, File cannot be orthorectified"
        
        # parse glt file
        loc = rxr.open_rasterio(glt_path).transpose('y', 'x', 'band')
        glt_array = loc.values.astype(int)
        
        # retrive data
        ds_array = self._ds.values
        ds_x, ds_y = self._ds.x.values.astype(int), self._ds.y.values.astype(int)

        
        # create a mask that filters o ut nodata values
        valid_glt = np.all(glt_array != -9999, axis=-1)
        # subtract 1 from all indicies so they are 0 based
        glt_array[valid_glt] -= 1 
        
        
        # replace indicies outside of the sub-selection to nodata values for x and y
        temp = glt_array[valid_glt, 1]
        temp[np.logical_or((glt_array[valid_glt, 1] < min(ds_y)), (glt_array[valid_glt, 1] > max(ds_y)))] = -9999
        glt_array[valid_glt, 1] = temp

        temp = glt_array[valid_glt, 0]
        temp[np.logical_or((glt_array[valid_glt, 0] < min(ds_x)), (glt_array[valid_glt, 0] > max(ds_x)))] = -9999
        glt_array[valid_glt, 0] = temp
        
        # update the valid glt mask to capture new no data values
        valid_glt = np.all(glt_array != -9999, axis=-1)
        # update the indicies to reflect the subset range
        glt_array[valid_glt, 1] -= min(ds_y)
        glt_array[valid_glt, 0] -= min(ds_x)
        
        # Loop twice, on the first loop, determine the maximum length, min_y, max_y and the number of x's
        # On the second loop use the info from the first loop to create a outdata set of the appropriate size
        for get_shapes in [True, False]:
            if get_shapes:
                xs = []
                min_y = 99999
                max_y = -99999
                max_len = -99999
            else:
                window = (min_y, max_y)
                out_ds = np.zeros((len(xs), window[1] - window[0] + 1, ds_array.shape[-1]), dtype=np.float32)  + np.nan
                out_elev = np.zeros((len(xs), window[1] - window[0] + 1), dtype=np.float32)  + np.nan
            for x in range(valid_glt.shape[0]):
                if valid_glt[x,:].sum() != 0:
                    y = valid_glt[x,:]

                    if get_shapes:
                        if len(ds_array[glt_array[x, y, 1], glt_array[x, y, 0], :]) > max_len:
                            max_len = len(ds_array[glt_array[x, y, 1], glt_array[x, y, 0], :])

                        if min(np.nonzero(y)[0]) < min_y:
                            min_y = min(np.nonzero(y)[0])

                        if max(np.nonzero(y)[0]) > max_y:
                            max_y = max(np.nonzero(y)[0])

                        xs += [x]
                    else:
                        out_ds[x - min(xs) ,y[window[0]:window[1] + 1], :] = ds_array[glt_array[x, y, 1], glt_array[x, y, 0], :]
                        out_elev[x - min(xs) ,y[window[0]:window[1] + 1]] = elev[glt_array[x, y, 1], glt_array[x, y, 0]]
        
        # get the transform and create the coords
        GT = loc.rio.transform()
        
        dim_x = loc.x.shape[0]
        dim_y = loc.y.shape[0]
        lon = np.zeros(dim_x)
        lat = np.zeros(dim_y)
        
        for x in np.arange(dim_x):
            x_geo = GT[2] + x * GT[0]
            lon[x] = x_geo
        for y in np.arange(dim_y):
            y_geo = GT[5] + y * GT[4]
            lat[y] = y_geo
            
        lon = lon[window[0]: window[1] + 1]
        lat = lat[min(xs) : max(xs) + 1]
        
        # Set up the metadata for the output
        wvl = self._ds.wavelength.values
       
        map_info = get_map_info(loc.attrs['map_info'], self.crs)
        
        metadata = {
            'description': self._ds.attrs['description'],        
            'bands': self._ds.attrs['bands'],
            'interleave': 'bil',
            'data_type': self._ds.attrs['data_type'],
            'file_type': self._ds.attrs['file_type'],
            'map_info': map_info,
            'coordinate_system_string': self.crs.to_wkt(),
            'wavelength': wvl,
        }
        
        
       
        if data_var == 'obs':
            coords = {'lat':(['lat'], lat), 'lon':(['lon'], lon)}
            data_vars = {
                'path_length':(['lat','lon'], out_ds[:, :, 0]),
                'to_sensor_azimuth':(['lat','lon'], out_ds[:, :, 1]),
                'to_sensor_zenith':(['lat','lon'], out_ds[:, :, 2]),
                'to_sun_azimuth':(['lat','lon'], out_ds[:, :, 3]),
                'to_sun_zenith':(['lat','lon'], out_ds[:, :, 4]),
                'solar_phase':(['lat','lon'], out_ds[:, :, 5]),
                'slope':(['lat','lon'], out_ds[:, :, 6]),
                'aspect':(['lat','lon'], out_ds[:, :, 7]),
                'cosine':(['lat','lon'], out_ds[:, :, 8]),
                'utc_time':(['lat','lon'], out_ds[:, :, 9]),
                'earth_sun_distance':(['lat','lon'], out_ds[:, :, 10]),
                'elevation':(['lat','lon'], out_elev)
            }
        elif data_var == 'igm':
            coords = {'lat':(['lat'], lat), 'lon':(['lon'], lon)}
            data_vars = {'elevation':(['lat','lon'], out_elev)}
        else:
             # create coords and data vars for output
            coords = {'lat':(['lat'], lat), 'lon':(['lon'], lon), 'wavelength':(['wavelength'], wvl)}
            data_vars = {data_var:(['lat','lon','wavelength'], out_ds), 'elevation':(['lat','lon'], out_elev)}
        
        
        # create the output xarray dataset
        self._ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=metadata)
        self._ds.rio.write_crs(self.crs, inplace=True)
        self._ds.rio.write_transform(GT, inplace=True)
        self._ds.rio.set_spatial_dims('lon', 'lat', inplace=True)
        try:
            fwhm = self._ds.fwhm.values 
            self._ds = self._ds.assign_coords(fwhm=("wavelength", fwhm))
        except:
            pass # no fwhm coord
        
        # if a geodf was passed subset it!
        if self.geodf is not None:
            self._ds = self._ds.rio.clip(self.geodf.geometry.values)
