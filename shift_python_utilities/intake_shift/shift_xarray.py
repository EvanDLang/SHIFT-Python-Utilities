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

    def __init__(self, urlpath, ortho=False, filter_bands=False, subset=None, chunks={'y': 1}, concat_dim='concat_dim',
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
    
    def _subset_glt(self, glt):

        y_inds = xr.DataArray(np.arange(self.subset['y'].start, self.subset['y'].stop, 1))
        x_inds = xr.DataArray(np.arange(self.subset['x'].start, self.subset['x'].stop, 1))
        
        return glt.where(glt[:, :, 1].compute().isin(y_inds) & (glt[:, :, 0].compute().isin(x_inds)), drop=True)
        
    
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
        glt = rxr.open_rasterio(glt_path).transpose('y', 'x', 'band')
        # glt_array = glt.values.astype(int)

        if self.subset is not None:
            glt = self._subset_glt(glt)
        glt_array = glt.fillna(-9999).astype(int).values
        
        # create a mask that filters o ut nodata values
        valid_glt = np.all(glt_array != -9999, axis=-1)
        
        # create output datasets
        out_ds = np.zeros((glt_array.shape[0], glt_array.shape[1], self._ds.shape[-1]), dtype=np.float32)  + np.nan
        out_elev = np.zeros((glt_array.shape[0], glt_array.shape[1]), dtype=np.float32)  + np.nan
           
        # load data into memory
        ds_array = self._ds.values
        
        # adjust indicies based on the subset
        ds_x, ds_y = self._ds.x.values.astype(int), self._ds.y.values.astype(int)
        glt_array[valid_glt, 1] -= min(ds_y)
        glt_array[valid_glt, 0] -= min(ds_x)

        # broadcast values
        for x in range(valid_glt.shape[0]):
            if valid_glt[x,:].sum() != 0:
                y = valid_glt[x,:]
                out_ds[x, y, :] = ds_array[glt_array[x, y, 1], glt_array[x, y, 0], :]
                out_elev[x, y] = elev[glt_array[x, y, 1], glt_array[x, y, 0]]
        
        # get the transform and create the coords
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
        
        # Set up the metadata for the output
        wvl = self._ds.wavelength.values
       
        map_info = get_map_info(glt.attrs['map_info'], self.crs)
        
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
            coords = {'y':(['y'], lat), 'x':(['x'], lon)}
            data_vars = {
                'path_length':(['y','x'], out_ds[:, :, 0]),
                'to_sensor_azimuth':(['y','x'], out_ds[:, :, 1]),
                'to_sensor_zenith':(['y','x'], out_ds[:, :, 2]),
                'to_sun_azimuth':(['y','x'], out_ds[:, :, 3]),
                'to_sun_zenith':(['y','x'], out_ds[:, :, 4]),
                'solar_phase':(['y','x'], out_ds[:, :, 5]),
                'slope':(['y','x'], out_ds[:, :, 6]),
                'aspect':(['y','x'], out_ds[:, :, 7]),
                'cosine':(['y','x'], out_ds[:, :, 8]),
                'utc_time':(['y','x'], out_ds[:, :, 9]),
                'earth_sun_distance':(['y','x'], out_ds[:, :, 10]),
                'elevation':(['y','x'], out_elev)
            }
        elif data_var == 'igm':
            coords = {'y':(['y'], lat), 'x':(['x'], lon)}
            data_vars = {'elevation':(['y','x'], out_elev)}
        else:
            # create coords and data vars for output
            coords = {'y':(['y'], lat), 'x':(['x'], lon), 'wavelength':(['wavelength'], wvl)}
            data_vars = {data_var:(['y','x', 'wavelength'], out_ds), 'elevation':(['y','x'], out_elev)}
        
        # create the output xarray dataset
        self._ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=metadata)
        self._ds.rio.write_crs(self.crs, inplace=True)
        self._ds.rio.write_transform(GT, inplace=True)
        self._ds.rio.set_spatial_dims('x', 'y', inplace=True)
        try:
            fwhm = self._ds.fwhm.values 
            self._ds = self._ds.assign_coords(fwhm=("wavelength", fwhm))
        except:
            pass # no fwhm coord
        
        # if a geodf was passed subset it!
        if self.geodf is not None:
            self._ds = self._ds.rio.clip(self.geodf.geometry.values)
