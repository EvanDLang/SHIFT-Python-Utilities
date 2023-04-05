import numpy as np
import fsspec
from intake_xarray.raster import RasterIOSource
from intake.source.utils import reverse_formats
from intake_xarray.base import Schema
import glob
import os
import xarray as xr
import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class ShiftXarray(RasterIOSource):
    name = 'SHIFT_xarray'

    def __init__(self, urlpath, ortho=False, filter_bands=False, subset=None, chunks={"y": 1}, concat_dim='concat_dim',
                 xarray_kwargs=None, metadata=None, path_as_pattern=True,
                 storage_options=None, **kwargs):
      
        super().__init__(urlpath, chunks=chunks, concat_dim=concat_dim,
                 xarray_kwargs=xarray_kwargs, metadata=metadata, path_as_pattern=path_as_pattern,
                 storage_options=storage_options, **kwargs)
        
        self.ortho = ortho
        self.filter_bands = filter_bands
        self.subset = subset

    def _get_supporting_file(self, path, name):
        # parse filepath
        base, file = os.path.split(path)
        base, directory = os.path.split(base)
        key = {'L2a': 'reflectance', 'rdn': 'radiance', 'obs': 'obs', 'igm': 'igm'}
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
        else:
            from .bad_bands import bad_bands
            self._ds = self._ds.isel(wavelength=bad_bands)
        
    
    def _open_files(self, files):
        """
        Not currently supported
        """
        das = [xr.open_rasterio(f, chunks=self.chunks, **self._kwargs)
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
            self._ds = xr.open_rasterio(files, chunks=self.chunks,
                                        **self._kwargs).swap_dims({"band":"wavelength"}).drop("band").transpose('y', 'x', 'wavelength')
            
        
            if self.filter_bands is not False and (not "igm" in files and not "glt" in files and not "obs" in files):
                self._filter_bands()
            
            if self.subset:
                self._get_subset()
                
            if (not "igm" in files and not "glt" in files) and not self.ortho:
                igm_path, _ = self._get_supporting_file(files, 'igm')
            
                igm = xr.open_rasterio(igm_path, chunks=self.chunks, **self._kwargs).swap_dims({"band":"wavelength"}).drop("band").transpose('y', 'x', 'wavelength')

                if self.subset:
                    igm = igm.isel(self.subset)

                lon = igm.isel(wavelength=0).values
                lat = igm.isel(wavelength=1).values
                elev = igm.isel(wavelength=2).values

                self._ds = self._ds.assign_coords({'lat':(['y','x'], lat)})
                self._ds = self._ds.assign_coords({'lon':(['y','x'], lon)})
                self._ds = self._ds.assign_coords({'elevation':(['y','x'], elev)})
            
            if self.ortho and not "glt" in files:
                self._orthorectify()
            elif self.ortho and "glt" in files:
                print("glt file cannot be orthorectified")
            else:
                
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
        self._ds = self._ds.isel(self.subset)
    
    def _orthorectify(self):
        # get the path to the GLT file
        glt_path, data_var = self._get_supporting_file(self.urlpath, 'glt')
        igm_path, _ = self._get_supporting_file(self.urlpath, 'igm')
        
        igm = xr.open_rasterio(igm_path, chunks=self.chunks,**self._kwargs).swap_dims({"band":"wavelength"}).drop("band").transpose('y', 'x', 'wavelength')
        
        if self.subset:
            igm = igm.isel(self.subset)
        elev = igm.isel(wavelength=2).values

        
        # verify a glt file exists
        assert os.path.exists(glt_path), "No glt file exists, File cannot be orthorectified"
        
        # parse glt file
        loc = xr.open_rasterio(glt_path).transpose('y', 'x', 'band')
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
        

        GT = loc.transform
        
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
   
        metadata = {
            'description': self._ds.attrs['description'],
            'lines': loc.attrs['lines'],
            'samples': loc.attrs['samples'],
            'is_tiled ': self._ds.attrs['is_tiled'],           
            'bands': self._ds.attrs['bands'],
            'interleave': 'bil',
            'data_type': self._ds.attrs['data_type'],
            'file_type': self._ds.attrs['file_type'],
            'transform': loc.attrs['transform'],
            'res': loc.attrs['res'],
            'map info': loc.attrs['map_info'],
            'coordinate_system_string': loc.attrs['coordinate_system_string'],
            'nodatavals': self._ds.attrs['nodatavals'],
            'descriptions': wvl,
            'scales': self._ds.attrs['scales']
        }
        
        
        # create coords and data vars for output
        coords = {'lat':(['lat'], lat), 'lon':(['lon'], lon), 'wavelength':(['wavelength'], wvl)}
        data_vars = {data_var:(['lat','lon','wavelength'], out_ds), 'elevation':(['lat','lon'], out_elev)}
        
        
        # create the output xarray dataset
        self._ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=metadata)
        try:
            fwhm = self._ds.fwhm.values 
            self._ds = self._ds.assign_coords(fwhm=("wavelength", fwhm))
        except:
            pass # no fwhm coord
