import numpy as np
import fsspec
from intake_xarray.raster import RasterIOSource
from intake.source.utils import reverse_formats
from intake_xarray.base import Schema
import glob
import os
import xarray as xr

class ShiftXarray(RasterIOSource):
    name = 'SHIFT_xarray'

    def __init__(self, urlpath, ortho=False, chunks=None, concat_dim='concat_dim',
                 xarray_kwargs=None, metadata=None, path_as_pattern=True,
                 storage_options=None, **kwargs):
      
        super().__init__(urlpath, chunks=chunks, concat_dim=concat_dim,
                 xarray_kwargs=xarray_kwargs, metadata=metadata, path_as_pattern=path_as_pattern,
                 storage_options=storage_options, **kwargs)
        
        self.ortho = ortho
        
        if ortho:
            import warnings
            import rasterio
            warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    def _open_files(self, files):
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
        import xarray as xr
        if self._can_be_local:
            files = fsspec.open_local(self.urlpath, **self.storage_options)
        else:
            # pass URLs to delegate remote opening to rasterio library
            files = self.urlpath
            #files = fsspec.open(self.urlpath, **self.storage_options).open()
        if isinstance(files, list):
            self._ds = self._open_files(files)
        else:
            self._ds = xr.open_rasterio(files, chunks=self.chunks,
                                        **self._kwargs).swap_dims({"band":"wavelength"}).drop("band").transpose('y', 'x', 'wavelength')
            if self.ortho:
                self._orthorectify()
    
    
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
    
    def _orthorectify(self):
        # parse filepath
        base, file = os.path.split(self.urlpath)
        base, directory = os.path.split(base)
        key = {'L2a': 'reflectance', 'rdn': 'radiance'}
        
        # currently only reflectance and radiance files have been tested
        assert directory in key, "Currenly only reflectance and radiance files are supported"
        data_var = key[directory]     
        
        # edit filepath to point to glt file
        if directory != "L2a":
            directory = "glt"
        else:
            directory = os.path.join("L1", 'glt')
        
        file_base = file.split('_')[0]
        file = file_base + "_glt"
        glt_path = os.path.join(base, directory, file)
        
        # verify a glt file exists
        assert os.path.exists(glt_path), "No glt file exists, File cannot be orthorectified"
        
        #parse glt file
        loc = xr.open_rasterio(glt_path).transpose('y', 'x', 'band')
        glt_array = loc.values.astype(int)
        
        # retrive data
        ds_array = self._ds.values
        
        # create an output array
        out_ds = np.zeros((loc.shape[0], loc.shape[1], ds_array.shape[-1]), dtype=np.float32) + np.nan
        
        # create a mask that filters o ut nodata values
        valid_glt = np.all(glt_array != -9999, axis=-1)
        # subtract 1 from all indicies so they are 0 based
        glt_array[valid_glt] -= 1 
        
        # loop through the valid glt use indicides to transfer data to the correct position in the output array
        for x in range(valid_glt.shape[0]):
            if valid_glt[x,:].sum() != 0:
                y = valid_glt[x,:]
                out_ds[x, y, :] = ds_array[glt_array[x, y, 1], glt_array[x, y, 0], :]
 
        # Set up the metadata for the output
        wvl = self._ds.wavelength.values
        fwhm = self._ds.fwhm.values 
        metadata = {
            'description': self._ds.attrs['description'],
            'lines': loc.attrs['lines'],
            'samples': loc.attrs['samples'],
            'is_tiled ': self._ds.attrs['is_tiled'],           
            'bands': self._ds.attrs['bands'],
            'interleave': self._ds.attrs['interleave'],
            'data_type': self._ds.attrs['data_type'],
            'file_type': self._ds.attrs['file_type'],
            'transform': loc.attrs['transform'],
            'res': loc.attrs['res'],
            'coordinate_system_string': loc.attrs['coordinate_system_string'],
            'nodatavals': self._ds.attrs['nodatavals'],
            'descriptions': wvl,
            'scales': self._ds.attrs['scales']
        }
        
        
        # create coords and data vars for output
        coords = {'y':(['y'], loc.y.data), 'x':(['x'], loc.x.data), 'wavelength':(['wavelength'], wvl)}
        data_vars = {data_var:(['y','x','wavelength'], out_ds)}
        
        # create the output xarray dataset
        self._ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=metadata)
        self._ds = self._ds.assign_coords(fwhm=("wavelength", fwhm))
