"""
WIP

- support 4D???
"""
import xarray as xr
import hvplot.xarray
import numpy as np
import inspect

def normalize_band(band):
    mask = ~np.isnan(band)
    band_min, band_max = (band[mask].min(), band[mask].max())
    band[mask] = ((band[mask] - band_min)/((band_max - band_min)))
    return band

def brighten(band, alpha=0.13, beta=0):
    return np.clip(alpha*band+beta, 0,255)

def gammacorr(band, gamma=2):
    return np.power(band, 1/gamma)

def plot_rgb(data, normalize=False, **kwargs):
    
    assert isinstance(data, xr.DataArray), "Input data must be an Xarray DataArary"
    assert 'y' in data.coords and 'x' in data.coords, "y and x must be used as spatial coords"
    assert len(data.dims) == 3, "plot_rgb only supports 3 dimensions currently"
    
    ds_rgb = xr.DataArray(data, dims=('bands', 'y', 'x'), coords=({'x':data.x.values, 'y':data.y.values, 'bands': [0,1,2]})).to_dataset(name='rgb')
    
    brighten_args = [p for p in inspect.signature(brighten).parameters if p != 'band']
    gamma_args = [p for p in inspect.signature(brighten).parameters if p != 'band']
    
    b_kwargs = {}
    gamma_kwargs = {}
    
    for k, v in kwargs.items():
        if k in brighten_args:
            b_kwargs[k] = v 
        elif gamma_args:
             gamma_kwargs[k] = v
    
    if normalize:
        ds_rgb = ds_rgb.assign(rgb=(('bands', 'y', 'x'), 
                  xr.apply_ufunc(normalize_band, ds_rgb.rgb.values, dask='allowed', vectorize=True)))
    
    if len(b_kwargs) > 0:
        ds_rgb = ds_rgb.assign(rgb=(('bands', 'y', 'x'), 
                  xr.apply_ufunc(brighten, ds_rgb.rgb.values, *tuple(b_kwargs.values()),  dask='allowed', vectorize=True)))
        
    
    if len(gamma_kwargs) > 0:

        ds_rgb = ds_rgb.assign(rgb=(('bands', 'y', 'x'), 
                  xr.apply_ufunc(gammacorr, ds_rgb.rgb.values, *tuple(gamma_kwargs.values()), dask='allowed', vectorize=True)))
       
    return ds_rgb.hvplot.rgb(x='x', y='y', bands='bands', aspect='equal').opts(tools=["hover"])