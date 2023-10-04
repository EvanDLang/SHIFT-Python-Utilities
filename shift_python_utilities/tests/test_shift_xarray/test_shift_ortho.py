import pytest
import shift_python_utilities.shift_xarray.xr
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rxr
import numpy as np

def _load_rfl(date, time, chunks):
    return rxr.open_rasterio(f"/efs/efs-data-curated/v1/{date}/L2a/ang{date}t{time}_rfl", chunks=(chunks))

def _load_igm(date, time, chunks=None):
    if chunks is None:
        return rxr.open_rasterio(f"/efs/efs-data-curated/v1/{date}/L1/igm/ang{date}t{time}_igm")
    else:
        return rxr.open_rasterio(f"/efs/efs-data-curated/v1/{date}/L1/igm/ang{date}t{time}_igm", chunks=(chunks))

def _load_glt(date, time):
    return rxr.open_rasterio(f"/efs/efs-data-curated/v1/{date}/L1/glt/ang{date}t{time}_glt")


def _subset_data(ds, x_sub, y_sub,z_sub):
    return ds.isel(x=slice(x_sub[0], x_sub[1]), y=slice(y_sub[0], y_sub[1]), band=slice(z_sub[0], z_sub[1]))


def _subset_glt(glt, x_sub, y_sub):
    y_inds = xr.DataArray(list(range(y_sub[0], y_sub[1])))
    x_inds = xr.DataArray(list(range(x_sub[0], x_sub[1])))
    return glt.where(glt[1, :, :].compute().isin(y_inds) & (glt[0, :, :].compute().isin(x_inds)), drop=True)

def _get_elevation(date, time, subset, chunks):
    igm = _load_igm(date, time, chunks).isel(x=slice(*subset['x']), y=slice(*subset['y']))
    elev = igm.isel(band=2)
    return elev

def _subset_lat_lon(ds, date, time, lat_sub, lon_sub):
    igm = _load_igm(date, time)
    lon = igm.isel(band=0).values
    lat = igm.isel(band=1).values
    lat_min, lat_max = min(lat_sub), max(lat_sub)
    lon_min, lon_max = min(lon_sub), max(lon_sub)
    ds = ds.assign_coords({'lat':(['y','x'], lat)})
    ds = ds.assign_coords({'lon':(['y','x'], lon)})
    lon_mask = (ds.coords['lon'] > lon_min) & (ds.coords['lon'] < lon_max)
    lat_mask = (ds.coords['lat'] > lat_min) & (ds.coords['lat'] < lat_max)
    mask = (lon_mask) & (lat_mask)
    inds = np.argwhere(mask.values)
    y_min, y_max, x_min, x_max = inds[:, 0].min(), inds[:, 0].max(), inds[:, 1].min(), inds[:, 1].max()
    subset = {'y':slice(y_min, y_max), "x": slice(x_min, x_max)}
    subset = {'y':(y_min, y_max), "x": (x_min, x_max)}
    return subset


@pytest.fixture
def date():
    return 20220224

@pytest.fixture
def time():
    return 200332


@pytest.fixture
def ds_subset(date, time):
    eastings = np.array([228610.68861488, 237298.11871802])
    northings = np.array([3812959.0852389 , 3810526.08057343])
    subset = {'lat':northings, "lon": eastings}
    chunks = {'band': 100, 'y': 200, 'x': 200}
    ds = _load_rfl(date, time,chunks)
    subset = _subset_lat_lon(ds, date, time, subset['lat'], subset['lon'])
    return ds.isel(x=slice(*subset['x']), y=slice(*subset['y'])), subset

@pytest.fixture
def glt_shape(ds_subset, date, time):
    _, subset = ds_subset
    shape = _subset_glt(_load_glt(date,time), subset['x'], subset['y']).shape
    return shape[1], shape[2]

@pytest.fixture
def elev(ds_subset, date, time):
    ds, subset = ds_subset
    return _get_elevation(date, time, subset, chunks={k: v for k,v in {'y': 200, 'x': 200}.items() if k == 'x' or k == 'y'})


@pytest.fixture
def glt_url(date, time):
    return f"/efs/efs-data-curated/v1/{date}/L1/glt/ang{date}t{time}_glt"

def test_data_array_2d(elev, glt_shape, glt_url):
    assert elev.SHIFT.orthorectify().shape == glt_shape
    assert elev.SHIFT.orthorectify(url=glt_url).shape ==  glt_shape
    

def test_data_array_3d(ds_subset, glt_shape):
    ds, _ = ds_subset
    ortho_shape = ds.SHIFT.orthorectify().shape
    assert (ortho_shape[1], ortho_shape[2]) == glt_shape
    
def test_data_array_4d(ds_subset, glt_shape):
    ds, _ = ds_subset
    ds = xr.concat([ds, ds], dim='time').assign_coords({'time': ('time', [0, 1])})
    ortho_shape = ds.SHIFT.orthorectify().shape
    assert (ortho_shape[2], ortho_shape[3]) == glt_shape


def test_data_set(ds_subset, elev, glt_url, glt_shape):
    ds, _ = ds_subset
    ds = ds.to_dataset(name='reflectance')
    ds = ds.assign({'elevation': (('y', 'x'), elev.data)})
    
    # no url or source should raise an exception
    with pytest.raises(Exception):
        ds.SHIFT.orthorectify()
    
    ortho_shape = ds.SHIFT.orthorectify(url=glt_url).reflectance.shape
    assert (ortho_shape[1], ortho_shape[2]) == glt_shape
    
    ortho_shape = ds.SHIFT.orthorectify(url=[glt_url, glt_url]).reflectance.shape
    assert (ortho_shape[1], ortho_shape[2]) == glt_shape
    

    
def test_result(date, time):
 
    ds = _load_igm(date, time, chunks={'y':100})
    ds = ds.rio.write_nodata(np.nan)
    
    ds2 = ds.SHIFT.orthorectify().compute()
    
    fig,ax=plt.subplots(1,1, figsize=(20, 6))
    cp = ax.contourf(ds2.x.values, ds2.y.values, ds2.isel(band=2).values, levels=15)
    plt.close()
    
    fig,ax=plt.subplots(1,1, figsize=(20, 6))
    plt.close()
    cp2 = ax.contourf(ds.isel(band=0).values, ds.isel(band=1).values, ds.isel(band=2).values, levels=15)

    assert np.all(cp.cvalues == cp2.cvalues)
