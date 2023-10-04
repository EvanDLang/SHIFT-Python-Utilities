import shift_python_utilities.shift_xarray.xr
import rioxarray as rxr
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import dask.array as da
import xarray as xr
import numpy as np
import pytest
import copy


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
def iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = da.from_array(X, chunks=(50))
    return xr.DataArray(X, dims=('x', 'y'), coords={'x':np.arange(150), 'y':np.arange(4)})

@pytest.fixture
def iris_nans(iris):
    iris_nans = copy.deepcopy(iris)
    iris_nans[0,3] = np.nan
    iris_nans[3,0] = np.nan
    return iris_nans


def test_pcas(iris, iris_nans):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(iris.compute()).T
    sk = iris.SHIFT.init_ipca(feature_dim=1, base='sklearn').dask_fit().dask_transform().compute().values
    dk = iris.SHIFT.init_ipca(feature_dim=1, base='dask').dask_fit().dask_transform().compute().values
    assert np.abs(np.abs(sk) - np.abs(X_pca)).mean() < .0001
    assert np.abs(np.abs(dk) - np.abs(X_pca)).mean() < .0001
    assert np.abs(np.abs(sk) - np.abs(dk)).mean() < .0001
    
    sk = iris_nans.SHIFT.init_ipca(feature_dim=1, base='sklearn').dask_fit().dask_transform().compute().values
    dk = iris_nans.SHIFT.init_ipca(feature_dim=1, base='dask').dask_fit().dask_transform().compute().values
    sk = sk[~np.isnan(sk)]
    dk = dk[~np.isnan(dk)]
    assert np.abs(np.abs(sk) - np.abs(dk)).mean() < .0001
    
    
    
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
def glt_shape(ds_subset, date, time,):
    _, subset = ds_subset
    shape = _subset_glt(_load_glt(date,time), subset['x'], subset['y']).shape
    return shape[1], shape[2]

def test_sk_and_dask_pcas(ds_subset, glt_shape):
    ds, _ = ds_subset
    sk = ds.SHIFT.init_ipca(feature_dim=0, base='sklearn').dask_fit().dask_transform()
    dsk = ds.SHIFT.init_ipca(feature_dim=0, base='sklearn').dask_fit().dask_transform()
    assert sk.shape[0] == 2 and dsk.shape[0] == 2
    assert sk.shape == dsk.shape
    
    assert np.abs(np.abs(sk.compute().values) - np.abs(dsk.compute().values)).mean() < .01

    #integration level test
    sk = sk.SHIFT.orthorectify()
    dsk = dsk.SHIFT.orthorectify()
    assert (sk.shape[1], sk.shape[2]) == glt_shape
    assert (dsk.shape[1], dsk.shape[2]) == glt_shape