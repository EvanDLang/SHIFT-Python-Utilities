# required packages
import numpy as np
from uuid import uuid4
import dask.array as da
import dask
from dask.highlevelgraph import HighLevelGraph
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA as IPCA_SK
from dask_ml.decomposition import IncrementalPCA as IPCA_Dask
import xarray as xr
import rioxarray
import copy as cpy
import threading
from dask.distributed import Lock
from joblib import Parallel, delayed


from ._utils import _nested_iterator

def _create_windows(window_size, number_of_values):
    windows = []
    num_windows = number_of_values // window_size
    remainder = number_of_values % window_size
    for i in range(num_windows + 1):
        
        if i == num_windows:
            window = (i * window_size, i * window_size + remainder)
        else:
             window = (i * window_size, (i + 1) * window_size)
        windows += [window]
    
    return windows


def _get_overlap(dst_idx, chunks):
    src_shape = tuple(map(len, chunks))
    dst_idx = list(dst_idx)
    blocks = []
    for i in range(src_shape[1]):
        dst_idx[1] = i
        blocks += [tuple(dst_idx)]
    return blocks

class Dask_IPCA_SK(IPCA_SK):
    def __init__(self, src, feature_dim=0, n_components=2, whiten=False, copy=True, batch_size=None):
        super().__init__(n_components=n_components, whiten=whiten, copy=copy, batch_size=batch_size)
        
        dim_map = {dim: i for i, dim in enumerate(src.dims)}
        
        if isinstance(feature_dim, str):
            self.feature_dim = dim_map[feature_dim]
        else:
            self.feature_dim = feature_dim
            
        feat_shape = src.shape[self.feature_dim]
        temp = {k:v for k, v in dim_map.items() if v!= self.feature_dim}
        self.src = src.stack(combined=(list(temp.keys())))
        if self.src.shape[-1] != feat_shape:
            self.src = self.src.T
        
        print(self.src.shape)

        
    def _do_chunked_transform(self, *data):

        data = np.hstack(data)
        out = np.zeros((len(data), self.n_components)) + np.nan
   
        if np.sum(~np.isnan(data).any(axis=1)) != 0:
            temp = data[~np.isnan(data).any(axis=1), :]
            X_ipca = self.transform(temp)
            
            out[~np.isnan(data).any(axis=1), :] = X_ipca

        return out
     

    def dask_fit(self, window_size=None):

        if window_size is None:
            window_size = 1024**2 // self.src[0,0].nbytes // self.src.shape[1] * 1000

        windows = _create_windows(window_size, self.src.shape[0])   
        blocks = [self.src[w1:w2, :] for w1, w2 in windows]
        with tqdm(total=len(blocks), desc='Fitting') as pbar:
            for block in blocks:
                temp = block.compute()
                if np.sum(~np.isnan(temp).any(axis=1)) != 0:
                    temp = temp[~np.isnan(temp).any(axis=1), :]
                    self.partial_fit(temp)
                    del temp
                pbar.update(1)
        
        return self
       
        
    def dask_transform(self):
        src_block_keys = self.src.data.__dask_keys__()
    
        dst_shape = list(self.src.data.shape)
        dst_shape[1] = self.n_components
        dst_shape = tuple(dst_shape)

     
        dst_chunks = list(self.src.data.chunks)
        dst_chunks[1] = (dst_shape[1],)
        dst_chunks = tuple(dst_chunks)

        shape_in_blocks = tuple(map(len, dst_chunks))

        name = 'ipca_transform'
        tk = uuid4().hex
        name = f"{name}-{tk}"

        dsk = {}
        for idx in np.ndindex(shape_in_blocks):
            k = (name, *idx)
            blocks = _get_overlap(idx, self.src.data.chunks)
            block_deps = [_nested_iterator(src_block_keys, b) for b in blocks]
            dsk[k] = (self._do_chunked_transform), *tuple(block_deps)

        dsk = HighLevelGraph.from_collections(name, dsk, dependencies=(self.src,))
        dsk = da.Array(dsk, name, chunks=dst_chunks, dtype=self.src.dtype, shape=dst_shape)

        dim_map = {dim: i for i, dim in enumerate(self.src.dims)}
        new_dims = list(dim_map.keys())
        feat_dim = new_dims.pop(1)
        new_dims.insert(1, 'pca_features')
        coords = {k: v for k,v in self.src.coords.items() if feat_dim not in v.dims}
        coords['pca_features'] = xr.DataArray(np.arange(self.n_components), coords={'pca_features': np.arange(self.n_components)}, dims='pca_features')
       
        out = xr.DataArray(dsk, coords=coords, dims=new_dims).unstack('combined')
        

        out.encoding = self.src.encoding
        
        return out

    
class Dask_IPCA_DS(IPCA_Dask):
    def __init__(self, src, feature_dim=0, n_components=2, whiten=False, copy=True, batch_size=None, svd_solver='auto', iterated_power=0, random_state=None):
        super().__init__(n_components=n_components, whiten=whiten, copy=copy, batch_size=batch_size, svd_solver=svd_solver, iterated_power=iterated_power, random_state=random_state)
        
        
        
        dim_map = {dim: i for i, dim in enumerate(src.dims)}
        
        if isinstance(feature_dim, str):
            self.feature_dim = dim_map[feature_dim]
        else:
            self.feature_dim = feature_dim
        
        feat_shape = src.shape[self.feature_dim]
        temp = {k:v for k, v in dim_map.items() if v!= self.feature_dim}
        self.src = src.stack(combined=(list(temp.keys())))
        if self.src.shape[-1] != feat_shape:
            self.src = self.src.T        
    
    def dask_fit(self, window_size=None):
        if window_size is None:
            window_size = 1024**2 // self.src.data[0,0].nbytes // self.src.data.shape[1] * 1000

        windows = _create_windows(window_size, self.src.shape[0])   
        blocks = [self.src.data[w1:w2, :] for w1, w2 in windows]
        with tqdm(total=len(blocks), desc='Fitting') as pbar:
            for block in blocks:
                self.partial_fit(block[~da.isnan(block).any(axis=1), :].compute_chunk_sizes())
                del block
                pbar.update(1)
        
        return self
        
    def dask_transform(self):
        X_ipca = super().transform(self.src.data)
        
        dim_map = {dim: i for i, dim in enumerate(self.src.dims)}
        new_dims = list(dim_map.keys())
        feat_dim = new_dims.pop(1)
        new_dims.insert(1, 'pca_features')
        coords = {k: v for k,v in self.src.coords.items() if feat_dim not in v.dims}
        coords['pca_features'] = xr.DataArray(np.arange(self.n_components), coords={'pca_features': np.arange(self.n_components)}, dims='pca_features')
        out = xr.DataArray(X_ipca, coords=coords, dims=new_dims).unstack('combined')
        out.encoding = self.src.encoding
    
        return out
