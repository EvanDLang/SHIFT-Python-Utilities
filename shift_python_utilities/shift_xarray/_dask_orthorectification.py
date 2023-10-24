"""
credit to odc geobox for providing the loose structure
"""

from uuid import uuid4
from functools import partial
import numpy as np
import pandas as pd
import rioxarray as rxr
import dask.array as da
import copy
from dask.highlevelgraph import HighLevelGraph
from ._utils import _nested_iterator, _chunk_maker, _create_block_index_map_ncls, _create_block_index_map_dict


def _chunk_value_retriever(ind, chunk_size, previous_ind=None, previous_value=None):
    if ind == 0:
        ind1 = ind * chunk_size
        ind2 = (ind + 1) * chunk_size
    else:
        if previous_ind == ind:
            ind1 = previous_value[0]
            ind2 = previous_value[0] + chunk_size
        else:
            ind1 = previous_value[1]
            ind2 = previous_value[1] + chunk_size

    return ind1, ind2


def _do_chunked_orthorectification(glt_array, v_glt, src_intervals_dict, retrieve_ind, dim_map, nodata, slices, blocks, glt_idxs, keys, dst_shape, *args):
    # create output array
    dst = np.zeros(dst_shape) + nodata
    
    # adjust the glt values from image level to block level
    new_inds = (glt_idxs - np.apply_along_axis(retrieve_ind, 1, blocks, src_intervals_dict))[:, [dim_map['y'], dim_map['x']]]
    
    # slice the glt array based on the destination chunk 
    glt_array_slice = glt_array[Ellipsis, slices[dim_map['y']], slices[dim_map['x']]]
    
    # reshape the slice for creating a mask
    reshaped_slice = glt_array_slice.reshape((glt_array_slice.shape[0], glt_array_slice.shape[1] * glt_array_slice.shape[2]))
    
    # transpose and trip off non x and y dimensions
    y_x = glt_idxs.T[[dim_map['y'], dim_map['x']]]
    
    # creates empty masks
    src_mask = np.zeros(len(glt_idxs)) + np.nan
    dst_mask = np.zeros(reshaped_slice.shape[1]) + np.nan
    
    # iterate through each unique source key and create a key which indicates what values came from what source block
    for i, dep in enumerate(keys):
        src_mask[np.argwhere(np.prod(blocks == dep, axis = -1))] = i
        dst_mask[(np.in1d(reshaped_slice[1], y_x[:, src_mask==i][0]) & np.in1d(reshaped_slice[0], y_x[:, src_mask==i][1]))] = i
    
    # reshape the dst mask back to the output shape
    dst_mask = dst_mask.reshape((glt_array_slice.shape[1], glt_array_slice.shape[2]))
    
    # if y and x are not the last two axes, set up to transpose the dimensions for broadcasting and then transpose them to there original order
    reorder = False
    if list(dim_map)[-2:] != ['y', 'x']:
        reorder = True
        reordered_dim_map = copy.deepcopy(dim_map) 
        for  k in ['y', 'x']:
            reordered_dim_map[k] = reordered_dim_map.pop(k)

        original_dims = {k: list(reordered_dim_map).index(k) for k in dim_map}
    
    # iterate through each source chunk
    for i, arr in enumerate(args):
        # create a mask for the source block using the keys
        m1 = src_mask == i
        m2 = dst_mask == i
        
        # if the last two dims are not y, x then tranpose
        if reorder:
            dst = dst.transpose(list(reordered_dim_map.values()))
            arr = arr.transpose(list(reordered_dim_map.values()))
        
        # broadcast values
        dst[Ellipsis, m2] = arr[Ellipsis, new_inds[m1, 0], new_inds[m1, 1]] 
        
        # if transposed, reorder to original dims order
        if reorder:
            dst = dst.transpose(list(original_dims.values()))
      
    return dst


def _dask_orthorectification(src, dim_map, glt_array, v_glt, glt_dims, nodata):
    
    name = 'orthorectify'
    tk = uuid4().hex
    name = f"{name}-{tk}"
    
    src_block_keys = src.__dask_keys__()

    # get the dst shape from the glt shape (y, x) and src(other dims), create dst chunks based on the src chunk sizes and the dst shape
    dst_shape = tuple([glt_array.shape[glt_dims[k]] if k == 'x' or k == 'y' else src.shape[v] for k,v in dim_map.items()])
    dst_chunks = _chunk_maker(src.chunksize, dst_shape)
    
    # get the blocks shape for iterating
    shape_in_blocks = tuple(map(len, dst_chunks))
    
    # create index intervals
    src_intervals = _create_block_index_map_ncls(src.chunks, dim_map)
    src_intervals_dict = _create_block_index_map_dict(src.chunks, dim_map)
    dst_intervals = _create_block_index_map_dict(dst_chunks, dim_map)
    
    def _retrieve_ind(row, intervals):
        starts, ends = list(zip(*intervals[tuple(row)]))
        return np.array(starts)


    ortho_proc = partial(
        _do_chunked_orthorectification,
        glt_array,
        v_glt,
        src_intervals_dict,
        _retrieve_ind,
        dim_map,
        nodata
    )
    
    dsk = {}
    
    for idx in list(np.ndindex(shape_in_blocks)):
        # create a block name
        k = (name, *idx)

        # get the shape of the chunk
        chunk_shape = tuple(ch[i] for ch, i in zip(dst_chunks, idx))

        # generate slices from the destination intervals
        slices = [slice(s[0], s[1]) for s in dst_intervals[idx]]

        # use the x and y slices to retrieve the mask from the v_glt
        mask = v_glt[slices[dim_map['y']], slices[dim_map['x']]]

        # check if the slice is all nodata values
        if np.sum(mask) > 0:

            # for the glt the y axis is 1 and the x axis is 0
            y_axis = glt_array[1, slices[dim_map['y']], slices[dim_map['x']]][mask]
            x_axis = glt_array[0, slices[dim_map['y']], slices[dim_map['x']]][mask]
            
            glt_idxs = []
            for dim, axis in dim_map.items():
                if dim == 'x':
                    glt_idxs += [x_axis]
                elif dim == 'y':
                    glt_idxs += [y_axis]
                else:
                    glt_idxs += [[slices[axis].start for j in range(len(y_axis))]]

            glt_idxs = np.vstack(glt_idxs).T

            def get_blocks(a, interval):
                _, res = interval.all_overlaps_both(a, a+1, np.arange(len(a)))
                return res

            blocks = np.array([get_blocks(glt_idxs[:, i], src_intervals[i]) for i in range(glt_idxs.shape[1])]).T
            keys = pd.DataFrame(blocks).drop_duplicates().values
            block_deps = [_nested_iterator(src_block_keys, tuple(dep)) for dep in keys]


            dsk[k] = (ortho_proc, slices, blocks, glt_idxs, keys, chunk_shape, *tuple(block_deps))

        else:
            dsk[k] = (np.full, chunk_shape, nodata, src.dtype)

    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=(src))
    
    dsk = da.Array(dsk, name, chunks=dst_chunks, dtype=src.dtype, shape=dst_shape)
    
    return dsk