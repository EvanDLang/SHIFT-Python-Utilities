import rioxarray as rxr
from shift_python_utilities.shift_xarray._utils import _chunk_maker, _nested_iterator
from affine import Affine
import numpy as np
import xarray as xr
from functools import partial
from uuid import uuid4
from rasterio import windows
from rioxarray.rioxarray import _make_coords
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import itertools
from shift_python_utilities.shift_xarray._dask_orthorectification import _create_block_index_map_dict, _create_block_index_map_ncls
from rasterio.merge import copy_first, copy_last, copy_min, copy_max, copy_sum, copy_count

MERGE_METHODS = {
    "first": copy_first,
    "last": copy_last,
    "min": copy_min,
    "max": copy_max,
    "sum": copy_sum,
    "count": copy_count,
}

def get_chunk_bounds(ds):
    
    dim_map = {dim: i for i, dim in enumerate(ds.dims)}
    intervals = [[sum(c[:i]) for i in range(len(c)+1)] for c in ds.chunks]
    iterable =  [range(len(interval) - 1) for interval in intervals]
    
    bounds = []
    for xs in itertools.product(*iterable):
        subset = {dim: slice(intervals[ind][xs[ind]], intervals[ind][xs[ind] + 1]) for dim, ind in dim_map.items()}
        bounds += [ds.isel(subset).rio.bounds()]

    return bounds

def _retrieve_keys(li):
    val = []
    for l in li:
        if isinstance(l, list):
            val += _retrieve_keys(l)
        else:
            first, rest = l[0], l[1:]
            val += [rest]
    return val

def append_dict(d, k, v):
    if k in d:
        d[k] += [v]
    else:
        d[k] = [v]


def create_mask(arr, nodataval):
    if np.isnan(nodataval):
        mask = np.isnan(arr)
    elif np.issubdtype(arr.dtype, np.floating):
        mask = np.isclose(arr, nodataval)
    else:
        mask = arr == nodataval
    
    return mask
        
def do_dask_merge(copyto, nodataval, dst_shape, dst_masks, src_masks, indices, roff, coff, *blocks):
    # create an output array the shape of the chunk and fill with nodata
    region = np.zeros(dst_shape) + nodataval
    
    # loop through dependent blocks
    for i in range(len(blocks)):
        # retrieve masks and create the sub region
        m1 = dst_masks[i].astype(bool)
        m2 = src_masks[i].astype(bool)
        sub_region = region[:, m1] 
        
        # create the region mask
        region_mask = create_mask(sub_region, nodataval)
        
        # retrieve the new data and create the new_mask
        temp = blocks[i][:, m2]
        temp_mask = create_mask(temp, nodataval)
       
        # copy new data to sub_region using the provided copy function
        copyto(sub_region, temp, region_mask, temp_mask, index=indices[i], roff=roff[i], coff=coff[i])
        # copy sub_region to the output region
        region[:, m1] = sub_region
    
    return region

# other arguments later?
def dask_merge_arrays(srcs, method='first'):
    
    assert isinstance(srcs, list) and len(srcs) > 1, 'srcs must be a list of at least length 2!'
    
    if isinstance(method, str):
        assert method in MERGE_METHODS, f"Invalid method, The method must be one of the following {list(MERGE_METHODS.keys())} or a custom callable"
        copyto = MERGE_METHODS[method]
    elif isinstance(method, callable):
        copyto = method
    else:
        raise ValueError('Unknown method {0}, must be one of {1} or callable'
                         .format(method, list(MERGE_METHODS.keys())))
    
    # Get geospatial information from the first raster
    res = srcs[0].rio.resolution()
    first_crs = srcs[0].rio.crs
    nodataval = srcs[0].rio.nodata if srcs[0].rio.nodata is not None else -9999.
    dt = srcs[0].dtype
    
    xs = []
    ys = []

    for src in srcs:
        left, bottom, right, top = src.rio.bounds()
        xs.extend([left, right])
        ys.extend([bottom, top])

    dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    output_width = int(round((dst_e - dst_w) / res[0]))
    output_height = int(round((dst_n - dst_s) / -res[1]))

    output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], res[1])

    output_dims = {"y": output_height, "x": output_width}
    dim_map = {dim: i for i, dim in enumerate(srcs[0].dims)}
    dst_shape = tuple([output_dims[k] if k == 'x' or k == 'y' else srcs[0].shape[v] for k,v in dim_map.items()])
    dst_chunks = _chunk_maker(srcs[0].data.chunksize, dst_shape)

    coords = _make_coords(
        srcs[0],
        output_transform,
        output_dims['x'],
        output_dims['y'],
    )

    dest = da.zeros(dst_shape, dtype=dt, chunks=dst_chunks) 

    temp_out = xr.DataArray(
            name=srcs[0].name,
            data=dest,
            coords=coords,
            dims=tuple(srcs[0].dims),
            attrs=srcs[0].attrs,
        ).chunk(dst_chunks)
    
    temp_out.rio.write_nodata(srcs[0].rio.nodata, inplace=True)
    temp_out.rio.write_crs(srcs[0].rio.crs, inplace=True)
    temp_out.rio.write_transform(output_transform, inplace=True)

    dim_map = {dim: i for i, dim in enumerate(temp_out.dims)}
    # dst_chunk_bounds = get_chunk_bounds(temp_out)
    dst_shape_in_blocks = tuple(map(len, dst_chunks))
    # dst_inds = list(np.ndindex(dst_shape_in_blocks))
    range_blocks = _create_block_index_map_dict(temp_out.chunks, dim_map)
    chunks = temp_out.chunks[dim_map[temp_out.rio.y_dim]], temp_out.chunks[dim_map[temp_out.rio.x_dim]]
    range_map = _create_block_index_map_ncls(chunks, {'y': 0, 'x': 1})
    
    block_map = {}
    # loop through source rasters
    for raster_idx, src in enumerate(srcs):
        src_block_keys = src.__dask_keys__()
        shape_in_blocks = tuple(map(len, src.data.chunks))
        chunk_bounds = get_chunk_bounds(src)
        src_dim_map = {dim: i for i, dim in enumerate(src.dims)}

        # loop through source raster blocks
        for i, idx in enumerate(np.ndindex(shape_in_blocks)):

            # retrieve bounds of the chunk
            src_w, src_s, src_e, src_n = chunk_bounds[i]

            int_w = src_w if src_w > dst_w else dst_w
            int_s = src_s if src_s > dst_s else dst_s
            int_e = src_e if src_e < dst_e else dst_e
            int_n = src_n if src_n < dst_n else dst_n

            # create a destination
            dst_window = windows.from_bounds(int_w, int_s, int_e, int_n, output_transform)

            # round offsets and lengths 
            dst_window_rnd_off = dst_window.round_offsets().round_lengths()

            temp_height, temp_width = (dst_window_rnd_off.height, dst_window_rnd_off.width) 

            roff, coff = (
                max(0, dst_window_rnd_off.row_off),
                max(0, dst_window_rnd_off.col_off),
            )

            # retrieve destination data
            dst_data = temp_out.rio.isel_window(dst_window_rnd_off, pad=False)

            # use roff, height, coff and width to get the chunks from the output array
            vals = [(roff, roff + temp_height), (coff, coff + temp_width)]
            def get_blocks(r1, r2, interval):
                    _, res = interval.all_overlaps_both(np.array([r1]), np.array([r2]), np.arange(1))
                    return res

            t = [get_blocks(*vals[i], range_map[i]) for i in range(len(range_map))]
            inds = np.array(np.meshgrid(t[0], t[1])).T.reshape(-1, 2)
            inds = np.concatenate((np.zeros((len(inds),len(idx)-2)), inds), axis=1)
            inds = inds.astype(int)
            indst = inds.T 
            inds = list(zip(*indst))
         
    

            # create an ind map for the chunks present in the dst window
            dst_dim_map = {dim: i for i, dim in enumerate(dst_data.dims)}
            ind_map = _create_block_index_map_dict(dst_data.chunks, dst_dim_map)
          
            # loop through chunks in dst data
            for j, ind in enumerate(itertools.product(*map(range, dst_data.data.blocks.shape))):
                # create a mask for t src chunk (m1) and a mask for the dst data chunk (m2)
                chunk_shape = tuple(ch[i] for ch, i in zip(dst_chunks, inds[j]))
                chunk_shape = chunk_shape[src_dim_map[src.rio.y_dim]], chunk_shape[src_dim_map[src.rio.x_dim]]
               
                dst_data_shape =  dst_data.shape[dst_dim_map[dst_data.rio.y_dim]], dst_data.shape[dst_dim_map[dst_data.rio.x_dim]]

                m1 = np.zeros(chunk_shape).flatten()
                m2 = np.zeros(dst_data_shape)

                window_slices = [slice(*i) for i in ind_map[ind]]
                dst_slices = [slice(*i) for i in range_blocks[inds[j]]]

                t1 = temp_out.isel(y=dst_slices[1], x=dst_slices[2])
                t2 = dst_data.isel(y=window_slices[1], x=window_slices[2])

                x_mask = np.in1d(t1.x.values, t2.x.values)
                y_mask = np.in1d(t1.y.values, t2.y.values)
                
                m1[np.arange(np.prod(m1.shape)).reshape(chunk_shape)[y_mask][:, x_mask].flatten()] = 1.
                m1 = m1.reshape(tuple(ch[i] for ch, i in zip(dst_chunks, inds[j]))[1:])

                m2[window_slices[1], window_slices[2]] = 1

                append_dict(block_map, inds[j], (_nested_iterator(src_block_keys, idx), m1, m2, raster_idx, roff, coff))
    
    name = 'merge'
    tk = uuid4().hex
    name = f"{name}-{tk}"

    dsk = {}

    merge_proc = partial(
            do_dask_merge,
            copyto,
            nodataval
        )

    for i, idx in enumerate(np.ndindex(dst_shape_in_blocks)):
        k = (name, *idx)
        chunk_shape = tuple(ch[i] for ch, i in zip(dst_chunks, idx))

        if idx in block_map:

            block_deps, dst_masks, src_masks, indices, roff, coff = list(zip(*block_map[idx]))
            dsk[k] = (merge_proc, chunk_shape, dst_masks, src_masks, indices, roff, coff, *tuple(block_deps))
        else:
            dsk[k] = (np.full, chunk_shape, nodataval, src.dtype)

    dsk = HighLevelGraph.from_collections(name, dsk, dependencies=(srcs))  
    out = da.Array(dsk, name, chunks=dst_chunks, dtype=src.dtype, shape=dst_shape)
    
    out = xr.DataArray(
        name=name,
        data=out,
        coords=coords,
        dims=tuple(srcs[0].dims),
        attrs=srcs[0].attrs,
    )

    out.rio.write_nodata(nodataval, inplace=True)
    out.rio.write_crs(first_crs, inplace=True)
    out.rio.write_transform(output_transform, inplace=True)
    return out


def dask_merge_datasets(srcs, method='first'):
    
    representative_ds = srcs[0]
    merged_data = {}
    
    for data_var in representative_ds.data_vars:
        merged_data[data_var] = dask_merge_arrays(
            [src[data_var] for src in srcs],
            method=method          
        )
        
    data_var = list(representative_ds.data_vars)[0]
    
    xds = xr.Dataset(
        merged_data,
        coords=_make_coords(
            merged_data[data_var],
            merged_data[data_var].rio.transform(),
            merged_data[data_var].shape[-1],
            merged_data[data_var].shape[-2],
            force_generate=True,
        ),
        attrs=representative_ds.attrs,
    )
    xds.rio.write_crs(merged_data[data_var].rio.crs, inplace=True)
    return xds