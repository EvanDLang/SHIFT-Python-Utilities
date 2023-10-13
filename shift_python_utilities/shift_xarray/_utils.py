from rasterio.crs import CRS
from ncls import NCLS
import os
import numpy as np

def _create_interval(ind, chunk_size, previous_ind=None, previous_value=None):
    if ind == 0:
        ind1 = ind * chunk_size
        ind2 = (ind+1) * chunk_size
    else:
        if previous_ind == ind:
            ind1 = previous_value[0]
            ind2 = previous_value[0] + chunk_size
        else:
            ind1 = previous_value[1]
            ind2 = previous_value[1] + chunk_size
    return ind1, ind2

def _create_block_index_map_dict(chunks, dim_map):
    prev = tuple([None for i in dim_map])
    block_map = {}
    for ind in np.ndindex(tuple(map(len, chunks))):
        block_map[ind] = [(0, 0) for i in ind]
        for axis, i in enumerate(ind):
            chunk_size = chunks[axis][i]
            r = _create_interval(i, chunk_size, prev[axis], block_map[prev][axis] if prev in block_map else None)
            block_map[ind][axis] = r
        prev = ind
    
    return block_map


def _create_block_index_map_ncls(chunks, dim_map):
    intervals = [[] for i in dim_map]
    for axis, chunksizes in enumerate(chunks):
        for i, chunksize in enumerate(chunksizes):
            if i == 0:
                ind = 0, (i+1) * chunksize
            else:
                ind = prev[1], prev[1] + chunksize
            prev = ind
            intervals[axis] += [prev]
            
    for i in range(len(intervals)):
        interval = np.array(intervals[i])
        intervals[i] =  NCLS(interval[:, 0], interval[:, 1], np.arange(len(interval[:, 0])))
        
    return intervals


def reformat_path(dataset, path):
    assert dataset in ['rfl', 'rdn', 'glt', 'igm']
    
    if 'L1' in path:
        src_dataset = path.split('L1/')[-1].split('/')[0]
    else:
        src_dataset = path[-3:]
    
    if src_dataset == 'rfl':
        new_path = path.replace('L2a', f'L1/{dataset}')[:-4] + f'_{dataset}'
    elif src_dataset == 'rdn':
        segs = path.split(src_dataset)[:-1]
        new_path = segs[0] + dataset + segs[1] + dataset
        
    else:
        new_path = path.replace(src_dataset, dataset)
    
    if dataset == 'rdn':
        new_path += '_v2aa1_clip'
    
    return new_path

def _nested_iterator(li, key):
    for l in li:
        if isinstance(l, list):
            val = _nested_iterator(l, key)
            if val is not None:
                return val
        else:
            first, rest = l[0], l[1:]
            if rest == key:
                return l

def _retrieve_crs_from_igm(igm):
    # igm = _load_file(url)
    description = igm.attrs['description']
    ind = description.find("UTM")
    coord_system, _, zone, direction = description[57:].split(" ")
    direction = False if direction == 'North' else True
    epsg_code = 32600
    epsg_code += int(zone)
    if direction is True:
        epsg_code += 100

    return CRS.from_epsg(epsg_code)

def _chunk_maker(chunk, shape):

    remainders = [shape[i] % chunk[i] for i in range(len(chunk))]
    
    chunks = []
    for i in range(len(chunk)):
        temp = tuple([chunk[i] for j in range(shape[i] // chunk[i])])
        if remainders[i] > 0:
            temp += (remainders[i],)
        chunks += [temp]

    return tuple(chunks)
