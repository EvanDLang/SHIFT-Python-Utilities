from rasterio.crs import CRS
import os

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
