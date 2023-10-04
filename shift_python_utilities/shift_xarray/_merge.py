"""
WIP

- I can get the output shape, and transform
- I still need to figure how to retrieve the appropriate source chunks when looping through the destination chunks
- write the do_merge function after mapping problem is solved
"""



for src in srcs:
    left, bottom, right, top = src.rio.bounds()
    xs.extend([left, right])
    ys.extend([bottom, top])
    
dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

res = (srcs[0].rio.transform()[0], -srcs[0].rio.transform()[4])

output_width = int(round((dst_e - dst_w) / res[0]))
output_height = int(round((dst_n - dst_s) / res[1]))
output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])
output_dtype = srcs[0].dtype
output_nodata = srcs[0].rio.nodata
output_crs = srcs[0].rio.crs
output_dims = {"y": output_height, "x": output_width}

dim_map = {dim: i for i, dim in enumerate(srcs[0].dims)}
dst_shape = tuple([output_dims[k] if k == 'x' or k == 'y' else srcs[0].shape[v] for k,v in dim_map.items()])
dst_chunks = _chunk_maker(srcs[0].data.chunksize, dst_shape)


shape_in_blocks = tuple(map(len, dst_chunks))
for idx in np.ndindex(shape_in_blocks):
    print(idx)
    #idx to coords?
    #map blocks to src blocks for all datasets
    #if overlap then method
    #else copy over
    break
    
    """
inputs so far: dim map (data sets as list), merge method (see rasterio merge)
validation, same number of bands, dtype(default to first?), resolution, crs, and nodata?
output shape
output chunks
make output profile
loop through destination chunks
determine chunks from sources****
"""
    
# From rasterio merge
# 0. Precondition checks
    #    - Check that source is within destination bounds
    #    - Check that CRS is same
# 1. Compute spatial intersection of destination and source
# 2. Compute the source window
# 3. Compute the destination window
# 4. Read data in source window into temp
# 5. Copy elements of temp into dest