from shift_python_utilities.envi_kerchunk import kerchunk_shift_rfl
import os
root_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
root_dir = os.path.join(root_dir, "test_data")
out_file = os.path.join(root_dir, "kerchunk_shift_rfl.json")
input_file= "/efs/efs-data-curated/v1/20220228/L2a/ang20220228t183924_rfl.hdr"
kerchunk_shift_rfl(input_file, out_file)


# import xarray as xr
# import rioxarray

# dtest = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
#     "consolidated": False,
#     "storage_options": {
#         "fo": "s3://dh-shift-curated/aviris/v1/gridded/zarr.json"
#     }
# })

# dsub = dtest.isel(x=5000,y=5000)
# %time dsubv = dsub.reflectance.values

# dat = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
#     "consolidated": False,
#     "storage_options": {"fo": "s3://dh-shift-curated/aviris/v1/gridded/20220224_box_rfl_phase.json"}
# })

# dat.y.max()
# dat.y.min()

# dsub = dat.sel(x = slice(730000, 730500), y = slice(3810500, 3810000))
# %time dvals = dsub.reflectance.values


# import rasterio as rio

# dat.attrs["map info"]

# dat2 = dat
# dat2 = dat2.rio.set_crs(32610)
# dat2 = dat2.rio.transfo

# # Sample --> X
# # Line --> Y
# drxr = xr.open_dataset("s3://dh-shift-curated/aviris/v1/gridded/20220224_box_rfl_phase", engine="rasterio")

# drxr.spatial_ref.attrs

# drxr.y.min().values
# ycoords.min()
# drxr.y.max().values
# ycoords.max()

# drxr.x.min().values
# xcoords.min()
# drxr.x.max().values
# xcoords.max()

# drxr.spatial_ref