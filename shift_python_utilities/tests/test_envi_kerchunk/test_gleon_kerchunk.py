# from shift_python_utilities.envi_kerchunk import gleon_kerchunk
# import xarray as xr
# import os

# root_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
# root_dir = os.path.join(root_dir, "test_data")

# try:
#     rfl_path = os.path.join(root_dir, "testing_raster.hdr")
#     output_file = rfl_path.replace(".hdr", ".json")
#     gleon_kerchunk(rfl_path, output_file)


#     dat = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
#         "consolidated": False,
#         "storage_options": {"fo": output_file}
#     })

#     dat.isel(sample=300, line=3500).reflectance.values
# except:
#     assert False, "Test Failed"
# finally:
#     if os.path.exists(output_file):
#             os.remove(output_file)
  