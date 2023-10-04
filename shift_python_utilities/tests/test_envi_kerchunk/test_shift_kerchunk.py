# from shift_python_utilities.envi_kerchunk import kerchunk_shift_rfl
# import os
# root_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
# root_dir = os.path.join(root_dir, "test_data")
# out_file = os.path.join(root_dir, "kerchunk_shift_rfl.json")
# input_file = "/home/jovyan/s3/dh-shift-curated/aviris/v1/gridded/20220224_box_rfl_phase.hdr" 
# kerchunk_shift_rfl(input_file, out_file)

# dtest = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
#     "consolidated": False,
#     "storage_options": {
#         "fo": "/home/jovyan/SHIFT-Python-Utilities/shift_python_utilities/test_data/kerchunk_shift_rfl.json"
#     }
# })