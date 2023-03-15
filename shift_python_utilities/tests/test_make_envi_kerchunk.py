import make_envi_kerchunk as mek
import xarray as xr

# Test
# rdn_path = "test-data/ang20170323t202244_rdn_7000-7010.hdr"
rdn_path = "../sbg-uncertainty/data/isofit-test-data/medium_chunk/ang20170323t202244_rdn_7k-8k.hdr"
loc_path = rdn_path.replace("_rdn_", "_loc_")
obs_path = rdn_path.replace("_rdn_", "_obs_")

output_file = mek.make_envi_kerchunk(rdn_path, loc_path, obs_path, "big-example.json")

dtest = xr.open_dataset("reference://", engine="zarr", backend_kwargs={
    "consolidated": False,
    "storage_options": {
        "fo": "test-cli.json"
    }
})

# from matplotlib import pyplot as plt
# dtest.sel(line=5, sample=3).radiance.plot(); plt.show()