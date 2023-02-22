import pytest
import rasterio as rio
import os
from shift_utilities import reproject_raster

root_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
root_dir = os.path.join(root_dir, "test_data")


def reproject_raster_driver(**kwargs):
    in_path = os.path.join(root_dir, "test_raster_rotated")
    out_path =os.path.join(root_dir, "test_raster_rotated_output")
    
    with rio.open(in_path) as src:
        src_profile =src.profile
    
    try:
        reproject_raster(in_path,out_path, **kwargs)
    except:
        for file in [out_path, out_path + ".hdr", out_path + ".aux.xml"]:
            if os.path.isfile(file):
                os.remove(file)
        raise Exception("Reprojection failed!")
                
    with rio.open(out_path) as src:
        dst_profile = src.profile
    
    os.remove(out_path)
    os.remove(out_path + ".hdr")
    os.remove(out_path + ".aux.xml")
    
    return src_profile, dst_profile
 

@pytest.mark.parametrize(
    "crs, resampling_method, resolution, expected_transform, expected_crs",
    [
        (None, 'nearest', None, (2.6, -0., -0., -2.6), "EPSG:32616"),
        ("EPSG:32616", 'nearest', None, (2.6, -0. , -0., -2.6), "EPSG:32616"),
        ("EPSG:4326", 'nearest', (3.7, 3.7), (3.7, -0., -0., -3.7), "OGC:CRS84"),
    ],
)

def test_reproject_raster(crs, resampling_method, resolution ,expected_transform, expected_crs):
    src, dst = reproject_raster_driver(crs=crs, resampling_method=resampling_method, resolution=resolution)
    a,b,_,d,e,_,_,_,_ = dst['transform']
    assert (a,b, d, e) == expected_transform
    assert dst['crs'].to_string() == expected_crs

