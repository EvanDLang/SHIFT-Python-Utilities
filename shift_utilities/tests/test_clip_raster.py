import pytest
import rasterio as rio
import geopandas as gpd
import os
from shift_utilities.raster_utilities import clip_raster

root_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
root_dir = os.path.join(root_dir, "test_data")

@pytest.fixture
def geodf():
    shapefile = os.path.join(root_dir,"testing_shape_file", "testing_shape_file.shp")
    return gpd.read_file(shapefile)

def clip_raster_driver(geodf):
    in_path = os.path.join(root_dir, "test_raster_rotated")
    out_path = os.path.join(root_dir, "test_raster_rotated_clipped")
    
    with rio.open(in_path) as src:
        src_profile = src.profile
    
    try:
        clip_raster(in_path, geodf, out_path)
    except:
        for file in [out_path, out_path + ".hdr", out_path + ".aux.xml"]:
            if os.path.isfile(file):
                os.remove(file)
        return False
    
    with rio.open(out_path) as src:
        dst_profile = src.profile
    
    os.remove(out_path)
    os.remove(out_path + ".hdr")
    os.remove(out_path + ".aux.xml")
    
    return src_profile, dst_profile
 
    
def test_clip_raster(geodf):
    # fails because of crs differences
    assert not clip_raster_driver(geodf)
    
    src, dst = clip_raster_driver(geodf.to_crs(geodf.estimate_utm_crs(datum_name='WGS 84')))
    a1, b1, _, d1, e1, _ , _ , _ ,_ = src['transform']
    a2, b2, _, d2, e2, _ , _ , _ ,_ = dst['transform']
    assert a1 == a2
    assert b1 == b2
    assert d1 == d2
    assert e1 == e2

