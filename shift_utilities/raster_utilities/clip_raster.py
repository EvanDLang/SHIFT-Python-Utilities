import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd


def clip_raster(input_raster, geodf, output_raster):
    """
    clip_raster(
        input_raster, 
        geodf,
        output_raster
    )

    Clips a raster using a geodf.

    Parameters
    ----------
    input_raster : str
        path to input raster
    geodf: Geopandas.DataFrame
        A geodataframe containing all the shapes you would like to clip out(must use the same crs as the input raster)
    output_raster : str
        write path
    Returns
    -------
        None (writes clipping to the output_raster)
    """
    with rio.open(input_raster) as src:
        
        # verify the crs of the shape file and input raster are the same
        assert src.crs == rio.crs.CRS.from_string(str(geodf.crs)), "Geodataframe must have the same crs as the input raster"
        
        # copy the input rasters profile
        profile = src.profile
        
        # crop the raster using the polygons from the shapefile
        crop_img, crop_transform = mask(src, shapes=geodf.geometry.values, crop=True, all_touched=True)
        
        # update the profile
        profile['transform'] = crop_transform
        profile['height'] = crop_img.shape[1]
        profile['width'] = crop_img.shape[2]
        
        # retrieve the band descriptions
        tag_values = list(src.tags().values())
        
        # create a new raster using the updated profile
        with rio.open(output_raster, 'w', **profile) as dst:
            # set band descriptions
            dst.descriptions = src.descriptions
            for i in range(1, profile['count'] + 1):
                dst.write_band(i, crop_img[i - 1])