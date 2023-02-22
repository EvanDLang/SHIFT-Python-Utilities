"""thread_pool_executor.py

Operate on a raster dataset window-by-window using a ThreadPoolExecutor.

Simulates a CPU-bound thread situation where multiple threads can improve
performance.

With -j 4, the program returns in about 1/4 the time as with -j 1.
"""

import concurrent.futures
import multiprocessing
import threading

import rasterio
from rasterio._example import compute


def main(infile, outfile, num_workers=6, bands_per_task=1):
    """Process infile block-by-block and write to a new file

    The output is the same as the input, but with band order
    reversed.
    """

    with rasterio.open(infile) as src:

        # Create a destination dataset based on source params. The
        # destination will be tiled, and we'll process the tiles
        # concurrently.
        profile = src.profile

        #calculate transform array and shape of reprojected raster
        transform, width, height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height,resolution=src.transform._scaling, *src.bounds)

        profile.update(transform=transform, width=width, height=height)
        profile['APPEND_SUBDATASET'] = 'YES'


        with rio.open(output, "w+", **profile) as dst:
            windows = []
            
            for i in range(1, src.count + 1, bands_per_task):
                if i > src.count + 1 - bands_per_task:
                    windows.append(list(np.arange(i, src.count + 1)))
                else:
                    windows.append(list(np.arange(i, i+bands_per_task)))
            read_lock = threading.Lock()
            write_lock = threading.Lock()
            
            dst.descriptions = src.descriptions

            def process(window):
                with read_lock:
                    src_array = src.read(window)
                destination = np.zeros((len(window), height, width))
                # The computation can be performed concurrently
                result, _ =  rio.warp.reproject(
                    source=src_array,
                    destination=destination,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    dst_crs=src.crs,
                    dst_transform=transform,
                    dst_nodata=src.nodata,
                    resampling=Resampling.nearest
                )

                with write_lock:
                    dst.write(result, window)

            # We map the process() function over the list of
            # windows.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(process, windows)

# input_raster = "/home/jovyan/outputs/ang20160822t193312_rfl_v1n2_img_1_band"
# output = os.path.join("/efs/edlang1/test_output",  os.path.basename(input_rasters[0]) + "_reprojected_multi_thread")
# shapefile = "/efs/edlang1/Buoy_100m_offset/Buoy_100m.shp"

# geodf = gpd.read_file(shapefile)
# geodf = geodf.to_crs(geodf.estimate_utm_crs(datum_name='WGS 84'))
# %time main(input_rasters[0], output)
# %time clip_raster(output, geodf, os.path.join("/efs/edlang1/test_output", 'clip_test_rasterio_multi_thread'))