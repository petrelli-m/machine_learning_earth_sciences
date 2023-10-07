import rasterio
import numpy as np

imagePath = 'SENTINEL2B_20210621-001635-722_L2A_T55HDB_C_V2-2/SENTINEL2B_20210621-001635-722_L2A_T55HDB_C_V2-2_FRE_'

bands_to_be_inported = ['B2', 'B3', 'B4', 'B8']

bands_dict = {}
for band in bands_to_be_inported:
    with rasterio.open(imagePath+ band +'.tif', 'r',
                       driver='GTiff') as my_band:
        bands_dict[band] = my_band.read(1)

