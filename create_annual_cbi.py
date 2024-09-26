# Script to create individual yearly data from a large CBI raster

# Import packages
import os
from os.path import join
import numpy as np
import rasterio as rio

# Setting directory paths
home = os.path.dirname(os.path.realpath("__file__"))
data_folder = join(home, "data")
out_folder = join(data_folder, "landfire_cbi", "full_cbi_annual")

path = join(data_folder, "landfire_cbi", "landfire_fire_raster_cbi_bc.tif")

# Setting values found in data
values = list(range(1, 23))
years = list(range(1999, 2021))

# Looping through each year, converting it to an np.int16 file, 
# writing out the nodata value as -99, and saving it. 
for i in values:
    print("On year {}".format(years[i-1]))
    with rio.open(path) as src:
        arr = src.read(i)
        arr *= 10000
        arr[np.isnan(arr)] = -99
        arr = arr.astype(np.int16)
        profile = src.profile
        profile.update(dtype=np.int16, count=1, nodata=-99)
    output_file = join(out_folder, "landfire_cbi_{}.tif".format(years[i-1]))
    with rio.open(output_file, 'w', **profile) as dst:
            dst.write(arr, 1)




