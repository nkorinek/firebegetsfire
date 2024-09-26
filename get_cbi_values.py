# Import packages
import os
from os.path import join
import datetime
import numpy as np
from glob import glob
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
import rasterio as rio

# Setting directory paths
home = os.path.dirname(os.path.realpath("__file__"))
data_folder = join(home, "data")

# Helper functions
def check_bounds(geom_bounds, raster_bounds):
    """Check that a geometry is contained by a raster's bounds 
    so data can be extracted from the raster based on the geometry.
    
    Parameters
    -----------
    geom_bounds : GeoPandas bounds object
        The bounds returned by 'gdf.bounds' for the geometry data is being 
        extracted into. 
    raster_bounds : BoundingBox
        The BoundingBox returned by 'image.bounds' for the image data is 
        being extracted from.

    Returns
    -----------
    boolean
        A boolean checking if both the latitude and longitude of the geometry
        are contained within the raster
    """
    # Check geometry is withing raster bounds
    lon_bounds = (geom_bounds[0] > raster_bounds[0]) & (
        geom_bounds[2] < raster_bounds[2]
    )
    lat_bounds = (geom_bounds[1] > raster_bounds[1]) & (
        geom_bounds[3] < raster_bounds[3]
    )
    return lon_bounds & lat_bounds


def get_average_max_cbi(year, geom, path):
    """
    Get the average and maximum values from an image within a geometry.

    Parameters
    -----------
    year : int
        The year which the fire occured.
    geom : Polygon or Multipolygon
        Polygon for which data will be extracted.
    path : string
        The path to the raster containing data.

    Returns
    -----------
    tuple
        A tuple containing two floats, one that is the maximum value of CBI
        found within the fire geometry, and one that is the average value of 
        the CBI found within the fire geometry.
    """
    # Open image to get bounds
    with rio.open(path) as src:
        raster_bounds = src.bounds
    
    # Check that geometry is contained within image
    in_bounds = check_bounds(geom.bounds, raster_bounds)
    if in_bounds:
        # Open and clip raster
        area = rxr.open_rasterio(path).rio.clip([geom], from_disk=True)
        # Convert raster to float32 in order to add nan values to it
        masked = area.values.astype(np.float32)
        masked[masked == -99] = np.nan
        # Convert to CBI to actual units
        masked = masked / 10000
        # Check that data is present within the geometry
        if not np.isnan(masked).all():
            # Get the max and average of the data
            max_val = np.nanmax(masked)
            average_val = np.nanmean(masked)
            # Delete arrays
            del masked
            del area
            return (max_val, average_val)
        else:
            # Delete arrays
            del masked
            del area
            # Return nan values if no data present
            return (-99, -99)
    else:
        # Return nan values if not within bounds
        return (-99, -99)


# Creating a list of all CBI tifs
cbi_tifs = sorted(glob(join(data_folder, "landfire_cbi", "full_cbi_annual", "*.tif")))

# Get CRS from example cbi
with rio.open(cbi_tifs[0]) as src:
    crs = src.crs

print("Opening western ecoregion shape file")
# Creating a western US shape to clip to
western_eco_path = join(
    data_folder, "EPA-ecoregions", "western_us", "western_us_eco_l3.shp"
)
western_eco = gpd.read_file(western_eco_path)

print("Opening clipped firedpy data")
# Open clipped firedpy to western US ecoregions
clipped_fire_path = join(
    data_folder, "clipped_data", "clipped_firedpy", "clipped_firedpy.shp"
)

fired_gdf_orig = gpd.read_file(clipped_fire_path, parse_dates=["ig_date", "last_date"])

# Sort fired events by date and reproject
fired_gdf = fired_gdf_orig.sort_values(by="ig_date").to_crs(crs)
fired_gdf_drought = fired_gdf_orig.to_crs("EPSG:5070")


events = []
# Create yearly dataframe
for event_year, group in fired_gdf.groupby("ig_year"):
    # Time the operation
    start = datetime.datetime.now()
    fire_len = len(group)
    pos = 0
    print("-------")
    print("Working on year {}".format(event_year))
    print("-------")
    # Get the tif for the relevant year
    tif_check = glob(
        join(
            data_folder,
            "landfire_cbi",
            "full_cbi_annual",
            "*{}*.tif".format(event_year),
        )
    )
    # Check that the year exists
    if len(tif_check) > 0:
        tif = tif_check[0]
        # Loop through the dataframe based on the year
        for i, row in group.iterrows():
            if pos % 200 == 0:
                print("On row {} of {}".format(pos, fire_len))
            # Get relevant data from dataframe
            year = row["ig_year"]
            geometry = row["geometry"]
            maximum = 0
            ave = 0
            fire_id = row["id"]
            fire_area = row["tot_ar_km2"]
            start_date = row["ig_date"]
            # Get the final date of the fire
            end_date = str(
                (
                    datetime.datetime.strptime(row["last_date"], "%Y-%m-%d")
                    + datetime.timedelta(1)
                ).date()
            )
            # Find CBI values
            max_test, ave_test = get_average_max_cbi(event_year, geometry, tif)
            # Check that there are valid values
            if max_test > 0:
                maximum = max_test
                ave = ave_test
            if maximum > 0:
                # Add all data to a dataframe and add it to the end of a list
                events.append(
                    pd.DataFrame(
                        {
                            "id": fire_id,
                            "year": year,
                            "str_date": start_date,
                            "end_date": end_date,
                            "ave_sev": ave,
                            "fire_area": fire_area,
                            "geometry": geometry,
                        },
                        index=[1],
                    )
                )
            pos += 1
    end = datetime.datetime.now()
    print("Time of {} for {}".format(end - start, event_year))

# Output results

print("Outputting files!")
final_gdf_path = join(data_folder, "firedpy_severities", "firedpy_severities.shp")

main_gdf = pd.concat(events)
final_gdf = gpd.GeoDataFrame(main_gdf, geometry=main_gdf.geometry, crs="EPSG:4326")
final_gdf = final_gdf.drop_duplicates()

final_gdf.to_file(final_gdf_path)

# Performing overlay in order to find reburn events
overlay_gdf_path = join(
    data_folder, "firedpy_severities_overlay", "firedpy_severities_overlay.shp"
)
overlay = gpd.overlay(final_gdf, final_gdf, how="intersection")

# Only use overlays where the second fire happened chronologically after the 
# first fire
overlay = overlay[overlay["year_1"] < overlay["year_2"]]

overlay.to_file(overlay_gdf_path)

print("File output complete!")
