import os
from os.path import join
import subprocess
import pandas as pd
import numpy as np
from glob import glob
import richdem as rd
import rioxarray as rxr
import geopandas as gpd
import rasterio as rio
import elevation
import gc
import geemap
import ee

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
        are contained within the raster.
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



def calc_av_slope_elevation(geom):
    """
    A function to get the average slop and elevation within a geometry.
    
    Parameters
    -----------
    geom : Polygon or Multipolygon
        Polygon for which elevation will be calculated.

    Returns
    -----------
    tuple
        A tuple of two floats containing the average elevation and the average
        slope for the geometric area.
    """
    try:
        path = join(data_folder, "temp_elevation.tif")
        # Download elevation data within the geometry bounds into a temporary
        # file
        elevation.clip(geom.bounds, output=path)
        # Get the DEM of the area
        dem = rd.LoadGDAL(path)
        # Calculate the slope for the area
        slope = rd.TerrainAttribute(dem, attrib='slope_riserun')
        # Remove all data no longer needed
        os.remove(path)
        elevation.clean()
        gc.collect()
        # Find the average values for the DEM and the slope data
        av_elev = np.nanmean(dem)
        av_slope = np.nanmean(slope)
        # Return those values
        return(av_elev, av_slope)
    # If there is an error with the area, return nan values
    except subprocess.CalledProcessError:
        return(np.nan, np.nan)

def cal_area(geom):
    """
    A fuction to calculate the area of a Polygon or Multi Polygon.

    Parameters
    -----------
    geom : Polygon or Multipolygon
        Polygon for which elevation will be calculated.

    Returns
    -----------
    float
        The area of the geometry in sqkm.
    """
    # Convert geometry to a non-projected coordinate system for accurate 
    # area calculations
    area_geom = geom.to_crs("EPSG:6933")
    area = area_geom.geometry.area
    # Return the area converted to sqkm
    return(area.values[0]/1000000)

def get_geometry(gdf, fire_id, col_name, fire2_id=None, col_name2=None):
    """
    A fuction to get the data from either a specific fire, or from a reburn
    event with two specific fire ids.

    Parameters
    -----------
    gdf : GeoDataFrame
        A GeoDataFrame containing the fire's data.
    fire_id : string
        The ID value of a specific fire.
    col_name : string
        The name of the column containing that fire's id.
    fire2_id : string (optional)
        The ID value of a second fire if you need a reburn event with two
        specific fires.
    col_name2 : string (optional)
        The name of the column containing the second fire's id.

    Returns
    -----------
    GeoSeries
        A single row GeoSeries containing the desired row(s).
    """
    if fire2_id:
        desired_rows = gdf[(gdf[col_name] == fire_id)&
                           (gdf[col_name2] == fire2_id)]
        return desired_rows
    desired_rows = gdf[gdf[col_name] == fire_id]
    return(desired_rows)

def get_clip_diff(year1_gdf, year2_gdf):
    """
    Find the difference between two geometries.

    Parameters
    -----------
    year1_gdf : GeoDataframe
        The GeoDataframe containing the area you want to compare to another 
        area.
    year2_gdf : GeoDataframe
        The GeoDataframe containing the area you want to find the difference
        in.
    
    Returns
    -----------
    geometry
        The geometry of the area of year2_gdf that didn't overlap year1_gdf
    """
    out_fire = year2_gdf.reset_index().difference(year1_gdf.reset_index())
    return(out_fire)

def get_diff_overlay_geom(overlay_gdf, fire_gdf, fire1_id, fire2_id):
    """
    Return the difference and overlaying areas between two fires.

    Parameters
    -----------
    overlay_gdf : GeoDataframe
        The GeoDataframe containing the overlaid area between two fires.
    fire_gdf : GeoDataframe
        The GeoDataframe containing all original fires. 
    fire1_id : string
        The ID value of the first fire being compared.
    fire2_id : string
        The ID value of the second fire being compared.

    Returns
    -----------
    tuple
        A tuple containing both the area that doesn't overlap in fire2_id and
        the overlaying area between fire1_id and fire2_id
    """
    # Get the overlayed area
    overlay = get_geometry(overlay_gdf, fire1_id, "id_1", fire2_id, "id_2")
    fire1_geom = get_geometry(fire_gdf, fire1_id, "id")
    fire2_geom = get_geometry(fire_gdf, fire2_id, "id")
    # Calculate the difference in area between two fires
    fire2_diff = get_clip_diff(fire1_geom, fire2_geom)
    return(fire2_diff, overlay)


def get_ecoregion(ecoregions, geom):
    """
    A function to find the ecoregion in which a geometry is contained within.

    Parameters
    -----------
    ecoregions : GeoDataframe
        A GeoDataframe containing all US level 3 ecoregions.
    geom : GeoDataframe
        A GeoDataframe containing the geometry of a reburn event.

    Returns
    -----------
    string
        A string of the L3 EPA ecoregion code.
    """
    overlap = ecoregions.overlay(geom)
    return(overlap['US_L3CODE'].values[0])    


def find_nearest_nlcd_year(year):
    """
    Find the nearest year that there is NLCD data before any given year.
    
    Parameters
    -----------
    year : int
        An int of the year the reburn event occured in. 

    Returns
    -----------
    string
        A string of the path to the nearest NLCD tif that occured before the 
        reburn event.
    """
    NLCD_years = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]
    
    # If the year is before 2001, return 2001
    if year < 2001:
        return 2001
    
    # Otherwise, find the nearest NLCD year before the given year
    nearest_year = max(filter(lambda x: x <= year, NLCD_years))
    nlcd_formatter = os.path.join(data_folder, "NLCD", "NLCD_clip{}.tif")

    return nlcd_formatter.format(nearest_year)

def check_forest_cover(year, geometry):
    """A function to calculate what percent of a geometry is covered by 
    forested areas according to the NLCD

    Parameters
    -----------
    year : int
        The year that the fire occured in
    geometry : Polygon
        The area the fire occured in

    Returns
    -----------
    tuple
        A tuple containing the most frequent forest cover value in the 
        geometry and the percent of the geometry that was covered by 
        forests.
    """
    # Find the nearest NLCD data for the given year
    nlcd_path = find_nearest_nlcd_year(year)
    
    # Read raster data and clip with the given geometry
    area = rxr.open_rasterio(nlcd_path).rio.clip(geometry, from_disk=True)
    masked = area.values

    # Filter out zero values
    non_zero_values = masked[masked != 0]
    
    # Calculate the percentage of forest cover (41, 42, 43)
    forest_cover_values = [41, 42, 43]
    forest_pixels = np.isin(non_zero_values, forest_cover_values)
    forest_count = np.sum(forest_pixels)
    total_non_zero_pixels = len(non_zero_values)
    
    if total_non_zero_pixels > 0:
        # Calculate the percentage of the area with forested pixels
        forest_cover_percentage = (forest_count / total_non_zero_pixels) * 100
        # Find the most common pixel value
        most_frequent_value = np.bincount(non_zero_values).argmax()
    else:
        # If there are no non-zero values, return None
        most_frequent_value = None
        forest_cover_percentage = 0
    
    return most_frequent_value, forest_cover_percentage


def gdf_to_geojson(gdf, output_file):
    """Convert a GeoDataFrame to a GeoJSON file.
    
    Parameters
    -----------
    gdf : GeoDataframe
        GeoDataframe to be turned into a geojson.
    output_file : string
        The path to write the file to.
    """
    gdf.to_file(output_file, driver='GeoJSON')


def calculate_relative_humidity(T, D):
    """
    Calculate relative humidity using temperature (T) and dewpoint (D).

    Parameters
    -----------
    T : float 
        Temperature in Celsius.
    D : float 
        Dewpoint temperature in Celsius.

    Returns
    -----------
    float
        The relative humidity value.
    """
    return round(np.exp(((17.625*D)/(243.04+D))-((17.625*T)/(243.04+T)))*100, 3)


def calculate_svp(T):
    """
    Calculate the saturation vapor pressure for a given temperature.

    Parameters
    -----------
    T : float
        The temperature to calculate the svp for.

    Returns
    -----------
    float
        The svp value.

    """
    return(round(610.78*np.exp(T/(T +237.3)*17.2694), 3))


def calculate_vpd(svp, rh):
    """
    Calculate the vapor pressure deficit based on the saturation vapor pressure
    and relative humidity of an area. 

    Parameters
    -----------
    svp : float
        The saturation vapor pressure to base calculations on.
    rh : float
        The relative humidity to base calculations on. 

    Returns
    -----------
    float
        The VPD value.
    """
    return(round(svp*(1-rh/100), 2))


def get_era5_statistics(geometry, start_date, end_date, temp_path):
    """
    Retrieve ERA5 data for a given geometry and time range, and calculate 
    mean temperature and mean dewpoint for that area.

    Parameters
    -----------
    geometry : GeoDataframe
        The GeoDataframe which temperature and dewpoint will be calculated for.
    start_date : string
        The start date for the event data is being collected for.
    end_date : string
        The end date for the event data is being collected for.
    temp_path : string
        A path to temporarily write out needed geojson files. 

    Returns
    -----------
    float
        The mean Vapor Pressure Deficit value.

    """
    
    # Convert start_date and end_date strings to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Calculate the number of days in the date range
    num_days = (end_date - start_date).days + 1
    
    # If the number of days is less than 1, return None
    if num_days < 1:
        return None, None
    
    # Set the number of days to sample
    num_samples = 7
    
    # If the number of days is greater than the number of samples, evenly 
    # sample dates
    if num_days > num_samples:
        sample_dates = pd.date_range(start=start_date, 
                                     end=end_date, 
                                     periods=num_samples)
    else:
        sample_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Authenticate to Google Earth Engine
    ee.Initialize()

    # Select the ERA5 dataset
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')

    # Convert GeoDataFrame to GeoJSON
    temp_geojson_file = temp_path
    gdf_to_geojson(geometry, temp_geojson_file)

    # Convert GeoJSON to EE geometry
    ee_geometry = geemap.geojson_to_ee(temp_geojson_file)

    # Initialize lists to store mean temperature and mean dewpoint
    mean_temperature_list = []
    mean_dewpoint_list = []
    print("Getting data for {} days".format(len(sample_dates)))
    # Iterate over sampled dates
    for sample_date in sample_dates:
        time_str = sample_date.strftime('%Y-%m-%d') + ' 12:00:00'
        sample_start_date = pd.to_datetime(time_str)
        sample_end_date = sample_start_date + pd.Timedelta(days=1)

        # Filter by date (use the sampled date itself as start and end dates)
        era5_filtered = era5.filterDate(sample_start_date, sample_end_date)
        
        # Filter by date (use the sampled date itself as start and end dates)
        era5_filtered = era5.filterDate(sample_date, sample_end_date)
        era5_filtered_roi = era5_filtered.filterBounds(ee_geometry)

        # Function to calculate mean temperature and mean dewpoint
        def calculate_statistics(image):
            # Calculate mean temperature
            temp_mean = image.reduceRegion(reducer=ee.Reducer.mean(), 
                                           geometry=ee_geometry, 
                                           scale=1000).get('temperature_2m')
            # Calculate mean dewpoint
            dewpoint_mean = image.reduceRegion(reducer=ee.Reducer.mean(), 
                                               geometry=ee_geometry, 
                                               scale=1000).get(
                                                'dewpoint_temperature_2m'
                                                )
            
            return ee.Feature(None, 
                             {'temperature_mean': temp_mean, 
                             'dewpoint_mean': dewpoint_mean})

        # Map the function over the ImageCollection
        statistics = era5_filtered_roi.map(calculate_statistics)

        # Get the list of mean temperature and mean_dewpoint
        mean_temperature = statistics.aggregate_array('temperature_mean')
        mean_dewpoint = statistics.aggregate_array('dewpoint_mean')

        # Calculate the mean for mean temperature and the mean for the mean 
        # dewpoint
        mean_temperature = ee.List(mean_temperature).reduce(ee.Reducer.mean())
        mean_dewpoint = ee.List(mean_dewpoint).reduce(ee.Reducer.mean())

        # Append results to lists
        mean_temperature_list.append(mean_temperature.getInfo())
        mean_dewpoint_list.append(mean_dewpoint.getInfo())
    
    # Convert temperatures from Kelvin to Celsius
    mean_dewpoint_list = [
        value - 273.15 if value is not None else np.nan 
        for value in mean_dewpoint_list
        ]
    mean_temperature_list = [
        value - 273.15 if value is not None else np.nan 
        for value in mean_temperature_list
        ]

    # Calculate the VPD value for each day in the date range. 
    mean_vpd_list = []
    for i, dewpoint in enumerate(mean_dewpoint_list):
        rh = calculate_relative_humidity(mean_temperature_list[i], 
                                         mean_dewpoint_list[i])
        svp = calculate_svp(mean_temperature_list[i])
        mean_vpd_list.append(round(calculate_vpd(svp, rh)/1000, 2))

    mean_vpd = np.nanmean(mean_vpd_list)

    return mean_vpd


print("Opening western ecoregion shape file")
# Creating a western US shape to clip to
western_eco_path = join(
    data_folder, "EPA-ecoregions", "western_us", "western_us_eco_l3.shp"
)
western_eco = gpd.read_file(western_eco_path)

# Opening up EPA level 3 ecoregions
full_eco_path = join(
    data_folder, "EPA-ecoregions", "us_eco_l3", "us_eco_l3.shp"
)

full_eco = gpd.read_file(full_eco_path)

# Creating list of ecoregions of interest
western_forests = ([str(i) for i in list(range(27))] + 
[str(i) for i in list(range(41, 45))] +
[str(i) for i in list(range(77, 82))] 
)

# Selecting only ecoregions in the western US. 
western_epa = full_eco[full_eco['US_L3CODE'].isin(
                                        western_forests)].to_crs(
                                                            western_eco.crs)

# Opening up the overlaid data 
overlay_gdf_path = join(data_folder, 
                        "firedpy_severities_overlay", 
                        "firedpy_severities_overlay.shp")
overlay_gdf = gpd.read_file(overlay_gdf_path)

# Opening up the original fire severity data
full_gdf_path = join(data_folder, 
                     "firedpy_severities", 
                     "firedpy_severities.shp")
full_gdf = gpd.read_file(full_gdf_path)

temp_path = os.path.join(data_folder, 
                         "temp", 
                         "temp.geojson")

# Iterate over rows to calculate statistics for reburn events
outputs = []
for i, row in overlay_gdf.iterrows():
    # Collect all relevant metadata from the geodataframes
    diff_ave, overlay_ave = 0, 0
    over1_ave = 0
    fire1_id, fire2_id = row["id_1"], row["id_2"]
    fire1_year, fire2_year = row["year_1"], row["year_2"]
    fire1_start, fire1_end = row["str_date_1"], row["end_date_1"]
    fire2_start, fire2_end = row["str_date_2"], row["end_date_2"]

    # Find the overlaid and difference areas between the two geometries
    diff, shared = get_diff_overlay_geom(overlay_gdf, 
                                         full_gdf, 
                                         fire1_id, 
                                         fire2_id)
    diff_gdf = gpd.GeoDataFrame({"value":"fill"}, 
                                index=[1], 
                                geometry=[diff.iloc[0]], 
                                crs=full_gdf.crs)

    # Calculate the area for the difference and overlaid areas
    diff_area = cal_area(diff)
    shared_area = cal_area(shared)
    shared_nlcd = shared.to_crs("EPSG:5070")
    diff_nlcd = diff_gdf.to_crs("EPSG:5070")
    diff_geo = diff_gdf.geometry.values[0]
    shared_geo = shared.geometry.values[0]

    # Get the CBI rasters for both fire years
    cbi_tifs = sorted(
        glob(join(data_folder, 
                  "landfire_cbi", 
                  "full_cbi_annual", 
                  "*{}*.tif".format(fire2_year)))
    )
    year_1_cbi_tifs = sorted(
        glob(join(data_folder, 
                  "landfire_cbi", 
                  "full_cbi_annual", 
                  "*{}*.tif".format(fire1_year)))
    )
    
    if len(cbi_tifs) > 0:
        tif = cbi_tifs[0]
        # Calculate the CBI statistics for the non reburned (difference) area
        diff_max_test, diff_ave_test = get_average_max_cbi(fire2_year, 
                                                           diff_geo, 
                                                           tif)
        
        # Calculate the CBI statistics for the reburned (overlaid) area
        overlay_max_test, overlay_ave_test = get_average_max_cbi(fire2_year, 
                                                                 shared_geo, 
                                                                 tif)

        # If both areas have CBI data present
        if diff_max_test > 0 and overlay_max_test > 0:
            diff_max, overlay_max = diff_max_test, overlay_max_test
            diff_ave, overlay_ave = diff_ave_test, overlay_ave_test

            # Get the reburned area's severity for the first fire
            over1_max, over1_ave = get_average_max_cbi(fire1_year, 
                                                       shared_geo, 
                                                       year_1_cbi_tifs[0])
            if not over1_max > 0:
                over1_max, over1_ave = np.nan, np.nan
            # Calculate elevation and slope for the reburned and non reburned 
            # areas
            diff_ele, diff_slope = calc_av_slope_elevation(diff_geo)
            overlay_ele, overlay_slope = calc_av_slope_elevation(shared_geo)

            # Calculate VPD for reburned and non reburned areas
            diff_vpd = get_era5_statistics(diff_gdf, 
                                           fire2_start, 
                                           fire2_end, 
                                           temp_path)
            overlay1_vpd = get_era5_statistics(shared, 
                                               fire1_start, 
                                               fire1_end, 
                                               temp_path)
            overlay2_vpd = get_era5_statistics(shared, 
                                               fire2_start, 
                                               fire2_end, 
                                               temp_path)

            # Check the forest cover for the reburned area during both fires
            fire1_geo = get_geometry(full_gdf, 
                                     fire1_id, 
                                     "id").geometry.values[0]
            fire2_geo = get_geometry(full_gdf, 
                                     fire2_id, 
                                     "id").geometry.values[0]
            shared_nlcd_geo = shared_nlcd.geometry
            overlay1_forest, overlay1_perc = check_forest_cover(fire1_year, 
                                                                shared_nlcd_geo)
            overlay2_forest, overlay2_perc = check_forest_cover(fire2_year, 
                                                                shared_nlcd_geo)
            diff_forest, diff_perc = check_forest_cover(fire2_year, 
                                                        diff_nlcd.geometry)
            
            # Find the ecoregion code for the reburned area
            ecoregion_code = get_ecoregion(western_epa, shared)
            
            # Write all data to a dataframe
            outputs.append(
                pd.DataFrame(
                    {
                        "fire1_id":fire1_id,
                        "fire2_id": fire2_id,
                        "year_1":fire1_year,
                        "year_2": fire2_year,
                        "ave_sev_1":row["ave_sev_1"],
                        "ave_sev_2":row["ave_sev_2"],
                        "diff_av_sev": diff_ave,
                        "reburn_ave_sev":overlay_ave,
                        "reburn_1_ave": over1_ave,
                        "diff_elev": diff_ele,
                        "diff_slope": diff_slope,
                        "diff_vpd": diff_vpd,
                        "reburn1_vpd": overlay1_vpd,
                        "reburn_vpd": overlay2_vpd,
                        "reburn1_forest": overlay1_forest,
                        "reburn_forest": overlay2_forest,
                        "diff_forest": diff_forest,
                        "reburn1_perc": overlay1_perc,
                        "reburn_perc": overlay2_perc,
                        "diff_perc": diff_perc,
                        "reburn_elev": overlay_ele,
                        "reburn_slope": overlay_slope,
                        "ecoregion": ecoregion_code,
                        "year_gap": fire2_year - fire1_year,
                        "diff_area": diff_area,
                        "reburn_area": shared_area,
                        "geometry": diff_geo        
                    },
                    index=[1],
                )
            )

print("Outputting files!")
final_gdf_path = join(data_folder, 
                      "firedpy_overlay_full", 
                      "firedpy_overlay_full_vpd_2.geojson")

# Combine all dataframes and create a final file
main_gdf = pd.concat(outputs)
final_gdf = gpd.GeoDataFrame(main_gdf, 
                             geometry=main_gdf.geometry, 
                             crs="EPSG:4326")
final_gdf = final_gdf.drop_duplicates()

final_gdf.to_file(final_gdf_path)
