#Load required libraries

import xarray as xr
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json 
import os
from esmvaltool.diag_scripts.shared import run_diagnostic, get_cfg, group_metadata
from sklearn import linear_model
import glob
from scipy import signal
import netCDF4
from matplotlib import gridspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import matplotlib as mpl
import random
import seaborn as sns
from scipy import stats
import matplotlib
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from collections import defaultdict

def get_dpm(time, calendar='noleap'):
    """
    Calculate the number of days in each month for a given array of datetime objects,
    based on the specified calendar type.

    Parameters:
    -----------
    time : array-like of datetime objects
        An array of datetime objects representing the dates for which the days per month
        are to be calculated.

    calendar : str, optional (default='standard')
        The type of calendar to use for determining the number of days in each month.
        Supported calendar types include:
        - 'noleap': No leap years, February has 28 days.
        - 'all_leap': All years are leap years, February has 29 days.
        - '366_day': Same as 'all_leap'.
        - '360_day': All months have 30 days.

    Returns:
    --------
    month_length : numpy.ndarray of int
        An array of integers representing the number of days in each month corresponding
        to the input `time`.

    """
    dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
            'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
           '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}
 
    month_length = np.zeros(len(time), dtype=int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
    return month_length

def test_mean_significance(sample_data):
    """
    Perform a one-sample t-test to determine if the mean of the sample data is
    significantly different from zero.

    This function calculates the t-statistic and p-value for the null hypothesis
    that the mean of the sample data is equal to zero. The p-value indicates the
    probability of observing the data (or something more extreme) if the null
    hypothesis is true. 

    Parameters:
    -----------
    sample_data : numpy.ndarray
        A 1-dimensional array of sample data. The data should be numeric and
        represent a random sample from a population.

    Returns:
    --------
    float
        The p-value from the one-sample t-test. A small p-value (typically ≤ 0.05)
        indicates strong evidence against the null hypothesis, suggesting that the
        mean of the sample data is significantly different from zero.
    """        

    # Perform a one-sample t-test
    t_statistic, p_value = stats.ttest_1samp(sample_data, 0)
    
    return p_value

def clim_change(target,period1,period2,box=None,perc=False):
    """
    Calculate the difference in climatological mean between two periods for a specific geographical box.

    This function extracts data from a specific geographical region (defined by `box`), calculates the average 
    over two distinct time periods (`period1` and `period2`), and returns the difference between the two means. 
    It effectively computes how the climate has changed between two periods in a given region.

    Parameters:
    ----------
    target : xarray.DataArray
        The input data array containing the variable of interest (e.g., temperature, precipitation, etc.)
        with dimensions that include 'time', 'lat' (latitude), and 'lon' (longitude).

    period1 : list or tuple of two elements (strings or datetime-like objects)
        Defines the first time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the `target` data (e.g., 'YYYY-MM-DD' or datetime objects).
        
    period2 : list or tuple of two elements (strings or datetime-like objects)
        Defines the second time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the `target` data (e.g., 'YYYY-MM-DD' or datetime objects).

    box : list or tuple of four floats, optional (default=None)
        A list or tuple defining the geographical box [lon_min, lon_max, lat_min, lat_max]. 
        This selects the area of interest based on longitude and latitude ranges. If None, the entire 
        spatial domain of `target` is used.

    perc : bool, optional (default=False)
        If True, the result is expressed as a percentage change between the two periods. 
        If False, the result is expressed as an absolute difference.

    Returns:
    -------
    xarray.DataArray
        A DataArray representing the difference between the climatological means of the two periods for the 
        selected geographical box. The returned data will have the same spatial dimensions (lat, lon) as the 
        input data but reduced in time (since time is averaged over).
    """

    # Select the geographical box if provided
    if box is not None:
        target = target.sel(lon=slice(box[0], box[1]), lat=slice(box[2], box[3]))

    # Calculate the climatological mean for each period
    mean1 = target.sel(time=slice(period1[0], period1[1])).mean(dim='time')
    mean2 = target.sel(time=slice(period2[0], period2[1])).mean(dim='time')

    # Calculate the difference or percentage change
    if perc:
        output = ((mean2 - mean1) / mean1) * 100
    else:
        output = mean2 - mean1

    return output

def clim_change_seas(target,period1,period2,box=None,perc=False):
    """
    Calculate the difference in climatological mean between two periods for a specific geographical box for each season

    This function extracts data from a specific geographical region (defined by `box`), calculates the average 
    over two distinct time periods (`period1` and `period2`), and returns the difference between the two means for each 
    season (DJF, MAM, JJA, SON). It effectively computes how the climate has changed between two periods in a given region for each 
    season.

    Parameters:
    ----------
    target : xarray.DataArray
        The input data array containing the variable of interest (e.g., temperature, precipitation, etc.)
        with dimensions that include 'time', 'lat' (latitude), and 'lon' (longitude).
    
    period1 : list or tuple of two elements (strings or datetime-like objects)
        Defines the first time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the `target` data (e.g., 'YYYY-MM-DD' or datetime objects).
        
    period2 : list or tuple of two elements (strings or datetime-like objects)
        Defines the second time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the `target` data (e.g., 'YYYY-MM-DD' or datetime objects).

    box : list or tuple of four floats, optional (default=None)
        A list or tuple defining the geographical box [lon_min, lon_max, lat_min, lat_max]. 
        This selects the area of interest based on longitude and latitude ranges.
        If None, the entire spatial domain of `target` is used.

    perc : bool, optional (default=False)
        If True, the result is expressed as a percentage change between the two periods. 
        If False, the result is expressed as an absolute difference.

    Returns:
    -------
    tuple of xarray.DataArray
        A tuple containing four DataArrays representing the difference between the climatological means of the two periods 
        for the selected geographical box for each season:
        - output_djf: Difference for December-January-February (DJF).
        - output_mam: Difference for March-April-May (MAM).
        - output_jja: Difference for June-July-August (JJA).
        - output_son: Difference for September-October-November (SON).
    """

    # Compute seasonal means
    month_length = xr.DataArray(get_dpm(target.time.to_index(), calendar='noleap'),
                                coords=[target.time], name='month_length')

    target_season = ((target * month_length).resample(time='QS-DEC').sum() / 
                     month_length.resample(time='QS-DEC').sum())

    # Select seasons
    target_djf = target_season.sel(time=target_season.time.dt.month == 12)
    target_mam = target_season.sel(time=target_season.time.dt.month == 3)
    target_jja = target_season.sel(time=target_season.time.dt.month == 6)
    target_son = target_season.sel(time=target_season.time.dt.month == 9)

    # Apply geographical box selection if provided
    if box is not None:
        target_djf = target_djf.sel(lon=slice(box[0], box[1]), lat=slice(box[2], box[3]))
        target_mam = target_mam.sel(lon=slice(box[0], box[1]), lat=slice(box[2], box[3]))
        target_jja = target_jja.sel(lon=slice(box[0], box[1]), lat=slice(box[2], box[3]))
        target_son = target_son.sel(lon=slice(box[0], box[1]), lat=slice(box[2], box[3]))

    # Calculate the difference or percentage change for each season
    def calculate_change(season_data):
        mean1 = season_data.sel(time=slice(period1[0], period1[1])).mean(dim='time')
        mean2 = season_data.sel(time=slice(period2[0], period2[1])).mean(dim='time')
        if perc:
            return ((mean2 - mean1) / mean1) * 100
        else:
            return mean2 - mean1

    output_djf = calculate_change(target_djf)
    output_mam = calculate_change(target_mam)
    output_jja = calculate_change(target_jja)
    output_son = calculate_change(target_son)

    return output_djf, output_mam, output_jja, output_son

def MultiModelMean(var_change_sim,level_agreement):
    """
    Calculate the multi-model mean and sign agreement for each experiment.

    This function computes the mean of the multi-member mean differences across models/experiments and checks 
    if there is a significant agreement in the sign of the changes above a specified level of agreement.

    Parameters:
    ----------
    var_change_sim : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        multi-member mean difference between the climatological means of two periods for a selected 
        geographical box. The arrays should have the same spatial dimensions.

    level_agreement : float
        The threshold for sign agreement, expressed as a fraction (e.g., 0.8 for 80% agreement). 
        A value of 0.8 means that at least 80% of the models must agree on the sign (positive or negative) 
        of the change for a given grid point.

    Returns:
    -------
    mean_by_exp_var : dict
        A dictionary where keys are experiment names and values are arrays representing the multi-model mean 
        for each experiment. The arrays have the same spatial dimensions as the input arrays.

    sign_agreement_by_exp_var : dict
        A dictionary where keys are experiment names and values are boolean arrays indicating whether there 
        is significant agreement in the sign of the changes across models. The arrays have the same spatial 
        dimensions as the input arrays.


    """

    mean_by_exp_var = {}
    sign_agreement_by_exp_var = {}

    # Step 1: Group arrays by 'exp'
    grouped_by_exp_var = defaultdict(list)
    for (exp, dataset), array in var_change_sim.items():
        grouped_by_exp_var[exp].append(array)

    # Step 2: Compute the mean and sign agreement for each 'exp'
    for exp, arrays in grouped_by_exp_var.items():
        # Stack all arrays along a new axis
        stacked_arrays_var = np.stack(arrays, axis=0)  # Shape: (n_datasets, rows, cols)
        
        # Compute the mean across datasets
        mean_by_exp_var[exp] = np.nanmean(stacked_arrays_var, axis=0)
        
        # Compute sign agreement
        positive_signs = np.sum(stacked_arrays_var > 0, axis=0)
        total_datasets = np.sum(~np.isnan(stacked_arrays_var), axis=0)
        
        # Check if the fraction of models with the same sign meets the level of agreement
        sign_agreement = ((positive_signs / total_datasets) >= level_agreement) | ((positive_signs / total_datasets) <= (1-level_agreement))
        
        sign_agreement_by_exp_var[exp] = sign_agreement

    return mean_by_exp_var, sign_agreement_by_exp_var

def var_multimember_change(meta_dataset, meta_exp, meta, var_name, conv_unit, period1, period2, box=None, perc=False):
    """
    Calculate the multi-member mean difference in climatological mean between two periods for all models.

    This function computes the mean difference in a variable (e.g., temperature, precipitation) between two 
    time periods for each model and experiment. It also calculates the p-value (significance) of the difference 
    and returns the latitudes and longitudes for each dataset.

    Parameters:
    ----------
    meta_dataset : dict
        A dictionary where keys are dataset names and values are lists of metadata for each dataset.

    meta_exp : dict
        A dictionary where keys are experiment names and values are lists of metadata for each experiment.

    meta : dict
        A dictionary where keys are alias names and values are lists of metadata for each alias.

    var_name : str
        The name of the variable as specified in the recipe.

    conv_unit : int or float
        A conversion factor to apply to the variable (e.g., to convert units).

    period1 : list or tuple of two elements (strings or datetime-like objects)
        Defines the first time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the input data (e.g., 'YYYY-MM-DD' or datetime objects).

    period2 : list or tuple of two elements (strings or datetime-like objects)
        Defines the second time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the input data (e.g., 'YYYY-MM-DD' or datetime objects).

    box : list or tuple of four floats, optional (default=None)
        A list or tuple defining the geographical box [lon_min, lon_max, lat_min, lat_max]. 
        This selects the area of interest based on longitude and latitude ranges. If None, the entire 
        spatial domain of the input data is used.

    perc : bool, optional (default=False)
        If True, the result is expressed as a percentage change between the two periods. 
        If False, the result is expressed as an absolute difference.

    Returns:
    -------
    var_change_sim : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        multi-member mean difference between the climatological means of the two periods for the selected 
        geographical box.

    var_pval_sim : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        p-value (significance) of the multi-member mean difference.

    var_lat_sim : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        latitudes for each dataset.

    var_lon_sim : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        longitudes for each dataset.
    """
    var_change_sim = {}
    var_pval_sim= {}
    var_lat_sim= {}
    var_lon_sim = {}

    for exp, exp_list in meta_exp.items():

        for dataset, dataset_list in meta_dataset.items():
            # Lists to store  variable changes for each model
            model_var_changes = []

            print(f"Evaluating for {dataset} - {exp}\n")

            for alias, alias_list in meta.items():
                target_var = {}

                for m in alias_list:
                    if m["dataset"] == dataset and m["exp"] == exp:
                        if os.path.exists(m["filename"]):  # Check if the file exists
                            if m["variable_group"] == var_name:
                                target_var[m["variable_group"]] = xr.open_dataset(m["filename"])[m["short_name"]]

                try:
                    if var_name in target_var:
                        target_var = target_var[var_name] * conv_unit  # Convert units 

                    # Compute changes in variable between periods
                    target_var_change = clim_change(target_var, period1, period2, box, perc)
                    # Append changes to the model-specific lists
                    model_var_changes.append(target_var_change)

                except KeyError as e:
                    print(f"KeyError: {e}, skipping model {alias}\n")
                except Exception as e:
                    print(f"Error processing file {m['filename']}: {e}\n")

            # Combine all members of the dataset into xarray DataArrays
            if model_var_changes:
                model_var_changes_ar = xr.concat(model_var_changes, dim='ensemble')
                model_var_mean = model_var_changes_ar.mean(dim='ensemble')
                model_maps_var_pval = xr.apply_ufunc(
                    test_mean_significance, model_var_changes_ar,
                    input_core_dims=[["ensemble"]], output_core_dims=[[]],
                    vectorize=True, dask="parallelized"
                )

                var_change_sim[(exp, dataset)] = model_var_mean
                var_pval_sim[(exp, dataset)] = model_maps_var_pval
                var_lat_sim[(exp, dataset)] = model_var_mean.coords['lat']
                var_lon_sim[(exp, dataset)] = model_var_mean.coords['lon']

    return var_change_sim, var_pval_sim, var_lat_sim, var_lon_sim

def var_multimember_change_seas(meta_dataset, meta_exp, meta, var_name, conv_unit, period1, period2, box=None, perc=False):
    """
    Calculate the multi-member mean difference in climatological mean between two periods for all models, 
    for each season (DJF, MAM, JJA, SON).

    This function computes the mean difference in a variable (e.g., temperature, precipitation) between two 
    time periods for each model and experiment, broken down by season. It also calculates the p-value 
    (significance) of the difference and returns the latitudes and longitudes for each dataset.

    Parameters:
    ----------
    meta_dataset : dict
        A dictionary where keys are dataset names and values are lists of metadata for each dataset.

    meta_exp : dict
        A dictionary where keys are experiment names and values are lists of metadata for each experiment.

    meta : dict
        A dictionary where keys are alias names and values are lists of metadata for each alias.

    var_name : str
        The name of the variable as specified in the recipe.

    conv_unit : int or float
        A conversion factor to apply to the variable (e.g., to convert units).

    period1 : list or tuple of two elements (strings or datetime-like objects)
        Defines the first time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the input data (e.g., 'YYYY-MM-DD' or datetime objects).

    period2 : list or tuple of two elements (strings or datetime-like objects)
        Defines the second time period to be averaged, in the form [start_time, end_time].
        The format should match the time format in the input data (e.g., 'YYYY-MM-DD' or datetime objects).

    box : list or tuple of four floats, optional (default=None)
        A list or tuple defining the geographical box [lon_min, lon_max, lat_min, lat_max]. 
        This selects the area of interest based on longitude and latitude ranges. If None, the entire 
        spatial domain of the input data is used.

    perc : bool, optional (default=False)
        If True, the result is expressed as a percentage change between the two periods. 
        If False, the result is expressed as an absolute difference.

    Returns:
    -------
    var_change_sim_djf : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        multi-member mean difference for December-January-February (DJF).

    var_pval_sim_djf : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        p-value (significance) of the multi-member mean difference for DJF.

    var_change_sim_mam : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        multi-member mean difference for March-April-May (MAM).

    var_pval_sim_mam : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        p-value (significance) of the multi-member mean difference for MAM.

    var_change_sim_jja : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        multi-member mean difference for June-July-August (JJA).

    var_pval_sim_jja : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        p-value (significance) of the multi-member mean difference for JJA.

    var_change_sim_son : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        multi-member mean difference for September-October-November (SON).

    var_pval_sim_son : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        p-value (significance) of the multi-member mean difference for SON.

    var_lat_sim : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        latitudes for each dataset.

    var_lon_sim : dict
        A dictionary where keys are tuples of (experiment, dataset) and values are arrays representing the 
        longitudes for each dataset.

    """
    var_change_sim_djf = {}
    var_pval_sim_djf= {}
    var_change_sim_mam = {}
    var_pval_sim_mam= {}
    var_change_sim_jja = {}
    var_pval_sim_jja= {}
    var_change_sim_son = {}
    var_pval_sim_son= {}           
    var_lat_sim = {}
    var_lon_sim = {}

    for exp, exp_list in meta_exp.items():
        for dataset, dataset_list in meta_dataset.items():
            # Lists to store  variable changes for each model
            model_var_changes_djf = []
            model_var_changes_mam = []
            model_var_changes_jja = []
            model_var_changes_son = []            

            print(f"Evaluating for {dataset} - {exp}\n")

            for alias, alias_list in meta.items():
                target_var = {}

                for m in alias_list:
                    if m["dataset"] == dataset and m["exp"] == exp:
                        if os.path.exists(m["filename"]):  # Check if the file exists
                            if m["variable_group"] == var_name:
                                target_var[m["variable_group"]] = xr.open_dataset(m["filename"])[m["short_name"]]

                try:
                    if var_name in target_var:
                        target_var = target_var[var_name] * conv_unit  # Convert units 

                    print(f"Computing climatological change in model {alias}\n")
                    # Compute changes in variable between periods
                    if perc:
                        target_var_change_djf,target_var_change_mam,target_var_change_jja,target_var_change_son = clim_change_seas(target_var, period1, period2, box, True)
                    else:
                        target_var_change_djf,target_var_change_mam,target_var_change_jja,target_var_change_son = clim_change_seas(target_var, period1, period2, box)

                    # Append changes to the model-specific lists
                    model_var_changes_djf.append(target_var_change_djf)
                    model_var_changes_mam.append(target_var_change_mam)                   
                    model_var_changes_jja.append(target_var_change_jja)
                    model_var_changes_son.append(target_var_change_son)                    
                    
                except KeyError as e:
                    print(f"KeyError: {e}, skipping model {alias}\n")
                except Exception as e:
                    print(f"Error processing file {m['filename']}: {e}\n")

                # Combine all members of the dataset into xarray DataArrays
            if model_var_changes_djf:
                model_var_changes_ar_djf = xr.concat(model_var_changes_djf, dim='ensemble')
                model_var_mean_djf = model_var_changes_ar_djf.mean(dim='ensemble')
                model_maps_var_pval_djf = xr.apply_ufunc(
                    test_mean_significance, model_var_changes_ar_djf,
                    input_core_dims=[["ensemble"]], output_core_dims=[[]],
                    vectorize=True, dask="parallelized"
                )

                var_change_sim_djf[(exp, dataset)] = model_var_mean_djf
                var_pval_sim_djf[(exp, dataset)] = model_maps_var_pval_djf
                var_lat_sim[(exp, dataset)] = model_var_mean_djf.coords['lat']
                var_lon_sim[(exp, dataset)] = model_var_mean_djf.coords['lon']
                
                                # Combine all members of the dataset into xarray DataArrays
            if model_var_changes_mam:
                model_var_changes_ar_mam = xr.concat(model_var_changes_mam, dim='ensemble')
                model_var_mean_mam = model_var_changes_ar_mam.mean(dim='ensemble')
                model_maps_var_pval_mam = xr.apply_ufunc(
                    test_mean_significance, model_var_changes_ar_mam,
                    input_core_dims=[["ensemble"]], output_core_dims=[[]],
                    vectorize=True, dask="parallelized"
                )

                var_change_sim_mam[(exp, dataset)] = model_var_mean_mam
                var_pval_sim_mam[(exp, dataset)] = model_maps_var_pval_mam
                
                # Combine all members of the dataset into xarray DataArrays
            if model_var_changes_jja:
                model_var_changes_ar_jja = xr.concat(model_var_changes_jja, dim='ensemble')
                model_var_mean_jja = model_var_changes_ar_jja.mean(dim='ensemble')
                model_maps_var_pval_jja = xr.apply_ufunc(
                    test_mean_significance, model_var_changes_ar_jja,
                    input_core_dims=[["ensemble"]], output_core_dims=[[]],
                    vectorize=True, dask="parallelized"
                )

                var_change_sim_jja[(exp, dataset)] = model_var_mean_jja
                var_pval_sim_jja[(exp, dataset)] = model_maps_var_pval_jja            
                
                
                # Combine all members of the dataset into xarray DataArrays
            if model_var_changes_son:
                model_var_changes_ar_son = xr.concat(model_var_changes_son, dim='ensemble')
                model_var_mean_son = model_var_changes_ar_son.mean(dim='ensemble')
                model_maps_var_pval_son = xr.apply_ufunc(
                    test_mean_significance, model_var_changes_ar_son,
                    input_core_dims=[["ensemble"]], output_core_dims=[[]],
                    vectorize=True, dask="parallelized"
                )

                var_change_sim_son[(exp, dataset)] = model_var_mean_son
                var_pval_sim_son[(exp, dataset)] = model_maps_var_pval_son 

    return var_change_sim_djf,var_pval_sim_djf,var_change_sim_mam,var_pval_sim_mam,var_change_sim_jja,var_pval_sim_jja,var_change_sim_son,var_pval_sim_son,var_lat_sim,var_lon_sim

def plot_allmodels_plate(var_change_sim, var_pval_sim, var_lat_sim, var_lon_sim, lista_exp, lista_model, var_name, var_unit, levels, colors, LATMIN, LATMAX, LONMIN, LONMAX, P1_IY, P1_FY, P2_IY, P2_FY, seas=None):
    """
    Plot the multi-member mean difference in climatological mean between two periods for all models/experiments.

    This function generates a grid of subplots, where each subplot represents the difference in climatological mean 
    between two periods for a specific model and experiment. The plots are displayed on a PlateCarree projection 
    and include features like coastlines, borders, and gridlines.

    Parameters:
    ----------
    var_change_sim : dict
        A dictionary where keys are tuples of (experiment, model) and values are arrays representing the 
        multi-member mean difference between the climatological means of the two periods.

    var_pval_sim : dict
        A dictionary where keys are tuples of (experiment, model) and values are arrays representing the 
        p-value (significance) of the multi-member mean difference.

    var_lat_sim : dict
        A dictionary where keys are tuples of (experiment, model) and values are arrays representing the 
        latitudes for each dataset.

    var_lon_sim : dict
        A dictionary where keys are tuples of (experiment, model) and values are arrays representing the 
        longitudes for each dataset.

    lista_exp : list
        A list of experiment names to be plotted.

    lista_model : list
        A list of model names to be plotted.

    var_name : str
        The name of the variable for the plot title.

    var_unit : str
        The unit of the variable for the plot title.

    levels : list
        A list of levels for the contour plot.

    colors : list
        A list of colors for the contour plot.


    LATMIN : float
        The minimum latitude for the plot extent.

    LATMAX : float
        The maximum latitude for the plot extent.

    LONMIN : float
        The minimum longitude for the plot extent.

    LONMAX : float
        The maximum longitude for the plot extent.

    P1_IY : str
        The initial year of the first period.

    P1_FY : str
        The final year of the first period.

    P2_IY : str
        The initial year of the second period.

    P2_FY : str
        The final year of the second period.

    seas : str, optional (default=None)
        The season to be included in the plot title and filename. If None, the plot is assumed to represent annual data.

    """

    #Define figure
    fig = plt.figure(figsize=(4.5*len(lista_exp),1.4*len(lista_model)))
     #Define grid for subplots
    gs = gridspec.GridSpec(len(lista_model),len(lista_exp),figure=fig, width_ratios=4.5*np.ones(len(lista_exp)),height_ratios=1.4*np.ones(len(lista_model)))     

    # Create colormap and normalization
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors,extend='both')

    # Make one subplot for each experiment and model
    for i in range(len(lista_exp)):
        for j in range(len(lista_model)):
            # Check if lon_sim and lat_sim exist for the given experiment and model
            if (lista_exp[i], lista_model[j]) in var_lon_sim and (lista_exp[i], lista_model[j]) in var_lat_sim:
                #Subplot position and define projection to be used
                ax = fig.add_subplot(gs[i+j*(len(lista_exp))],projection=ccrs.PlateCarree(central_longitude=180))
                #Grid for latitudes/longitudes for that model
                lons, lats = np.meshgrid(var_lon_sim[(lista_exp[i],lista_model[j])], var_lat_sim[(lista_exp[i],lista_model[j])])
                #Define projected lat/lon
                crs_latlon = ccrs.PlateCarree()
                #Define lat/lon plot limits
                ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
                #Plot filled contours of mean differences                
                im=ax.contourf(lons, lats, var_change_sim[(lista_exp[i],lista_model[j])], levels, cmap=cmap, transform=crs_latlon, extend='both')
                #Add map fearures like coasts, border
                ax.add_feature(cartopy.feature.COASTLINE)
                ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
                #Add map fearures like coasts, border
                ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
                #Define x and y ticks format
                ax.set_xticks(np.linspace(LONMIN, LONMAX,  4), crs=crs_latlon)
                ax.set_yticks(np.linspace(LATMIN, LATMAX,  6), crs=crs_latlon)
                ax.tick_params(axis='both', which='major', labelsize=6)
                lon_formatter = LongitudeFormatter(zero_direction_label=True)
                lat_formatter = LatitudeFormatter()
                ax.xaxis.set_major_formatter(lon_formatter)
                ax.yaxis.set_major_formatter(lat_formatter)
                #Add subplot title                       
                ax.set_title(lista_model[j] + ' ' + lista_exp[i], fontsize="8") 
            
    # Define subplots arrangement
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.91,top=0.93, wspace=0.1 ,hspace=0.35)   

    #Plot colorbar
    cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax,orientation='vertical')

    # Add main title for the figure
    if seas is None:
        plt.figtext(0.4, .97, f"{var_name} ({var_unit}) change {P2_IY}-{P2_FY} - {P1_IY}-{P1_FY}", fontsize=16)
        # Save figure
        fig.savefig(f"{config['plot_dir']}/target_change/ensemble_mean_plate_{var_name}_change_{P2_IY}-{P2_FY}_{P1_IY}-{P1_FY}.png", dpi=150, bbox_inches='tight')
    else:
        plt.figtext(0.4, .97, f"{var_name} ({var_unit}) {seas} change {P2_IY}-{P2_FY} - {P1_IY}-{P1_FY}", fontsize=16)
        # Save figure
        fig.savefig(f"{config['plot_dir']}/target_change/ensemble_mean_plate_{var_name}_{seas}_change_{P2_IY}-{P2_FY}_{P1_IY}-{P1_FY}.png", dpi=150, bbox_inches='tight')

def plot_allmodels_zonalmean(var_change_sim, var_lat_sim, lista_exp, lista_model, var_name, var_unit, colors, LATMIN, LATMAX, P1_IY, P1_FY, P2_IY, P2_FY, YMIN, YMAX, seas=None):    
    """
    Plot the multi-member mean difference in climatological mean between two periods for all models and experiments.

    This function generates a figure with subplots, where each subplot corresponds to a model. For each model, the function
    plots the zonal mean difference in the climatological mean of a variable between two periods (P2 and P1) for all experiments.
    The plots are arranged in a grid, with one row for every two models.

    Parameters:
    ----------
    var_change_sim : dict
        A dictionary containing the multi-member mean difference in the climatological mean of the variable between two periods
        (P2 and P1) for all models and experiments. The keys are tuples of (experiment, model), and the values are arrays
        representing the zonal mean differences.

    var_lat_sim : dict
        A dictionary containing the latitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of latitude values.

    lista_exp : list of str
        A list of experiment names to be plotted.

    lista_model : list of str
        A list of model names to be plotted.

    var_name : str
        The name of the variable being plotted (e.g., "Temperature"). This is used for titles and labels.

    var_unit : str
        The unit of the variable (e.g., "ºC" for Celsius). This is used for titles and labels.

    colors : list of str
        A list of colors corresponding to each experiment. The colors are used to differentiate experiments in the plots.

    LATMIN : float
        The minimum latitude value for the x-axis of the plots.

    LATMAX : float
        The maximum latitude value for the x-axis of the plots.

    P1_IY : str
        The initial year of the first period (P1).

    P1_FY : str
        The final year of the first period (P1).

    P2_IY : str
        The initial year of the second period (P2).

    P2_FY : str
        The final year of the second period (P2).

    YMIN : float
        The minimum value for the y-axis of the plots.

    YMAX : float
        The maximum value for the y-axis of the plots.
    
    seas : str, optional
        The season to be plotted (e.g., "DJF" for December-January-February). If not provided (default is None), the plot
        will represent annual means.    

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the directory specified by `config["plot_dir"]`.
        
    """

    #Plot figure for all experiments and models
    #Define figure
    fig = plt.figure(figsize=(4.5*len(lista_model)/2,5*2))

    #Define grid for subplots
    gs = gridspec.GridSpec(2,int(len(lista_model)/2),figure=fig,width_ratios=4.5*np.ones(int(len(lista_model)/2)),height_ratios=5*np.ones(2))   

    # Make one subplot for each model
    for j in range(len(lista_model)):
        if any((lista_exp[i], lista_model[j]) in var_lat_sim and (lista_exp[i], lista_model[j]) in var_change_sim for i in range(len(lista_exp))):
            ax = fig.add_subplot(gs[j])  # Create a subplot for each model
            for i in range(len(lista_exp)):
                if (lista_exp[i], lista_model[j]) in var_lat_sim and (lista_exp[i], lista_model[j]) in var_change_sim:
                    # Plot diff with label
                    label = f'{lista_exp[i]}'
                    ax.plot(
                        var_lat_sim[(lista_exp[i], lista_model[j])], 
                        var_change_sim[(lista_exp[i], lista_model[j])].mean(dim='lon'),
                        label=label,
                        color=colors[i]
                    )  

        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        # Add legend to the current subplot
        ax.legend()
        ax.set_xlim([LATMIN, LATMAX])
        ax.set_ylim([YMIN, YMAX])
        # Optional: Add titles or labels if needed
        ax.set_title(f'Model: {lista_model[j]}', fontsize="12")
        ax.set_xlabel('Latitude', fontsize="8")
        ax.set_ylabel(var_name+' ('+var_unit+') change')
	
    # Define subplots arrangement
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.97,top=0.9, wspace=0.2 ,hspace=0.2)   

    if seas == None:
        #Add main title for the figure
        plt.figtext(0.4,.95, var_name+' ('+var_unit+') change '+P2_IY+'-'+P2_FY+' - '+P1_IY+'-'+P1_FY, fontsize=14)
        #Save figure
        fig.savefig(config["plot_dir"]+"/target_change/ensemble_mean_zonalmean_"+var_name+"_change_"+P2_IY+'-'+P2_FY+'_'+P1_IY+'-'+P1_FY+".png", dpi=150, bbox_inches='tight')  
    else:
        #Add main title for the figure
        plt.figtext(0.4,.95, var_name+' ('+var_unit+' ) '+ seas +' change '+P2_IY+'-'+P2_FY+' - '+P1_IY+'-'+P1_FY, fontsize=14)
        #Save figure
        fig.savefig(config["plot_dir"]+"/target_change/ensemble_mean_zonalmean_"+var_name+"_"+seas+"_change_"+P2_IY+'-'+P2_FY+'_'+P1_IY+'-'+P1_FY+".png", dpi=150, bbox_inches='tight')

def plot_multimodelmean_plate(mean_by_exp_var,sign_agreement_by_exp_var,var_lat_sim,var_lon_sim,lista_exp,lista_model,var_name,var_unit,levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY):
    """
    Plot the multi-model mean difference in climatological mean between two periods for all experiments.

    This function generates a figure with subplots, where each subplot corresponds to an experiment. For each experiment,
    it plots the multi-model mean difference in the climatological mean of a variable between two periods (P2 and P1).
    The plot includes contour fill for the mean difference and stippling to indicate regions where a significant fraction
    of models agree on the sign of the change.

    Parameters:
    ----------
    mean_by_exp_var : dict
        A dictionary containing the multi-model mean difference in the climatological mean of the variable between two periods
        (P2 and P1) for each experiment. The keys are experiment names, and the values are 2D arrays representing the mean
        differences.

    sign_agreement_by_exp_var : dict
        A dictionary containing the fraction of models that agree on the sign of the change for each experiment. The keys are
        experiment names, and the values are 2D arrays with values between 0 and 1, where values >= 0.5 indicate agreement.

    var_lat_sim : dict
        A dictionary containing the latitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of latitude values.

    var_lon_sim : dict
        A dictionary containing the longitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of longitude values.

    lista_exp : list of str
        A list of experiment names to be plotted.

    lista_model : list of str
        A list of model names used to calculate the multi-model mean.

    var_name : str
        The name of the variable being plotted (e.g., "Temperature"). This is used for titles and labels.

    var_unit : str
        The unit of the variable (e.g., "K" for Kelvin). This is used for titles and labels.

    levels : list of float
        A list of levels for the contour fill plot. Defines the boundaries for the color map.

    colors : list of str
        A list of colors corresponding to the levels. Used to create the color map.

    LATMIN : float
        The minimum latitude value for the plot.

    LATMAX : float
        The maximum latitude value for the plot.

    LONMIN : float
        The minimum longitude value for the plot.

    LONMAX : float
        The maximum longitude value for the plot.

    P1_IY : str
        The initial year of the first period (P1).

    P1_FY : str
        The final year of the first period (P1).

    P2_IY : str
        The initial year of the second period (P2).

    P2_FY : str
        The final year of the second period (P2).

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the directory specified by `config["plot_dir"]`.

    """

    #Define figure
    fig = plt.figure(figsize=(4.5*len(lista_exp),2))

    #Define grid for subplots
    gs = gridspec.GridSpec(1,len(lista_exp),figure=fig, width_ratios=4.5*np.ones(len(lista_exp)),height_ratios=[2])    

    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors,extend='both')

    # Make one subplot for each experiment and model
    for i in range(len(lista_exp)):      
        #Subplot position and define projection to be used
        ax = fig.add_subplot(gs[i],projection=ccrs.PlateCarree(central_longitude=180))

        #Grid for latitudes/longitudes for that model
        lons, lats = np.meshgrid(var_lon_sim[(lista_exp[0],lista_model[0])], var_lat_sim[(lista_exp[0],lista_model[0])])

        #Define projected lat/lon
        crs_latlon = ccrs.PlateCarree()

        #Define lat/lon plot limits
        ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)

        # Plot filled contours of mean differencesç
        im=ax.contourf(lons, lats, mean_by_exp_var[lista_exp[i]], levels, cmap=cmap, transform=crs_latlon, extend='both')

        # Add stippling where sign agreement is above threshold
        ax.contourf(lons, lats, sign_agreement_by_exp_var[lista_exp[i]], levels=[0.5, 1.5], hatches=['...'], colors='none', transform=crs_latlon)
        
        #Add map fearures like coasts, border
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)

        #Add map fearures like coasts, border
        ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')

        #Define x and y ticks format
        ax.set_xticks(np.linspace(LONMIN, LONMAX,  4), crs=crs_latlon)
        ax.set_yticks(np.linspace(LATMIN, LATMAX,  6), crs=crs_latlon)
        ax.tick_params(axis='both', which='major', labelsize=6)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        #Add subplot title                       
        ax.set_title(lista_exp[i], fontsize="9") 
     
    # Define subplots arrangement
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.91,top=0.93, wspace=0.1 ,hspace=0.35)   

    #Plot colorbar
    cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax,orientation='vertical')

    # Add main title for the figure
    plt.figtext(0.4, 0.97, f'MMM {var_name} ({var_unit}) change {P2_IY}-{P2_FY} - {P1_IY}-{P1_FY}', fontsize=16)

    # Save figure
    fig.savefig(
        f"{config['plot_dir']}/target_change/ensemble_MME_mean_plate_{var_name}_change_{P2_IY}-{P2_FY}_{P1_IY}-{P1_FY}.png",
        dpi=150,
        bbox_inches='tight'
    )

def plot_multimodelmean_plate_seas(mean_by_exp_var_seas,sign_agreement_by_exp_var_seas,var_lat_sim,var_lon_sim,lista_exp,lista_model,var_name,var_unit,levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY):
    """
    Plot the multi-model mean difference in climatological mean between two periods for all experiments, for the four seasons.

    This function generates a figure with subplots, where each row corresponds to a season (DJF, MAM, JJA, SON) and each column
    corresponds to an experiment. For each experiment and season, it plots the multi-model mean difference in the climatological
    mean of a variable between two periods (P2 and P1). The plot includes contour fill for the mean difference and stippling to
    indicate regions where a significant fraction of models agree on the sign of the change.

    Parameters:
    ----------
    mean_by_exp_var_seas : list of dict
        A list of dictionaries containing the multi-model mean difference in the climatological mean of the variable between two
        periods (P2 and P1) for each experiment and season. Each dictionary corresponds to a season (DJF, MAM, JJA, SON), and the
        keys are experiment names. The values are 2D arrays representing the mean differences.

    sign_agreement_by_exp_var_seas : list of dict
        A list of dictionaries containing the fraction of models that agree on the sign of the change for each experiment and
        season. Each dictionary corresponds to a season (DJF, MAM, JJA, SON), and the keys are experiment names. The values are
        2D arrays with values between 0 and 1, where values >= 0.5 indicate agreement.

    var_lat_sim : dict
        A dictionary containing the latitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of latitude values.

    var_lon_sim : dict
        A dictionary containing the longitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of longitude values.

    lista_exp : list of str
        A list of experiment names to be plotted.

    lista_model : list of str
        A list of model names used to calculate the multi-model mean.

    var_name : str
        The name of the variable being plotted (e.g., "Temperature"). This is used for titles and labels.

    var_unit : str
        The unit of the variable (e.g., "K" for Kelvin). This is used for titles and labels.

    levels : list of float
        A list of levels for the contour fill plot. Defines the boundaries for the color map.

    colors : list of str
        A list of colors corresponding to the levels. Used to create the color map.

    LATMIN : float
        The minimum latitude value for the plot.

    LATMAX : float
        The maximum latitude value for the plot.

    LONMIN : float
        The minimum longitude value for the plot.

    LONMAX : float
        The maximum longitude value for the plot.

    P1_IY : str
        The initial year of the first period (P1).

    P1_FY : str
        The final year of the first period (P1).

    P2_IY : str
        The initial year of the second period (P2).

    P2_FY : str
        The final year of the second period (P2).

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the directory specified by `config["plot_dir"]`.
    
    """

    # Define seasons
    SEAS=['DJF','MAM','JJA','SON']

    # Define figure size based on the number of experiments and seasons
    fig = plt.figure(figsize=(4.5*len(lista_exp),2*len(SEAS)))

    #Define grid for subplots
    gs = gridspec.GridSpec(len(SEAS),len(lista_exp),figure=fig, width_ratios=4.5*np.ones(len(lista_exp)),height_ratios=[2,2,2,2])    

    # Create colormap and normalization
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors,extend='both')

    # Make one subplot for each experiment and season
    for i in range(len(lista_exp)):      
        for j in range(len(SEAS)):      
            # Subplot position and define projection
            ax = fig.add_subplot(gs[j,i],projection=ccrs.PlateCarree(central_longitude=180))

            # Grid for latitudes/longitudes for the first model (assumed same for all models)
            lons, lats = np.meshgrid(var_lon_sim[(lista_exp[0],lista_model[0])], var_lat_sim[(lista_exp[0],lista_model[0])])

            #Define projected lat/lon
            crs_latlon = ccrs.PlateCarree()

            #Define lat/lon plot limits
            ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)

            # Plot filled contours of mean differences
            im=ax.contourf(lons, lats, mean_by_exp_var_seas[j][lista_exp[i]], levels, cmap=cmap, transform=crs_latlon, extend='both')

            # Add stippling where sign agreement is above the threshold
            ax.contourf(lons, lats, sign_agreement_by_exp_var_seas[j][lista_exp[i]], levels=[0.5, 1.5], hatches=['...'], colors='none', transform=crs_latlon)
            
            # Add map features (coastlines and borders)
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)

            # Add grid lines
            ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')

            #Define x and y ticks format
            ax.set_xticks(np.linspace(LONMIN, LONMAX,  4), crs=crs_latlon)
            ax.set_yticks(np.linspace(LATMIN, LATMAX,  6), crs=crs_latlon)
            ax.tick_params(axis='both', which='major', labelsize=6)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)

            #Add subplot title                       
            ax.set_title(f'{SEAS[j]} {lista_exp[i]}', fontsize="7")
                 
    # Define subplots arrangement
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.91,top=0.93, wspace=0.1 ,hspace=0.35)   

    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax,orientation='vertical')

    # Add main title for the figure
    plt.figtext(0.4, 0.97, f'MMM {var_name} ({var_unit}) change {P2_IY}-{P2_FY} - {P1_IY}-{P1_FY}', fontsize=16)

    # Save figure
    fig.savefig(
        f"{config['plot_dir']}/target_change/ensemble_MME_mean_plate_seas_{var_name}_change_{P2_IY}-{P2_FY}_{P1_IY}-{P1_FY}.png",
        dpi=150,
        bbox_inches='tight'
    )

def plot_multimodelmean_stereo(mean_by_exp_var,sign_agreement_by_exp_var,var_lat_sim,var_lon_sim,lista_exp,lista_model,var_name,var_unit,levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY):
    """
    Plot the multi-model mean difference in climatological mean between two periods for all experiments in a stereographic projection.

    This function generates a figure with subplots, where each subplot corresponds to an experiment. For each experiment, it plots
    the multi-model mean difference in the climatological mean of a variable between two periods (P2 and P1) using a South Polar
    stereographic projection. The plot includes contour fill for the mean difference and stippling to indicate regions where a
    significant fraction of models agree on the sign of the change.

    Parameters:
    ----------
    mean_by_exp_var : dict
        A dictionary containing the multi-model mean difference in the climatological mean of the variable between two periods
        (P2 and P1) for each experiment. The keys are experiment names, and the values are 2D arrays representing the mean
        differences.

    sign_agreement_by_exp_var : dict
        A dictionary containing the fraction of models that agree on the sign of the change for each experiment. The keys are
        experiment names, and the values are 2D arrays with values between 0 and 1, where values >= 0.5 indicate agreement.

    var_lat_sim : dict
        A dictionary containing the latitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of latitude values.

    var_lon_sim : dict
        A dictionary containing the longitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of longitude values.

    lista_exp : list of str
        A list of experiment names to be plotted.

    lista_model : list of str
        A list of model names used to calculate the multi-model mean.

    var_name : str
        The name of the variable being plotted (e.g., "Temperature"). This is used for titles and labels.

    var_unit : str
        The unit of the variable (e.g., "K" for Kelvin). This is used for titles and labels.

    levels : list of float
        A list of levels for the contour fill plot. Defines the boundaries for the color map.

    colors : list of str
        A list of colors corresponding to the levels. Used to create the color map.

    LATMIN : float
        The minimum latitude value for the plot.

    LATMAX : float
        The maximum latitude value for the plot.

    LONMIN : float
        The minimum longitude value for the plot.

    LONMAX : float
        The maximum longitude value for the plot.

    P1_IY : str
        The initial year of the first period (P1).

    P1_FY : str
        The final year of the first period (P1).

    P2_IY : str
        The initial year of the second period (P2).

    P2_FY : str
        The final year of the second period (P2).

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the directory specified by `config["plot_dir"]`.
    
    """

    # Define figure size based on the number of experiments
    fig = plt.figure(figsize=(4.5*len(lista_exp),4))

    #Define grid for subplots
    gs = gridspec.GridSpec(1,len(lista_exp),figure=fig, width_ratios=4.5*np.ones(len(lista_exp)),height_ratios=[4])   

    #Define grid for subplots
    gs = gridspec.GridSpec(1,len(lista_exp),figure=fig, width_ratios=4.5*np.ones(len(lista_exp)),height_ratios=[2])    

    # Create colormap and normalization
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors,extend='both')

    # Make one subplot for each experiment
    for i in range(len(lista_exp)):    
        # Subplot position and define projection
        ax = fig.add_subplot(gs[i],projection=ccrs.SouthPolarStereo())

        # Grid for latitudes/longitudes for the first model (assumed same for all models)
        lons, lats = np.meshgrid(var_lon_sim[(lista_exp[0],lista_model[0])], var_lat_sim[(lista_exp[0],lista_model[0])])

        #Define projected lat/lon
        crs_latlon = ccrs.PlateCarree()

        #Define features for stereographic plot
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        #Define lat/lon plot limits
        ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
        
        # Plot filled contours of mean differences
        im=ax.contourf(lons, lats, mean_by_exp_var[lista_exp[i]], levels, cmap=cmap, transform=crs_latlon, extend='both')

        # Add stippling where sign agreement is above threshold
        ax.contourf(lons, lats, sign_agreement_by_exp_var[lista_exp[i]], levels=[0.5, 1.5], hatches=['...'], colors='none', transform=crs_latlon)

        # Add map features (coastlines and borders)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)

        # Add grid lines
        ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')

        #Add subplot title                       
        ax.set_title(lista_exp[i], fontsize="9") 

    # Adjust subplots arrangement
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.91,top=0.93, wspace=0.1 ,hspace=0.35)   

    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax,orientation='vertical')

    # Add main title for the figure
    plt.figtext(0.4, 0.97, f'MMM {var_name} ({var_unit}) change {P2_IY}-{P2_FY} - {P1_IY}-{P1_FY}', fontsize=16)

    # Save figure
    fig.savefig(
        f"{config['plot_dir']}/target_change/ensemble_MME_mean_stereo_{var_name}_change_{P2_IY}-{P2_FY}_{P1_IY}-{P1_FY}.png",
        dpi=150,
        bbox_inches='tight'
    )

def plot_multimodelmean_stereo_seas(mean_by_exp_var_seas,sign_agreement_by_exp_var_seas,var_lat_sim,var_lon_sim,lista_exp,lista_model,var_name,var_unit,levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY):
    """
     Plot the multi-model mean difference in climatological mean between two periods for all experiments in a stereographic projection, for the four seasons.

    This function generates a figure with subplots, where each row corresponds to a season (DJF, MAM, JJA, SON) and each column
    corresponds to an experiment. For each experiment and season, it plots the multi-model mean difference in the climatological
    mean of a variable between two periods (P2 and P1) using a South Polar stereographic projection. The plot includes contour fill
    for the mean difference and stippling to indicate regions where a significant fraction of models agree on the sign of the change.
 
    Parameters:
    ----------
    mean_by_exp_var_seas : list of dict
        A list of dictionaries containing the multi-model mean difference in the climatological mean of the variable between two
        periods (P2 and P1) for each experiment and season. Each dictionary corresponds to a season (DJF, MAM, JJA, SON), and the
        keys are experiment names. The values are 2D arrays representing the mean differences.

    sign_agreement_by_exp_var_seas : list of dict
        A list of dictionaries containing the fraction of models that agree on the sign of the change for each experiment and
        season. Each dictionary corresponds to a season (DJF, MAM, JJA, SON), and the keys are experiment names. The values are
        2D arrays with values between 0 and 1, where values >= 0.5 indicate agreement.

    var_lat_sim : dict
        A dictionary containing the latitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of latitude values.

    var_lon_sim : dict
        A dictionary containing the longitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of longitude values.

    lista_exp : list of str
        A list of experiment names to be plotted.

    lista_model : list of str
        A list of model names used to calculate the multi-model mean.

    var_name : str
        The name of the variable being plotted (e.g., "Temperature"). This is used for titles and labels.

    var_unit : str
        The unit of the variable (e.g., "K" for Kelvin). This is used for titles and labels.

    levels : list of float
        A list of levels for the contour fill plot. Defines the boundaries for the color map.

    colors : list of str
        A list of colors corresponding to the levels. Used to create the color map.

    LATMIN : float
        The minimum latitude value for the plot.

    LATMAX : float
        The maximum latitude value for the plot.

    LONMIN : float
        The minimum longitude value for the plot.

    LONMAX : float
        The maximum longitude value for the plot.

    P1_IY : str
        The initial year of the first period (P1).

    P1_FY : str
        The final year of the first period (P1).

    P2_IY : str
        The initial year of the second period (P2).

    P2_FY : str
        The final year of the second period (P2).

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the directory specified by `config["plot_dir"]`.
    
    """

    # Define seasons
    SEAS=['DJF','MAM','JJA','SON']

    # Define figure size based on the number of experiments and seasons
    fig = plt.figure(figsize=(4.5*len(lista_exp),4*len(SEAS)))

    # Define grid for subplots
    gs = gridspec.GridSpec(len(SEAS),len(lista_exp),figure=fig, width_ratios=4.5*np.ones(len(lista_exp)),height_ratios=4*np.ones(len(SEAS)))   

    # Create colormap and normalization
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors,extend='both')

    # Make one subplot for each experiment and season
    for i in range(len(lista_exp)):  
        for j in range(len(SEAS)):        
            # Subplot position and define projection
            ax = fig.add_subplot(gs[j, i],projection=ccrs.SouthPolarStereo())

            # Grid for latitudes/longitudes for the first model (assumed same for all models)
            lons, lats = np.meshgrid(var_lon_sim[(lista_exp[0],lista_model[0])], var_lat_sim[(lista_exp[0],lista_model[0])])

        	#Define projected lat/lon
            crs_latlon = ccrs.PlateCarree()

            # Define features for stereographic plot
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            ax.set_boundary(circle, transform=ax.transAxes)
            
            #Define lat/lon plot limits
            ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)

            # Plot filled contours of mean differences
            im=ax.contourf(lons, lats, mean_by_exp_var_seas[j][lista_exp[i]], levels, cmap=cmap, transform=crs_latlon, extend='both')
            
            # Add stippling where sign agreement is above threshold
            ax.contourf(lons, lats, sign_agreement_by_exp_var_seas[j][lista_exp[i]], levels=[0.5, 1.5], hatches=['...'], colors='none', transform=crs_latlon)

            # Add map features (coastlines and borders)
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)

            # Add grid lines
            ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')

            # Add subplot title
            ax.set_title(f'{SEAS[j]} {lista_exp[i]}', fontsize="12")

    # Adjust subplots arrangement
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.91,top=0.93, wspace=0.1 ,hspace=0.1)   

    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax,orientation='vertical')

    #Add main title for the figure
    plt.figtext(0.3, 0.97, f'MMM {var_name} ({var_unit}) change {P2_IY}-{P2_FY} - {P1_IY}-{P1_FY}', fontsize=18)

    # Save figure
    fig.savefig(
        f"{config['plot_dir']}/target_change/ensemble_MME_mean_stereo_seas_{var_name}_change_{P2_IY}-{P2_FY}_{P1_IY}-{P1_FY}.png",
        dpi=150,
        bbox_inches='tight'
    )

def plot_multimodelmean_zonalmean(mean_by_exp_var,var_lat_sim,lista_exp,lista_model,var_name,var_unit,colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX):
    """
    Plot the zonal mean of the multi-model mean difference in climatological mean between two periods for all experiments.

    This function generates a single plot showing the zonal mean difference in the climatological mean of a variable between two
    periods (P2 and P1) for all experiments. Each experiment is represented by a line, and the plot includes a horizontal reference
    line at y=0.

    Parameters:
    ----------
    mean_by_exp_var : dict
        A dictionary containing the multi-model mean difference in the climatological mean of the variable between two periods
        (P2 and P1) for each experiment. The keys are experiment names, and the values are 2D arrays representing the mean
        differences.

    var_lat_sim : dict
        A dictionary containing the latitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of latitude values.

    lista_exp : list of str
        A list of experiment names to be plotted.

    lista_model : list of str
        A list of model names used to calculate the multi-model mean.

    var_name : str
        The name of the variable being plotted (e.g., "Temperature"). This is used for titles and labels.

    var_unit : str
        The unit of the variable (e.g., "K" for Kelvin). This is used for titles and labels.

    colors : list of str
        A list of colors corresponding to each experiment. The colors are used to differentiate experiments in the plot.

    LATMIN : float
        The minimum latitude value for the x-axis of the plot.

    LATMAX : float
        The maximum latitude value for the x-axis of the plot.

    P1_IY : str
        The initial year of the first period (P1).

    P1_FY : str
        The final year of the first period (P1).

    P2_IY : str
        The initial year of the second period (P2).

    P2_FY : str
        The final year of the second period (P2).

    YMIN : float
        The minimum value for the y-axis of the plot.

    YMAX : float
        The maximum value for the y-axis of the plot.

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the directory specified by `config["plot_dir"]`.

    """

    #Define figure
    fig = plt.figure(figsize=(4.5,5))

    #Define grid for subplots
    gs = gridspec.GridSpec(1,1) 

    # Create subplot
    ax = fig.add_subplot(gs[0])  

    # Plot zonal mean difference for each experiment
    for i in range(len(lista_exp)):
        label = f'{lista_exp[i]}'
        ax.plot(
            var_lat_sim[(lista_exp[0], lista_model[0])], # Use latitudes from the first model/experiment
            mean_by_exp_var[lista_exp[i]].mean(axis=1), # Compute zonal mean
            label=label,
            color=colors[i]
        )

    # Add a horizontal reference line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Add legend
    ax.legend()

    # Set axis limits
    ax.set_xlim([LATMIN, LATMAX])
    ax.set_ylim([YMIN, YMAX])

    # Add labels
    ax.set_xlabel('Latitude', fontsize="8")
    ax.set_ylabel(var_name+' ('+var_unit+') change')

    # Define subplots arrangement
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.97,top=0.9, wspace=0.2 ,hspace=0.2)   

    #Add main title for the figure
    plt.figtext(0.2, 0.95, f'MMM {var_name} ({var_unit}) change {P2_IY}-{P2_FY} - {P1_IY}-{P1_FY}', fontsize=16)

    #Save figure
    fig.savefig(
        f"{config['plot_dir']}/target_change/ensemble_MME_mean_zonalmean_{var_name}_change_{P2_IY}-{P2_FY}_{P1_IY}-{P1_FY}.png",
        dpi=150,
        bbox_inches='tight'
    )
    
def plot_multimodelmean_zonalmean_seas(mean_by_exp_var_seas,var_lat_sim,lista_exp,lista_model,var_name,var_unit,colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX):
    """
    Plot the zonal mean of the multi-model mean difference in climatological mean between two periods for all experiments, for the four seasons.

    This function generates a figure with subplots, where each subplot corresponds to a season (DJF, MAM, JJA, SON). For each season,
    it plots the zonal mean difference in the climatological mean of a variable between two periods (P2 and P1) for all experiments.
    Each experiment is represented by a line, and the plot includes a horizontal reference line at y=0.

    Parameters:
    ----------
    mean_by_exp_var_seas : list of dict
        A list of dictionaries containing the multi-model mean difference in the climatological mean of the variable between two
        periods (P2 and P1) for each experiment and season. Each dictionary corresponds to a season (DJF, MAM, JJA, SON), and the
        keys are experiment names. The values are 2D arrays representing the mean differences.

    var_lat_sim : dict
        A dictionary containing the latitude values for each dataset. The keys are tuples of (experiment, model), and the values
        are arrays of latitude values.

    lista_exp : list of str
        A list of experiment names to be plotted.

    lista_model : list of str
        A list of model names used to calculate the multi-model mean.

    var_name : str
        The name of the variable being plotted (e.g., "Temperature"). This is used for titles and labels.

    var_unit : str
        The unit of the variable (e.g., "K" for Kelvin). This is used for titles and labels.

    colors : list of str
        A list of colors corresponding to each experiment. The colors are used to differentiate experiments in the plot.

    LATMIN : float
        The minimum latitude value for the x-axis of the plot.

    LATMAX : float
        The maximum latitude value for the x-axis of the plot.

    P1_IY : str
        The initial year of the first period (P1).

    P1_FY : str
        The final year of the first period (P1).

    P2_IY : str
        The initial year of the second period (P2).

    P2_FY : str
        The final year of the second period (P2).

    YMIN : float
        The minimum value for the y-axis of the plot.

    YMAX : float
        The maximum value for the y-axis of the plot.

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the directory specified by `config["plot_dir"]`.
    """

    # Define seasons
    SEAS=['DJF','MAM','JJA','SON']

    # Define figure size based on the number of seasons
    fig = plt.figure(figsize=(4.5*2,5*2))

    #Define grid for subplots
    gs = gridspec.GridSpec(2,2,figure=fig, width_ratios=[4.5,4.5],height_ratios=[4,4])   

    # Make one subplot for each season
    for j in range(len(SEAS)):
        ax = fig.add_subplot(gs[j])
        for i in range(len(lista_exp)):
            # Plot zonal mean difference for each experiment
            label = f'{lista_exp[i]}'
            ax.plot(
            	var_lat_sim[(lista_exp[0], lista_model[0])],  # Use latitudes from the first model/experiment
            	mean_by_exp_var_seas[j][lista_exp[i]].mean(axis=1), # Compute zonal mean
            	label=label,
            	color=colors[i]
        	)

        # Add a horizontal reference line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

        # Add legend
        ax.legend()

        # Set axis limits
        ax.set_xlim([LATMIN, LATMAX])
        ax.set_ylim([YMIN, YMAX])

        # Add labels
        ax.set_xlabel('Latitude', fontsize="8")
        ax.set_ylabel(var_name+' ('+var_unit+') change')

        # Add subplot title
        ax.set_title(SEAS[j], fontsize="10") 

    # Adjust subplots arrangement
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.97,top=0.9, wspace=0.2 ,hspace=0.2)   

    # Add main title for the figure
    plt.figtext(0.2, 0.95, f'MMM {var_name} ({var_unit}) change {P2_IY}-{P2_FY} - {P1_IY}-{P1_FY}', fontsize=16)

    # Save figure
    fig.savefig(
        f"{config['plot_dir']}/target_change/ensemble_MME_mean_zonalmean_seas_{var_name}_change_{P2_IY}-{P2_FY}_{P1_IY}-{P1_FY}.png",
        dpi=150,
        bbox_inches='tight'
    )

def main(config):
    """
    Run the diagnostic to process and analyze model data.

    This function sets up the environment, organizes input data, and prepares directories for storing diagnostic results
    and plots. It reads configuration settings, groups metadata by dataset, experiment, and alias, and creates necessary
    directories for output.

    Parameters:
    ----------
    config : dict
        A dictionary containing configuration settings for the diagnostic. Expected keys include:
        - "run_dir": Path to the directory containing the settings file.
        - "input_data": Dictionary of input data with metadata.
        - "work_dir": Working directory for intermediate files.
        - "plot_dir": Directory for saving plots.

    """

    # Load configuration settings from the settings file
    cfg = get_cfg(os.path.join(config["run_dir"], "settings.yml"))

    # Group metadata by dataset (model), experiment, and alias
    meta_dataset = group_metadata(config["input_data"].values(), "dataset")  # List of datasets (models)
    meta_exp = group_metadata(config["input_data"].values(), "exp")          # List of experiments
    meta = group_metadata(config["input_data"].values(), "alias")            # List of aliases (e.g., MODEL_r#i#p#f#)

    # Change to the working directory
    os.chdir(config["work_dir"])
    current_dir = os.getcwd()  # Store the current directory for reference

    # Create a directory for storing diagnostic results
    os.makedirs("target_change", exist_ok=True)  # Create "target_change" directory if it doesn't exist

    # Change to the "target_change" directory
    os.chdir(os.path.join(config["work_dir"], "target_change"))

    # Change to the plot directory and create a subdirectory for diagnostic plots
    os.chdir(config["plot_dir"])
    os.makedirs("target_change", exist_ok=True)  # Create "target_change" directory in the plot directory

    #Experiments list
    lista_exp = list(meta_exp.keys())
    #Models list
    lista_model= list(meta_dataset.keys())

    #Define periods for analysis
    P1_IY='1850' #Initial year Period1
    P1_FY='1884' #Final year Period1
    P2_IY='1980' #Initial year Period2
    P2_FY='2014' #Final year Period2

    #Compute multi-member mean differences between periods for each experiment/model
    pr_change_sim,pr_pval_sim,pr_lat_sim,pr_lon_sim=var_multimember_change(meta_dataset,meta_exp,meta,'pr_interp',86400,[P1_IY, P1_FY],[P2_IY, P2_FY],[0, 360, -90, 0],True)
    #Compute seasonal multi-member mean differences between periods for each experiment/model
    pr_change_sim_djf,pr_pval_sim_djf,pr_change_sim_mam,pr_pval_sim_mam,pr_change_sim_jja,pr_pval_sim_jja,pr_change_sim_son,pr_pval_sim_son,pr_lat_sim_seas,pr_lon_sim_seas=var_multimember_change_seas(meta_dataset,meta_exp,meta,'pr_interp',86400,[P1_IY, P1_FY],[P2_IY, P2_FY],[0, 360, -90, 0],True)

    #Plot figure for all experiments and models
    LATMIN = -90
    LATMAX = 0
    LONMIN = 0
    LONMAX = 360
    levels = [-15,-12,-9,-6,-3,3,6,9,12,15]
    colors = ['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#f5f5f5','#c7eae5','#80cdc1','#35978f','#01665e','#003c30']

    plot_allmodels_plate(pr_change_sim,pr_pval_sim,pr_lat_sim_seas,pr_lon_sim_seas,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY,seas=None)
    # Seasonals
    plot_allmodels_plate(pr_change_sim_djf,pr_pval_sim_djf,pr_lat_sim_seas,pr_lon_sim_seas,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY,seas='DJF')
    plot_allmodels_plate(pr_change_sim_mam,pr_pval_sim_mam,pr_lat_sim_seas,pr_lon_sim_seas,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY,seas='MAM')
    plot_allmodels_plate(pr_change_sim_jja,pr_pval_sim_jja,pr_lat_sim_seas,pr_lon_sim_seas,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY,seas='JJA')
    plot_allmodels_plate(pr_change_sim_son,pr_pval_sim_son,pr_lat_sim_seas,pr_lon_sim_seas,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY,seas='SON')

    #Plot figure for all experiments and models (zonal mean)   
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']
    YMIN = -15
    YMAX = 15

    plot_allmodels_zonalmean(pr_change_sim,pr_lat_sim,lista_exp,lista_model,'pr','%',colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX,seas=None)
    # Seasonals
    plot_allmodels_zonalmean(pr_change_sim_djf,pr_lat_sim_seas,lista_exp,lista_model,'pr','%',colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX,seas='DJF')
    plot_allmodels_zonalmean(pr_change_sim_mam,pr_lat_sim_seas,lista_exp,lista_model,'pr','%',colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX,seas='MAM')
    plot_allmodels_zonalmean(pr_change_sim_jja,pr_lat_sim_seas,lista_exp,lista_model,'pr','%',colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX,seas='JJA')
    plot_allmodels_zonalmean(pr_change_sim_son,pr_lat_sim_seas,lista_exp,lista_model,'pr','%',colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX,seas='SON')

    #Calculate multimodel mean
    mean_by_exp_pr,sign_agreement_by_exp_pr=MultiModelMean(pr_change_sim,level_agreement=0.8)
    mean_by_exp_pr_djf,sign_agreement_by_exp_pr_djf=MultiModelMean(pr_change_sim_djf,level_agreement=0.8)
    mean_by_exp_pr_mam,sign_agreement_by_exp_pr_mam=MultiModelMean(pr_change_sim_mam,level_agreement=0.8)
    mean_by_exp_pr_jja,sign_agreement_by_exp_pr_jja=MultiModelMean(pr_change_sim_jja,level_agreement=0.8)
    mean_by_exp_pr_son,sign_agreement_by_exp_pr_son=MultiModelMean(pr_change_sim_son,level_agreement=0.8)

    # Concatenate seasonal MMM 
    mean_by_exp_pr_seas = [mean_by_exp_pr_djf, mean_by_exp_pr_mam, mean_by_exp_pr_jja, mean_by_exp_pr_son]
    sign_agreement_by_exp_pr_seas = [sign_agreement_by_exp_pr_djf, sign_agreement_by_exp_pr_mam, mean_by_exp_pr_jja, sign_agreement_by_exp_pr_son]

    #Plot MMM for all experiments
    LATMIN = -90
    LATMAX = 0
    LONMIN = 0
    LONMAX = 360
    levels = [-15,-12,-9,-6,-3,3,6,9,12,15]
    colors = ['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#f5f5f5','#c7eae5','#80cdc1','#35978f','#01665e','#003c30']

    plot_multimodelmean_plate(mean_by_exp_pr,sign_agreement_by_exp_pr,pr_lat_sim,pr_lon_sim,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY)
    plot_multimodelmean_plate_seas(mean_by_exp_pr_seas,sign_agreement_by_exp_pr_seas,pr_lat_sim,pr_lon_sim,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY)

    #Stereographic plot
    plot_multimodelmean_stereo(mean_by_exp_pr,sign_agreement_by_exp_pr,pr_lat_sim,pr_lon_sim,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY)
    plot_multimodelmean_stereo_seas(mean_by_exp_pr_seas,sign_agreement_by_exp_pr_seas,pr_lat_sim,pr_lon_sim,lista_exp,lista_model,'pr','%',levels,colors,LATMIN,LATMAX,LONMIN,LONMAX,P1_IY,P1_FY,P2_IY,P2_FY)

    #Plot figure for MMM (zonal mean)
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']
    YMIN = -15
    YMAX = 15
    plot_multimodelmean_zonalmean(mean_by_exp_pr,pr_lat_sim,lista_exp,lista_model,'pr','%',colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX)
    plot_multimodelmean_zonalmean_seas(mean_by_exp_pr_seas,pr_lat_sim,lista_exp,lista_model,'pr','%',colors,LATMIN,LATMAX,P1_IY,P1_FY,P2_IY,P2_FY,YMIN,YMAX)

if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
                                    
