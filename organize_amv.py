#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organize_amv

An attempt to catalogue and organize functions I've written in the amv module


Created on Sun Jan 15 18:40:30 2023

@author: gliu
"""

# ----------------------------------

"""
proc

Functions:
    
    -----------------------
    |||  Preprocessing  ||| ****************************************************
    -----------------------
        General functions for preprocessing data
    
        ~ Convenience Functions
    maxabs              : Return max absolute value for a variable
    minabs              : Return min absolute value for a variable
    nan_inv             : Invert boolean array with NaNs
    check_equal_nan     : Check if two N-D arrays are equal, ignoring NaNs
    nanargmaxds         : Take nanargmax for dataarray with masking
    nanargminds         : Take nanargmin for dataarray with masking
    make_mask           : Create a mask from a dataarray using specified nan values
    
        ~ Averaging ~
    ann_avg             : Take annual average of monthly time series
    area_avg            : Take (weighted) area average within bounding box
    area_avg_cosweight  : Take area-weighted average of xr.Dataset <xarray>
    area_avg_cosweight_cv : Curvilinear case of above
    
        ~ Seasonal Cycle ~
    year2mon            : Separate mon x year dimensions
    deseason            : Calculate monthly anomalies by removing the seasonal cycle
    xrdeseason          : Deseason DataArray by removing mean seasonal cycle <xarray>
    calc_savg           : Caclulate seasonal average of an ND input
    calc_savg_mon       : Same but using xarray groupby functions
    calc_clim           : Compute climatological monthly mean
    remove_ss_sinusoid  : Moves annual and semiannual seasonal cycles using least squares fit to sinusoids
    
        ~ Detrending ~
    detrend_dim         : Linear detrend along a dimension
    detrend_poly        : Perform polynomial detrend (2D)
    polyfit_1d          : 1-D polynomial fitting/detrending. Returns residuals and coeffs.
    xrdetrend           : Apply detrend_dim to xarray DataArray

        ~ Classification/Grouping ~
    make_classes_nd     : Make classes based on given thresholds.
    checkpoint          : Groups values based on thresholds and returns indices
    classify_bythres    : Assign classes based on threshold values from 2D map.
    make_thres_labels   : Make Threshold Labels
    
        ~ Spatial Analysis/Operations/Wrangling ~
    lon360to180         : Flip longitude from degrees East to West
    lon180to360         : Flip longitude from degrees West to East
    lon360to180_xr      : Flip longitude in a DataArray
    lon360to180_ds      : Flip longitude in a DataArray (repeat?)
    linear_crop         : Remove points above/below a specified line
    calc_dx_dy          : Retrieve 2d array of distances between grid points in meters
    
    ------------------------------
    |||  Statistical Analysis  ||| ****************************************************
    ------------------------------
    
        ~ Regression ~
    regress_2d          : Perform linear regression of Matrix B on A
    regress2ts          : Regression variable to a timeseries (uses regress_2d)
    
    
        ~ Lead/Lag Analysis ~
    calc_lagcovar       : Monthly lag-lead correlation
    calc_lagcovar_nd    : ND version of calc_lagcovar
    calc_lag_covar_ann  : Yearly (or any time unit) lag-lead correlation calculator
    calc_conflag        : Calculate confidence Intervals for autocorrelation function
    tilebylag           : Tile a monthly variable along a lag sequence
    leadlag_corr        : Compute lead/lag correlation of 2 variables (all months)
    
        
        ~ EOF Analysis ~
    eof_simple          : Perform EOF Analysis (cr. Yu-Chiao Liang)
    eof_filter          : Discard modes above a percentage threshold
    
        ~ Correlation ~
    pearsonr_2d         : Compute Pearson's Correlation Coefficient for 2D matrix.
    calc_pearsonconf    : Compute upper and lower bounds using fisher-Z transform (?)
    patterncorr         : Compute the pattern correlation between 2 2-D maps (modified from slutil)
    patterncorr_nd      : Vectorized version of paterncorr script
    nancorr             : Correlation on timeseries with NaN
    
        ~ Significance Testing ~
    calc_dof            : Compute effective degrees of freedom
    ttest_rho           : Perform T-Test
    calc_stderr         : Compute Standard Error (assuming normal distr)
    regress_ttest       : Perform regression and also compute significance via t-test
    
        ~ Other ~
    covariance_2d       : Calculate covariance for 2 2D arrays
    make_ar1            : Create AR1 timeseries given lag 1 correlation coefficient, signa, and length (from slutil)
    patterncorr         : Compute pattern correlation between 2 2D variables
    patterncorr_nd      : Compute pattern correlation between 2 ND variables
    calc_binwidth       : Use Freedman Diaconis rule to compute bin width
    expfit              : Fit Exponential to lagged autocorrelation function (from 12.S922 Final Project)
    calc_monvar         : Compute monthly variance of a timesesries
    
    -----------------------------------------
    |||  Spectral Analysis and Filtering  ||| ****************************************************
    -----------------------------------------
    
    lp_butter           : Design and apply a low-pass butterworth filter
    calc_specvar        : Calculate variance of spectra below a certain frequency threshold
    point_spectra       : Compute power spectra for a single timeseries
    get_freqdim         : Get frequency dimension for a yspectra calculation using yo_spec
    calc_confspec       : Calculate confidence interval of spectra
    plot_conflog        : Plot confidence intervals for spectral density on a log scale
    
    
    ----------------------------
    |||  Indexing & Querying ||| ****************************************************
    ----------------------------
        Functions for finding certain values and subsetting data
    
    find_latlon         : Find lat/lon indices
    find_tlatlon        : Find lat/lon indices on POP Tripolar grid
    find_nan            : Locate and remove NaN points
    selpt_ds            : Select a point in an xr.dataarray/dataset
    remap_nan           : Replace output of find_nan back into an Array with the original size
    sel_region          : Select bounding box and optinally perform average or sum (uses area_avg)
    sel_region_cv       : Select region for curvilienear/2D points in POP
    sel_region_xr       : Select region for xarray DataSet or DataArray <xarray>
    sel_region_xr_cv    : Same as sel_region_cv but for Dataset or DataArray
    get_bbox            : Get bounding box of a dataset from "lon" and "lat" dimensions <xarray>
    resize_ds           : Given list of datasets, resize all to the smallest bounding box <xarray>
    get_posneg          : get positive/negative years of a variable from an index/timeseries
    get_posneg_sigma    : get positive/neutral/negative years for a variable using a stdev threshold
    get_topN            : Get indices for top/bottom N values in an array (along last dimension)
    maxid_2d            : Find indices along each dimension for a 2D matrix. Ignores NaN values.
    sort_by_axis        : Sort arrays along axis by values from a single array 
    get_nearest         : Get indices of the closest value to an input (absolute diff)
    index_window        : Index a moving window of values (from scm.)
    
    -----------------------------------
    |||  Interpolation & Regridding ||| ****************************************************
    -----------------------------------
        Functions for interpolating and regridding data.
        
    coarsen_byavg       : Coarsen input by averaging over bins
    getpt_pop           : Average values on POP grid for a DataArray
    quick_interp2d      : Quick 2D interpolation of datapoints, works with [sel_region_cv]
    repair_timestep     : Given dataarray with all-NaN timeslice, replace with linear interpolation
    
    -------------------------
    |||  Climate Analysis ||| ****************************************************
    -------------------------
        Higher-level functions, such as calculation of indices or timescale, etc.
    
    calc_AMV            : Compute AMV Index
    calc_AMVquick       : Compute AMV Index and Pattern (uses ann_avg,calc_AMV,regress2ts)
    calc_DMI            : Calculate Dipole Mode Index over the Indian Ocean (Saji et al. 1999)
    calc_remidx_simple  : Compute the Re-emergence Index (Bjyu et al ____)
    calc_remidx_xr      : Same as above, but works with 1D input array of [lags] for xr.apply_ufunc support.
    calc_T2             : Compute the recorrelation timescale
    remove_enso         : Copied from scm, remove ENSO from variable using regression
    calc_HF             : Calculate heat flux feedback (copied from scm)
    check_flx           : Check the sign of heat flux, copied from (calc_Fprime_lens)
    
    
    ----------------------------
    ||| Dimension Gymnastics ||| **********************************************
    ----------------------------
        Functions focused on re-arranging/shuffling axes of an ND array
        
    combine_dims        : Combine first n dimensions of an ND-array and reshape
    dim2front           : Move selected dimension to fhe first position/axis
    restoredim          : Revert array back to oldshape (reverse dim2front)
    flipdims            : Reverse axis/dimension order
    
    ---------------------
    |||  Convenience ||| ****************************************************
    ---------------------
        General convenience functions, and other odds/ends 
        
    numpy_to_da         : Convert NumPy array into DataArray (and save)
    cftime2str          : Convert array of cftime objects to string
    noleap_tostr        : Convert Noleap to string
    ds_dropvars         : Drop all variables except those included in the list.
    make_encoding_dict  : Make encoding dictionary for each variable of an xarray dataset
    npz_to_dict         : Make loaded npz file a dict
    format_ds           : Standardize Lat/Lon direction and naming in netCDFs
    savefig_pub         : Save a publication-ready figure
    check_sum_ds        : Check summed output of DataArray
    get_xryear          : use xr.cftime to make dummy "year" for calculations
    rep_ds              : Repeat ds certain number of times along a specified dimension
    check_latlon_ds     : Check to make sure datasets of the same size have the same lat/lon coordinates
    splittime_ds        : Crop ds around a specific year
    get_dtmon           : Get number of seconds in each month
    stdsqsum_da         : Take the square root of the sum of squares for a DataArray
    printtime           : Print time elapsed in seconds
    
    -----------------
    |||  Labeling ||| ****************************************************
    -----------------
        Functions for labeling plots and files, and for managing directories.
        
    make_locstring      : Make file and plot title names of lat/lon coordinates
    make_locstring_bbox : Make file and plot title names for bounding box
    makedir             : Check to see if directory exists, and make new one if not.
    checkfile           : Check to see if a file exists. Return True if so.
    get_monstr          : Get Array containing strings of first N letters of each month
    addstrtoext         : Append string to end of file, before the extension
    get_stringnum       : Search for starting positive of keyword in a string
    fix_febstart        : For dataarrays, alter files that start with feb to jan.

"""
