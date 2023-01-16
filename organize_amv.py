#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organize_amv

An attempt to catalogue and organize functions I've written in the amv module
'

Created on Sun Jan 15 18:40:30 2023

@author: gliu
"""

# ----------------------------------

"""
proc

Functions:
    
    maxabs              : Return max absolute value for a variable
    minabs              : Return min absolute value for a variable
    ann_avg             : Take annual average of monthly time series
    area_avg            : Take (weighted) area average within bounding box
    year2mon            : Separate mon x year dimensions
    detrend_dim         : Linear detrend along a dimension
    detrend_poly        : Perform polynomial detrend
    xrdeseason          : Deseason DataArray by removing mean seasonal cycle
    calc_clim           : Compute climatological monthly means
    remove_ss_sinusoid  : Moves annual and semiannual seasonal cycles using least squares fit to sinusoids
    calc_savg           : Caclulate seasonal average of an ND input
    
    
    make_classes_nd     : Make classes based on given thresholds.
    checkpoint          : Groups values based on thresholds and returns indices
    
    
    
    regress_2d          : Perform linear regression of Matrix B on A
    regress2ts          : Regression variable to a timeseries (uses regress_2d)
    eof_simple          : Perform EOF Analysis (cr. Yu-Chiao)
    calc_lagcovar       : Monthly lag-lead correlation
    calc_lagcovar_nd    : ND version of calc_lagcovar
    pearsonr_2d         : Compute Pearson's Correlation Coefficient for 2D matrix.
    calc_pearsonconf    : Compute upper and lower bounds using fisher-Z transform (?)
    ttest_rho           : Perform T-Test
    covariance_2d       : Calculate covariance for 2 2D arrays
    calc_conflag        : Calculate confidence Intervals for autocorrelation function
    make_ar1            : Create AR1 timeseries given lag 1 correlation coefficient, signa, and length (from slutil)
    patterncorr         : Compute pattern correlation between 2 2D variables
    calc_T2             : Compute the recorrelation timescale
    
    
    
    lp_butter           : Design and apply a low-pass butterworth filter
    calc_specvar        : Calculate variance of spectra below a certain threshold
    
    
    
    getpt_pop           : Average values on POP grid for a DataArray
    find_latlon         : Find lat/lon indices
    find_tlatlon        : Find lat/lon indices on POP Tripolar grid
    find_nan            : Locate and remove NaN points
    remap_nan           : Replace output of find_nan back into an Array with the original size
    sel_region          : Select bounding box and optinally perform average or sum (uses area_avg)
    coarsen_byavg       : Coarsen input by averaging over bins
    quick_interp2d      : Quick 2D interpolation of datapoints, works with [sel_region_cv]
    tilebylag           : Tile a monthly variable along a lag sequence
    get_posneg          : get positive/negative years of a variable from an index/timeseries
    get_posneg_sigma    : get positive/neutral/negative years for a variable using a stdev threshold
    sel_region_cv       : Select region for curvilienear/2D points in POP
    get_topN            : Get indices for top/bottom N values in an array (along last dimension)
    
    
    
    calc_AMV            : Compute AMV Index
    calc_AMVquick       : Compute AMV Index and Pattern (uses ann_avg,calc_AMV,regress2ts)
    calc_DMI            : Calculate Dipole Mode Index over the Indian Ocean (Saji et al. 1999)
    calc_remidx_simple  : Compute the Re-emergence Index (Bjyu et al ____)
    
    
    
    naninv              : Invert boolean array with NaNs
    
    
    
    numpy_to_da         : Convert NumPy array into DataArray (and save)
    
    
    
    cftime2str          : Convert array of cftime objects to string
    
        ----------------------------
    <<< ||| Dimension Gymnastics ||| >>>
        ----------------------------
    combine_dims        : Combine first n dimensions of an ND-array and reshape
    dim2front           : Move selected dimension to fhe first position/axis
    restoredim          : Revert array back to oldshape (reverse dim2front)
    flipdims            : Reverse axis/dimension order
    
    
    
    lon360to180_xr      : Flip longitude in a DataArray
    lon360to180         : Flip longitude from degrees East to West
    lon180to360         : Flip longitude from degrees West to East
    lon360to180_ds      : Flip longitude in a DataArray (repeat?)
    
    
    make_locstring      : Make file and plot title names of lat/lon coordinates
    make_locstring_bbox : Make file and plot title names for bounding box
    makedir             : Check to see if directory exists, and make new one if not.
    get_monstr          : Get Array containing strings of first N letters of each month
    addstrtoext         : Append string to end of file, before the extension
    get_stringnum       : Search for starting positive of keyword in a string
    
    
    
    
    
    
    
    
   
   
   
   

"""