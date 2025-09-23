#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Xarray Functions (xrfunc)
Builds upon functions in proc, etc.
Recipes from various scripts

Description of Functions

Created on Thu May 16 11:09:14 2024

@author: gliu

"""

import sys
#sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("../")

from amv import proc

import numpy as np
import xarray as xr
import time
import yo_box as ybx

#%% Compute Lag Correlations

# Copied from reemergence/analysis/viz_SST_SSS_coupling.py
def leadlagcorr(ds1,ds2,lags):
    # Computes lead lag correlation between ds1 and ds2 over dimension ['time']
    # over specified lead/lags [lags]. Loops over all other dimensions (lat,lon,depth,etc)
    calc_leadlag    = lambda x,y: proc.leadlag_corr(x,y,lags,corr_only=True)
    crosscorrs = xr.apply_ufunc(
        calc_leadlag,
        ds1,
        ds2,
        input_core_dims=[['time'],['time']],
        output_core_dims=[['lags']],
        vectorize=True,
        )
    leadlags     = np.concatenate([np.flip(-1*lags)[:-1],lags],) 
    crosscorrs['lags'] = leadlags
    return crosscorrs

# Need to test this version below
def leadlagcorr_pt(ds1,dspt,lags):
    # Computes lead lag correlation between ds1 and dspt over dimension ['time'], where dspt
    # is a timeseries
    # over specified lead/lags [lags]. Loops over all other dimensions (lat,lon,depth,etc)
    calc_leadlag    = lambda x: proc.leadlag_corr(x,dspt,lags,corr_only=True)
    crosscorrs = xr.apply_ufunc(
        calc_leadlag,
        ds1,
        input_core_dims=[['time']],
        output_core_dims=[['lags']],
        vectorize=True,
        )
    leadlags     = np.concatenate([np.flip(-1*lags)[:-1],lags],) 
    crosscorrs['lags'] = leadlags
    return crosscorrs

#%% Compute Monthly ACF (from check_intraregional_metrics)

# Compute Monthly ACFs
def compute_monthly_acf(tsin,nlags):
    
    # Pointwise script acting on array [time]
    ts = tsin.copy()
    
    # Stupid fix, set NaNs to zeros
    if np.any(np.isnan(ts)):
        if np.all(np.isnan(tsin)):
            return np.zeros((12,nlags)) * np.nan # Return all NaNs
        nnan = np.isnan(tsin).sum()
        print("Warning, NaN points found within timeseries. Setting NaN to zero" % nnan)
        ts[np.isnan(ts)] = 0.
    
    # Set up lags, separate to month x yrear
    lags    = np.arange(nlags)
    ntime   = len(ts)
    nyr     = int(ntime/12)
    tsmonyr = ts.reshape(nyr,12).T # Transpose to [month x year]
    
    # Preallocate
    sst_acfs = np.zeros((12,nlags)) * np.nan # base month x lag
    for im in range(12):
        ac= proc.calc_lagcovar(tsmonyr,tsmonyr,lags,im+1,0,yr_mask=None,debug=False)
        sst_acfs[im,:] = ac.copy()
    return sst_acfs


def pointwise_acf(ds,nlags):
    st = time.time()
    #ds1  = dsreg[0] # Need to assign dummy ds here
    acffunc = lambda x: compute_monthly_acf(x,37)
    acfs = xr.apply_ufunc(
        acffunc,
        ds,
        input_core_dims=[['time']],
        output_core_dims=[['basemon','lags']],
        vectorize=True,
        )
    print("Computed Pointwise ACF computation in %.2fs" % (time.time()-st))
    acfs['basemon'] = np.arange(1,13,1)
    return acfs
    

#%% Compute Spectra (as in visualize atmospheric persistenc      --e)

# Compute the power spectra (Testbed function from xrfunc)
def pointwise_spectra(tsens,nsmooth=1, opt=1, dt=None, clvl=[.95], pct=0.1):
    calc_spectra = lambda x: proc.point_spectra(x,nsmooth=nsmooth,opt=opt,
                                                dt=dt,clvl=clvl,pct=pct)
    
    # Change NaN to Zeros for now
    tsens_nonan = xr.where(np.isnan(tsens),0,tsens)
    
    # Compute Spectra
    specens = xr.apply_ufunc(
        proc.point_spectra,  # Pass the function
        tsens_nonan,  # The inputs in order that is expected
        # Which dimensions to operate over for each argument...
        input_core_dims=[['time'],],
        output_core_dims=[['freq'],],  # Output Dimension
        exclude_dims=set(("freq",)),
        vectorize=True,  # True to loop over non-core dims
    )
    
    # # Need to Reassign Freq as this dimension is not recorded
    ts1  = tsens.isel(ens=0).values
    freq = proc.get_freqdim(ts1)
    specens['freq'] = freq
    return specens

# # Get the frequency dimension by selecting a 1d version of the variable
# da_1d           = da_sm.isel(var=0,ens=0,exp=0).values # Example
# freq            = proc.get_freqdim(da_1d)
# sm_spec['freq'] = freq

#%% Polynomial Detrend (copied form hfcalc/scrap/compute_t2_rei_spg)

def pointwise_detrend(ds,order):
    # Apply Polynomial Detrend, looping over Lat and Lon
    # 
    def point_detrend(ts,order):
        ntime  = len(ts)
        times  = np.arange(ntime)
        if np.any(np.isnan(ts)):
            dt_out = times*np.nan
        else:
            dt_out = proc.polyfit_1d(times,ts,order)
        return dt_out[2]
    detrend_pt = lambda x: point_detrend(x,order)
    
    st = time.time()
    ds_detrended = xr.apply_ufunc(
        detrend_pt,
        ds,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        )
    print("Detrended in %.2fs" % (time.time()-st))
    return ds_detrended

#%% Get first Nan Example (taken from smio/investigate_blowup_sm)

# Use Xr_ufunc
dsid = xr.apply_ufunc(
    proc.getfirstnan,
    dsin,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize=True,
    )

