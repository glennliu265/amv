#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Xarray Functions (xrfunc)
Builds upon functions in proc, etc.

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

#%%

