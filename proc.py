#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:12:31 2020

Data Processing Functions. See organize_amv.py for contents/details.
test from stormtrack (moved 20200815)

@author: gliu
"""

import numpy as np
import xarray as xr
import calendar as cal
import numpy.ma as ma
from scipy import signal,stats
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend
import os
import time
import scipy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy as sp
import pandas as pd
import datetime
import tqdm

#%% Optional Yobox Import
import_yobox = False
if import_yobox:
    import sys
    yopath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/"
    sys.path.append(yopath)
    import yo_box as ybx

"""
-----------------------
|||  Preprocessing  ||| ****************************************************
-----------------------
"""

#%% ~ Convenience Functions

def maxabs(invar,axis=None,keep_sign=False,debug=False):
    # Return max absolute value for 1 variable. Keep sign if set.
    # Default is flatten and use first axis
    if axis is None:
        invar = invar.flatten()
        axis  = 0
    outvar = np.nanmax(np.abs(invar),axis=axis) # Get the amplitude
    if keep_sign:
        idmax = np.nanargmax(np.abs(invar),axis=axis)
        if debug:
            test = np.take_along_axis(np.abs(invar), np.expand_dims(idmax, axis=0), axis=0).squeeze()
            print("Difference is: %f" % (np.nanmax(np.abs((outvar-test).flatten()))))
        signs = np.take_along_axis(np.sign(invar), np.expand_dims(idmax, axis=0), axis=0).squeeze()
        outvar *= signs
    return outvar

def minabs(invar,axis=None,keep_sign=False,debug=False):
    # Return min absolute value for 1 variable. Keep sign if set.
    # Default is flatten and use first axis
    if axis is None:
        invar = invar.flatten()
        axis  = 0
    outvar = np.nanmin(np.abs(invar),axis=axis) # Get the amplitude
    if keep_sign:
        idmax = np.nanargmin(np.abs(invar),axis=axis)
        if debug:
            test = np.take_along_axis(np.abs(invar), np.expand_dims(idmax, axis=0), axis=0).squeeze()
            print("Difference is: %f" % (np.nanmax(np.abs((outvar-test).flatten()))))
        signs = np.take_along_axis(np.sign(invar), np.expand_dims(idmax, axis=0), axis=0).squeeze()
        outvar *= signs
    return outvar

def nan_inv(invar):
    """
    Invert boolean array with NaNs

    Parameters
    ----------
    invar (ARRAY) : BOOLEAN Array with NaNs

    Returns
    -------
    inverted (ARRAY) : Inverted Boolean Array
    """
    vshape = invar.shape
    inverted  = np.zeros((vshape)) * np.nan
    inverted[invar == True]  = False
    inverted[invar == False] = True
    return inverted

def check_equal_nan(a,b):
    # 2 ND arrays a and b. Check if they are equal, ignoring nans, and return True/False.
    chk   = ((a == b) | (np.isnan(a) & np.isnan(b))).all()
    return chk

#%% ~ Averaging

def ann_avg(ts,dim,monid=None,nmon=12):
    """
    # Take Annual Average of a monthly time series
    where time is axis "dim"
    
    1) ts : ARRAY
        Target timeseries (can be N-D array)
    2) dim : INT
        Time axis
    3) monid : LIST of INT
        Months/Timesteps to average over (default is all 12)
    4) nmon : INT
        Number of timesteps per year (assumes 12 for monthly data)
    
    """
    if monid is None:
        monid = np.arange(0,nmon,1) # Average over all timesteps in a year
    
    # Find which axis is time
    tsshape = ts.shape
    ntime   = ts.shape[dim] 
    
    # Separate month and year
    newshape =    tsshape[:dim:] +(int(ntime/nmon),nmon) + tsshape[dim+1::]
    annavg = np.reshape(ts,newshape)
    
    # Take the specified months along the month axis
    annavg = np.take(annavg,monid,axis=dim+1)
    
    # Take the mean along the month dimension
    annavg = np.nanmean(annavg,axis=dim+1)
    return annavg

def area_avg(data,bbox,lon,lat,wgt=None):
    """
    Function to find the area average of [data] within bounding box [bbox], 
    based on wgt type (see inputs)
    Inputs:
        1) data: target array [lon x lat x otherdims]
        2) bbox: bounding box [lonW, lonE, latS, latN]
        3) lon:  longitude coordinate
        4) lat:  latitude coodinate
        5) wgt:  number (or str) to indicate weight type
                    0 or None     no weighting
                    1 or 'cos'  = cos(lat)
                    2 or 'cossq' = sqrt(cos(lat))
                
    
    Output:
        1) data_aa: Area-weighted array of size [otherdims]
        
    Dependencies:
        numpy as np
    

    """
    
    # Check order of longitude
    # vshape = data.shape
    #nlon = lon.shape[0]
    #nlat = lat.shape[0]
    
    # Find lat/lon indices 
    kw = np.abs(lon - bbox[0]).argmin()
    ke = np.abs(lon - bbox[1]).argmin()
    ks = np.abs(lat - bbox[2]).argmin()
    kn = np.abs(lat - bbox[3]).argmin()
    
    # Select the region
    sel_data = data[kw:ke+1,ks:kn+1,:]
    
    # If wgt == 1, apply area-weighting 
    if (wgt != 0) or (wgt is not None):
        
        # Make Meshgrid
        _,yy = np.meshgrid(lon[kw:ke+1],lat[ks:kn+1])
        
        # Calculate Area Weights (cosine of latitude)
        if (wgt == 1) or (wgt == "cos"):
            wgta = np.cos(np.radians(yy)).T
        elif (wgt == 2) or (wgt == "cossq"):
            wgta = np.sqrt(np.cos(np.radians(yy))).T
        
        # Remove nanpts from weight, ignoring any pt with nan in otherdims
        nansearch = np.sum(sel_data,2) # Sum along otherdims
        wgta[np.isnan(nansearch)] = 0
        
        # Apply area weights
        #data = data * wgtm[None,:,None]
        sel_data  = sel_data * wgta[:,:,None]
    
    # Take average over lon and lat
    if (wgt != 0) or (wgt is not None):

        # Sum weights to get total area
        sel_lat  = np.sum(wgta,(0,1))
        
        # Sum weighted values
        data_aa = np.nansum(sel_data/sel_lat,axis=(0,1))
    else:
        # Take explicit average
        data_aa = np.nanmean(sel_data,(0,1))
    return data_aa
    
def area_avg_cosweight(ds,sqrt=False):
    # Take area average of dataset, applying cos weighting
    # Based on https://docs.xarray.dev/en/latest/examples/area_weighted_temperature.html
    weights     = np.cos(np.deg2rad(ds.lat))
    if sqrt: # Take squareroot if option is set
        weights = np.sqrt(weights)
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(('lat','lon'))

def area_avg_cosweight_cv(ds,vname,sqrt=False):
    # Take area average of dataset, applying cos weighting
    weights     = np.cos(np.deg2rad(ds.TLAT))
    if sqrt: # Take squareroot if option is set
        weights = np.sqrt(weights)
    ds_weighted = ds[vname].weighted(weights)
    return ds_weighted.mean(('nlat','nlon'))

#%% ~ Seasonal Cycle

def year2mon(ts,return_monxyr=True):
    """
    Separate mon x year from a 1D timeseries of monthly data
    """
    ts = np.reshape(ts,(int(np.ceil(ts.size/12)),12))
    if return_monxyr:
        ts = ts.T # [year x mon] --> [mon x year]
    return ts 

def deseason(ts,dim=0,return_scycle=False):
    """
    Remove mean seasonal cycle from array of timeseries [ts], 
    with time in dimension [dim].

    Parameters
    ----------
    ts  : ND ARRAY with time in dimension [dim]
    dim : Axis/Dimension of time

    Returns
    -------
    tsanom : Deseasoned Timeseries
    scycle : Seasonal Cycle
    
    """
    scycle,tsmonyr = calc_clim(ts,dim=dim,returnts=1,keepdims=True) # Compute seasonal cycle
    tsanom = tsmonyr - scycle # Remove to compute anomalies
    if return_scycle:
        return tsanom,scycle
    return tsanom

def xrdeseason(ds,check_mon=True):
    """ Remove seasonal cycle, given an Dataarray with dimension 'time'"""
    if check_mon:
        try: 
            if ds.time[0].values.item().month != 1:
                print("Warning, first month is not Jan...")
        except:
            print("Warning, not checking for feb start")
    
    return ds.groupby('time.month') - ds.groupby('time.month').mean('time')

def calc_savg(invar,debug=False,return_str=False,axis=-1,ds=False):
    """
    Calculate Seasonal Average of input with time in the last dimension
    (or specify axis with axis=N)
    
    Inputs:
        1) invar : ARRAY[...,time], N-D monthly variable where time is the last axis
        2) debug : BOOL, Set to True to Print the Dimension Sizes
        3) return_str : BOOL, Set to True to return the Month Strings (DJF,MAM,...)
        4) ds: BOOL, if true, invar can be DataArray with 'mon' variable indicating month
    Outputs:
        1) savgs : LIST of ARRAYS, [winter, spring, summer, fall]
        2) snames : LIST of STR, season names, (returns if return_str is true)
    """
    
    snames = ("DJF"   ,"MAM"  ,"JJA"  ,"SON")
    sids   = ([11,0,1],[2,3,4],[5,6,7],[8,9,10])
    
    if ds is False:
        savgs = []
        for s in range(4):
            
            savgs.append(np.nanmean(np.take(invar,sids[s],axis=axis),axis)) # Take mean along last dimension
            if debug:
                print(savgs[s].shape)
        if return_str:
            return savgs,snames
    else:
        savgs = []
        for s in range(4):
            ds_season = invar.isel(mon=(sids[s])).mean('mon')
            savgs.append(ds_season)
        
        savgs = xr.concat(savgs,dim='season')
        savgs = savgs.assign_coords({'season': np.array(list(snames),dtype=str)})
    return savgs

def calc_savg_mon(ds):
    """Calculate the seasonal averages from ds with dimension "mon"""
    dstime = ds.assign_coords(mon=get_xryear()).rename({'mon':'time'})
    dssavg = dstime.groupby('time.season').mean('time')
    return dssavg
    
def calc_clim(ts,dim,returnts=0,keepdims=False):
    """
    Given monthly timeseries with time in axis [dim], calculate the climatology...
    
    Returns: climavg,tsyrmon (if returnts=1)
    
    """
    tsshape = ts.shape
    ntime   = ts.shape[dim] 
    newshape =    tsshape[:dim:] +(int(ntime/12),12) + tsshape[dim+1::]
    
    tsyrmon = np.reshape(ts,newshape)
    climavg = np.nanmean(tsyrmon,axis=dim,keepdims=keepdims)
    
    if returnts==1:
        return climavg,tsyrmon
    else:
        return climavg
    
def remove_ss_sinusoid(ts,t=None,dt=12,semiannual=True,Winv=None):
    """
    Removes annual and semiannual (optional) cycles
    using least squares fit to sinusoids:
    
        
    y = A + B sin(w*t*pi)   + C cos(w*t*pi)
          + D sin(2w*t*pi)  + E cos(2w*t*pi) 
    
    where w = 1/period
    
    Inputs
    ------
        ts : [n x m] ARRAY
            Matrix of timeseries n is time (rows), m is space (columns)
        t  : [n,] ARRAY, optional    
            Array of times (Default, uses np.arange(0,n,1))
        dt : INT, optional    
            Number of timesteps equivalent to 1 year
        semiannual: BOOL, optional
            Set to True to remove semiannual cycle as well (default is True)
        Winv : [n x n matrix], optional
            Weighting matrix, Default uses equal weights for all time points (1/n) 
    
    Outputs
    -------
        x : ARRAY
            Contains the coefficients [A,B,C,D,E]
            [5 x 1] if semiannual is true, [3 x 1] if false
        E : ARRAY
            Contains the sines and cosines [n x 5] (or [n x 3])
    
    """
    # Get shape
    nt,ns = ts.shape
    
    # Make time vector
    if t is None:
        t = np.arange(0,nt,1)
    
    # Set periods
    omegas = [1/dt]
    if semiannual:
        omegas.append(omegas[0]*2)
    
    # Prepare observational matrix
    E = []
    E.append(np.ones(nt))
    for w in omegas:
        sints = np.sin(np.pi*t*w)
        costs = np.cos(np.pi*t*w)
        E.append(sints)
        E.append(costs)
    E = np.array(E).T # [nt x 4]
    
    # Calculate weighting matrix
    if Winv is None:
        Winv = np.diag(np.ones(nt)) * 1/nt # [nt x nt]
    
    # Perform Least Squares Fit
    F = np.linalg.inv(E.T@Winv@E)@E.T@Winv
    x = F@ts
    return x,E

#%% ~ Detrending

def detrend_dim(invar,dim,return_dict=False,debug=False):
    
    """
    Detrends n-dimensional variable [invar] at each point along axis [dim].
    Performs appropriate reshaping and NaN removal, and returns
    variable in the same shape+order. Assumes equal spacing along [dim] for 
    detrending
    
    Also outputs linear model and coefficients.
    
    Dependencies: 
        numpy as np
        find_nan (function)
        regress_2d (function)
    
    Inputs:
        1) invar: variable to detrend
        2) dim: dimension of axis to detrend along
        
    Outputs:
        1) dtvar: detrended variable
        2) linmod: computed trend at each point
        3) beta: regression coefficient (slope) at each point
        4) interept: y intercept at each point
    """
    
    # Reshape variable
    varshape = invar.shape
    
    # Reshape to move time to first dim
    newshape = np.hstack([dim,np.arange(0,dim,1),np.arange(dim+1,len(varshape),1)])
    newvar = np.transpose(invar,newshape)
    
    # Combine all other dims and reshape to [time x otherdims]
    tdim        = newvar.shape[0]
    if len(varshape) <= 1:
        print("Warning, function will not work for 1-D arrays...")
    otherdims   = newvar.shape[1::]
    proddims    = np.prod(otherdims)
    newvar      = np.reshape(newvar,(tdim,proddims))

    
    # Find non nan points
    varok,knan,okpts = find_nan(newvar,0)
    
    # Ordinary Least Squares Regression
    tper    = np.arange(0,tdim)
    m,b     = regress_2d(tper,varok)
    
    # Squeeze things? (temp fix, need to check)
    m       = m.squeeze()
    b       = b.squeeze()
    
    # Detrend [Space,1][1,Time]
    ymod = (m[:,None] * tper + b[:,None]).T
    dtvarok = varok - ymod
    
    # Replace into variable of original size
    dtvar  = np.zeros(newvar.shape) * np.nan
    linmod = np.copy(dtvar)
    beta   = np.zeros(okpts.shape) * np.nan
    intercept = np.copy(beta)
    
    dtvar[:,okpts]      = dtvarok
    linmod[:,okpts]     = ymod
    beta[okpts]         = m
    intercept[okpts]    = b
    
    # Reshape to original size
    dtvar  = np.reshape(dtvar,((tdim,)+otherdims))
    linmod = np.reshape(linmod,((tdim,)+otherdims))
    beta = np.reshape(beta,(otherdims))
    intercept = np.reshape(beta,(otherdims))
    
    # Tranpose to original order
    oldshape = [dtvar.shape.index(x) for x in varshape]
    dtvar = np.transpose(dtvar,oldshape)
    linmod = np.transpose(linmod,oldshape)
    
    # Debug Plot
    if debug:
        klon = 33
        klat = 33
        
        mvmean = lambda x,w:  np.convolve(x, np.ones(w), 'same') / w
        w     = 120
        raw_y = invar[:,klat,klon]
        dt_y  = dtvar[:,klat,klon]
        x     = np.arange(0,len(dt_y))
        
        fig,ax = plt.subplots(1,1)
        ax.scatter(x,mvmean(raw_y,w),s=1.5,label="raw")
        ax.scatter(x,mvmean(dt_y,w),s=1.5,label='detrend')
        ax.plot(x,linmod[:,klat,klon],label="fit")
        ax.legend()
        plt.show()
    
    if return_dict:
        outdict = dict(detrended_var=dtvar,linearmodel=linmod,beta=beta,intercept=intercept)
        return outdict
    return dtvar,linmod,beta,intercept

def detrend_poly(x,y,deg):
    """
    Matrix for of polynomial detrend
    # Based on :https://stackoverflow.com/questions/27746297/detrend-flux-time-series-with-non-linear-trend
    
    Inputs:
        1) x --> independent variable
        2) y --> 2D Array of dependent variables
        3) deg --> degree of polynomial to fit
    
    """
    # Transpose to align dimensions for polyfit
    if len(y) != len(x):
        y = y.T
    
    # Get the fit
    fit    = np.polyfit(x,y,deg=deg)
    # Prepare matrix (x^n, x^n-1 , ... , x^0)
    #inputs = np.array([np.power(x,d) for d in range(len(fit))])
    inputs = np.array([np.power(x,d) for d in reversed(range(len(fit)))])
    
    # Calculate model
    model = fit.T.dot(inputs)
    
    # Remove trend
    ydetrend = y - model.T
    return ydetrend,model

def polyfit_1d(x,y,order):
    """
    Similar to detrend poly but just for a 1-D array, but returning the residuals
    as well.

    Parameters
    ----------
    x : ARRAY
        Input timeseries,independent variable
    y : ARRAY
        Target. dependent variable
    order : INT
        Order of the polynomial to fit

    Returns
    -------
    coeffs : LIST
        Coefficients of fitted polynomial in descending order
    newmodel : ARR
        The fitted model.
    residual : ARR
        Residuals.

    """
    coeffs   = np.polyfit(x,y,order,)
    
    newmodel = [np.power(x,order-N)*np.array(coeffs)[N] for N in range(order+1)] 
    newmodel = np.array(newmodel).sum(0)
    residual = y - newmodel
    return coeffs,newmodel,residual


def xrdetrend(ds,timename='time',verbose=True):
    
    st          = time.time()
    if len(ds.shape) == 1: 
        ts_dt       = sp.signal.detrend(ds.data)
    else:
        # Simple Linear detrend along dimension 'time'
        tdim        = list(ds.dims).index(timename) # Locate Time Dim
        dt_dict     = detrend_dim(ds.values,tdim,return_dict=True) # ASSUME TIME in first axis
        ts_dt       = dt_dict['detrended_var']
    ds_anom_out = xr.DataArray(ts_dt,dims=ds.dims,coords=ds.coords,name=ds.name)
    if verbose:
        print("Detrended in %.2fs" % (time.time()-st))
    return ds_anom_out
    
def xrdetrend_1d(ds,order,return_model=False):
    ntime = len(ds.time)
    x     = np.arange(ntime)
    y     = ds.data
    ydetrended,model=detrend_poly(x,y,order)
    if ds.name is None:
        ds = ds.rename('detrended_input')
    dsout  = xr.DataArray(ydetrended,coords=dict(time=ds.time),dims=dict(time=ds.time),name=ds.name)
    if return_model:
        dsfit  = xr.DataArray(model,coords=dict(time=ds.time),dims=dict(time=ds.time),name='fit')
        dsout2 = xr.merge([dsout,dsfit])
        return dsout2
    return dsout


def xrdetrend_nd(invar,order,regress_monthly=False,return_fit=False):
    """
    Given an DataArray [invar] and [order] of polynomial fit,
    fit the timeseries and detrend.
    Option to do so separately by month [regress_monthly=True]
    and return the fit [return_fit=True].
    Based on detrend_by_regression
    
    Note: Assumes the DataArray as [time] dimension names and merges
          the rest as ["space"] dimensions...
    See: /smio/scrap/check_enso_monthly_regression.py for testing script
    """
    # Change to [lon x lat x time]
    reshape_flag = False
    try:
        invar       = invar.transpose('lon','lat','time')
        invar_arr   = invar.data # [lon x lat x time]
        
    except:
        print("Warning, input is not 3d or doesn't have ('lon','lat','time')")
        reshape_output = make_2d_ds(invar,keepdim='time') #[1 x otherdims x time]
        invar_arr      = reshape_output[0].data
        reshape_flag = True
    
    # Filter out NaN points
    nlon,nlat,ntime = invar_arr.shape
    invar_rs = invar_arr.reshape(nlon*nlat,ntime) # Space x Time
    nandict  = find_nan(invar_rs,1,return_dict=True)
    
    if regress_monthly: # Do regression separately for each month
    
        # Reshape to space x yr x mon
        cleaned_data = nandict['cleaned_data']
        nok,_        = cleaned_data.shape
        nyr          = int(ntime/12)
        cd_yrmon     = cleaned_data.reshape((nok,nyr,12)) # Check 
        
        # Preallocate and detrend separately for each month
        detrended_bymon = np.zeros(cd_yrmon.shape)*np.nan # Detrended Variable
        fit_bymon       = detrended_bymon.copy() # n-order polynomial fit
        for im in range(12):
            # Get data for month
            cdmon                   = cd_yrmon[:,:,im]
            xdim                    = np.arange(nyr)
            detrended_mon,fit_mon   = detrend_poly(xdim,cdmon,order)
            detrended_bymon[:,:,im] = detrended_mon.T
            fit_bymon[:,:,im]       = fit_mon
            
            # Debug Plot (by month)
            # ii = 22
            # fig,ax = plt.subplots(1,1)
            # ax.plot(xdim,detrended_mon[:,ii],label="Detrended",color='blue')
            # ax.plot(xdim,fit_mon[ii,:],label="Fit",color="red")
            # ax.plot(xdim,cdmon[ii,:],label="Raw",color='gray',ls='dashed')
            # ax.legend()
        
        # Reshape the variables
        detrended_bymon = detrended_bymon.reshape(nok,ntime) # [Space x Time]
        fit_bymon = fit_bymon.reshape(nok,ntime) # [Space x Time]
        
        # # Debug Plot (full timeseries)
        # ii = 77
        # fig,ax = plt.subplots(1,1)
        # xdim = np.arange(ntime)
        # ax.plot(xdim,detrended_bymon.T[:,ii],label="Detrended",color='blue')
        # ax.plot(xdim,data_fit[ii,:],label="Fit",color="red")
        # ax.plot(xdim,nandict['cleaned_data'][ii,:],label="Raw",color='gray',ls='dashed')
        # ax.legend()
        
        # Replace Detrended data in original array
        arrout = np.zeros((nlon*nlat,ntime)) * np.nan
        arrout[nandict['ok_indices'],:] = detrended_bymon
        arrout = arrout.reshape(nlon,nlat,ntime).transpose(2,1,0) # Flip to time x lat x lon
        
        if return_fit:
            fitout = np.zeros((nlon*nlat,ntime)) * np.nan
            fitout[nandict['ok_indices'],:] = fit_bymon
            fitout = fitout.reshape(nlon,nlat,ntime).transpose(2,1,0)
        
    else: # Do regression for all months together...
        
        # Apply a fit using proc.detrend_poly
        xdim = np.arange(invar.shape[-1]) # Length of time dimension
        data_detrended,data_fit = detrend_poly(xdim,nandict['cleaned_data'],order)
        #data_detrended         # [time x space]
        #data_fit               # [space x Time]
        #nandict['cleaned_data' # [space x time]
        
        # # Debug Plot
        ii = 77
        fig,ax = plt.subplots(1,1)
        ax.plot(xdim,data_detrended[:,ii],label="Detrended",color='blue')
        ax.plot(xdim,data_fit[ii,:],label="Fit",color="red")
        ax.plot(xdim,nandict['cleaned_data'][ii,:],label="Raw",color='gray',ls='dashed')
        ax.legend()
        
        # Replace Detrended data in original array
        arrout = np.zeros((nlon*nlat,ntime)) * np.nan
        arrout[nandict['ok_indices'],:] = data_detrended.T
        arrout = arrout.reshape(nlon,nlat,ntime).transpose(2,1,0) # Flip to time x lat x lon
        
        if return_fit:
            fitout = np.zeros((nlon*nlat,ntime)) * np.nan
            fitout[nandict['ok_indices'],:] = data_fit
            fitout = fitout.reshape(nlon,nlat,ntime).transpose(2,1,0)
    
    # Replace into data array
    
    
    # Prepare Output as DataArrays # [(time) x lat x lon]
    if reshape_flag is False: # Directly transpose and assign coords [time x lat x lon]
        coords_full         = dict(time=invar.time,lat=invar.lat,lon=invar.lon)
        if regress_monthly: # Add "mon" coordinate for monthly regression
            coords          = dict(mon=np.arange(1,13,1),lat=invar.lat,lon=invar.lon)
        else:
            coords          = dict(lat=invar.lat,lon=invar.lon)
        
        da_detrend      = xr.DataArray(arrout,coords=coords_full,dims=coords_full,name=invar.name)
        if return_fit:
            da_fit          = xr.DataArray(fitout,coords=coords_full,dims=coords_full,name='fit')
    
    else: # Need to undo reshaping and reassign old coords...
        da_detrend      = reshape_2d_ds(arrout,invar,reshape_output[2],reshape_output[1])
        da_fit          = reshape_2d_ds(fitout,invar,reshape_output[2],reshape_output[1])
        
    if return_fit:
        dsout = xr.merge([da_detrend,da_fit])
    else:
        dsout = da_detrend
    return dsout

def detrend_by_regression(invar,in_ts,regress_monthly=False,return_pattern_only=False):
    # Given an DataArray [invar] and Timeseries [in_ts]
    # Detrend the timeseries by regression
    
    # Change to [lon x lat x time]
    reshape_flag = False
    try:
        invar       = invar.transpose('lon','lat','time')
        invar_arr   = invar.data # [lon x lat x time]
        
    except:
        print("Warning, input is not 3d or doesn't have ('lon','lat','time')")
        reshape_output = make_2d_ds(invar,keepdim='time') #[1 x otherdims x time]
        invar_arr      = reshape_output[0].data
        reshape_flag = True
    ints_arr         = in_ts.data # [time]
    
    if regress_monthly: # Do regression separately for each month
        
        nlon,nlat,ntime = invar_arr.shape
        nyr             = int(ntime/12)
        ints_monyr      = ints_arr.reshape(nyr,12)
        invar_monyr     = invar_arr.reshape(nlon,nlat,nyr,12) # [lat x lon x yr x mon]
        
        betas      = []
        intercepts = []
        ymodels    = []
        ydetrends  = []
        sigmasks   = []
        for im in range(12):
            
            outdict     = regress_ttest(invar_monyr[:,:,:,im],ints_monyr[:,im])
            beta        = outdict['regression_coeff'] # Lon x Lat
            intercept   = outdict['intercept'] 
            
            
            # Remove the Trend
            ymodel      = beta[:,:,None] * ints_monyr[None,None,:,im] + intercept[:,:,None]
            ydetrend    = invar_monyr[:,:,:,im] - ymodel
            
            betas.append(beta)
            intercepts.append(intercept)
            ymodels.append(ymodel)
            ydetrends.append(ydetrend)
            sigmasks.append(outdict['sigmask'])
        
        beta        = np.array(betas)       # [Month x lon x lat]
        intercept   = np.array(intercepts)  # [Month x lon x lat]
        ymodel      = np.array(ymodels)     # [Month x lon x lat x yr]
        ydetrend    = np.array(ydetrends)   # [Month x lon x lat x yr]
        sigmasks    = np.array(sigmasks)
        
        ymodel      = ymodel.transpose(1,2,3,0).reshape(nlon,nlat,ntime)
        ydetrend    = ydetrend.transpose(1,2,3,0).reshape(nlon,nlat,ntime)
        
        # Flip to [time x lat x lon]
        sigmask_out     = sigmasks.transpose(0,2,1) 
        beta            = beta.transpose(0,2,1)
        intercept       = intercept.transpose(0,2,1)
        
    else:
        # Perform the regression (all months)
        outdict     = regress_ttest(invar_arr,ints_arr)
        beta        = outdict['regression_coeff'] # Lon x Lat
        intercept   = outdict['intercept'] 
        
        # Remove the Trend
        ymodel      = beta[:,:,None] * ints_arr[None,None,:] + intercept[:,:,None]
        ydetrend    = invar_arr - ymodel
        
        # Prepare for input [lat x lon]
        sigmask_out     = outdict['sigmask'].T
        beta            = beta.T
        intercept       = intercept.T
        
    # Prepare Output as DataArrays # [(time) x lat x lon]
    if reshape_flag is False: # Directly transpose and assign coords [time x lat x lon]
        coords_full     = dict(time=invar.time,lat=invar.lat,lon=invar.lon)
        if regress_monthly: # Add "mon" coordinate for monthly regression
            coords          = dict(mon=np.arange(1,13,1),lat=invar.lat,lon=invar.lon)
        else:
            coords          = dict(lat=invar.lat,lon=invar.lon)
        
        da_detrend      = xr.DataArray(ydetrend.transpose(2,1,0),coords=coords_full,dims=coords_full,name=invar.name)
        da_fit          = xr.DataArray(ymodel.transpose(2,1,0),coords=coords_full,dims=coords_full,name='fit')
        
        da_pattern      = xr.DataArray(beta,coords=coords,dims=coords,name='regression_pattern')
        da_intercept    = xr.DataArray(intercept,coords=coords,dims=coords,name='intercept')
        da_sig          = xr.DataArray(sigmask_out,coords=coords,dims=coords,name='sigmask')
        
    else: # Need to undo reshaping and reassign old coords...
        
        da_detrend      = reshape_2d_ds(ydetrend,invar,reshape_output[2],reshape_output[1])
        da_fit          = reshape_2d_ds(ymodel,invar,reshape_output[2],reshape_output[1])
        
        if regress_monthly: # Add additional "Month" variable at the end
            ref_da        = invar.isel(time=0).squeeze().expand_dims(dim={'mon':np.arange(1,13,1)},axis=-1)
            newshape      = list(reshape_output[2][:-1]) + [12,] # [Lon x Lat x Mon]
            newshape_dims = reshape_output[1][:-1] + ['mon',]
        else:
            ref_da        = invar.isel(time=0).squeeze() #
            newshape      = reshape_output[2][:-1] # Just Drop Time Dimension # [Lat x Lon]
            newshape_dims = reshape_output[1][:-1]
            
        da_pattern      = reshape_2d_ds(beta, ref_da, newshape, newshape_dims) # Drop time dim
        da_intercept    = reshape_2d_ds(intercept, ref_da, newshape, newshape_dims) # Drop time dim
        da_sig          = reshape_2d_ds(sigmask_out, ref_da, newshape, newshape_dims) # Drop time dim
    
    if return_pattern_only: # Do not return detrended variable
        dsout = xr.merge([da_fit,da_pattern,da_intercept,da_sig],compat='override',join='override')
    else:
        dsout = xr.merge([da_detrend,da_fit,da_pattern,da_intercept,da_sig],compat='override',join='override')
    
    return dsout


# Commented out version, prior to adding monthly support...
# def detrend_by_regression(invar,in_ts):
#     # Given an DataArray [invar] and Timeseries [in_ts]
#     # Detrend the timeseries by regression
    
#     # Change to [lon x lat x time]
#     reshape_flag = False
#     try:
#         invar       = invar.transpose('lon','lat','time')
#         invar_arr   = invar.data # [lon x lat x time]
#     except:
#         print("Warning, input is not 3d or doesn't have ('lon','lat','time')")
#         reshape_output = make_2d_ds(invar,keepdim='time')
#         invar_arr      = reshape_output[0].data
#         reshape_flag = True
    
#     ints_arr    = in_ts.data # [time]
    
#     # Perform the regression
#     outdict     = regress_ttest(invar_arr,ints_arr)
#     beta        = outdict['regression_coeff'] # Lon x Lat
#     intercept   = outdict['intercept'] 
    
#     # Remove the Trend
#     ymodel      = beta[:,:,None] * ints_arr[None,None,:] + intercept[:,:,None]
#     ydetrend    = invar_arr - ymodel
    
#     # Prepare Output as DataArrays # [(time) x lat x lon]
#     if reshape_flag is False:
#         coords_full     = dict(time=invar.time,lat=invar.lat,lon=invar.lon)
#         coords          = dict(lat=invar.lat,lon=invar.lon)
#         da_detrend      = xr.DataArray(ydetrend.transpose(2,1,0),coords=coords_full,dims=coords_full,name=invar.name)
#         da_fit          = xr.DataArray(ymodel.transpose(2,1,0),coords=coords_full,dims=coords_full,name='fit')
#         da_pattern      = xr.DataArray(beta.T,coords=coords,dims=coords,name='regression_pattern')
#         da_intercept    = xr.DataArray(intercept.T,coords=coords,dims=coords,name='intercept')
#         da_sig          = xr.DataArray(outdict['sigmask'].T,coords=coords,dims=coords,name='sigmask')
#     else:
#         da_detrend      = reshape_2d_ds(ydetrend,invar,reshape_output[2],reshape_output[1])
#         da_fit          = reshape_2d_ds(ymodel,invar,reshape_output[2],reshape_output[1])
#         da_pattern      = reshape_2d_ds(beta.T,invar.isel(time=0).squeeze(),reshape_output[2][:-1],reshape_output[1][:-1]) # Drop time dim
#         da_intercept    = reshape_2d_ds(intercept.T,invar.isel(time=0).squeeze(),reshape_output[2][:-1],reshape_output[1][:-1]) # Drop time dim
#         da_sig          = reshape_2d_ds(outdict['sigmask'].T,invar.isel(time=0).squeeze(),reshape_output[2][:-1],reshape_output[1][:-1]) # Drop time dim
#     dsout = xr.merge([da_detrend,da_fit,da_pattern,da_intercept,da_sig],compat='override',join='override')
    
#     return dsout

def make_2d_ds(ds,keepdim='time'):
    
    
    # Get List of Dims, move time to front
    oldshape      = ds.shape
    dimnames      = ds.dims
    otherdims     = list(dimnames)#.remove(keepdim)
    otherdims.remove(keepdim)
    newdims        = otherdims  + [keepdim,] 
    dstranspose    = ds.transpose(*newdims)
    
    # Convert to 3D intput where time is last [1 x otherdims x time]
    dsarr          = dstranspose.data
    oldshape_trans = dsarr.shape
    ntime          = oldshape_trans[-1]
    notherdims     = np.array(oldshape_trans[:-1]).prod()
    dsarr          = dsarr.reshape(1,notherdims,ntime)
    coords_rs      = dict(lon=[1,],lat=np.arange(notherdims),time=ds.time)
    dsreshape      = xr.DataArray(dsarr,dims=coords_rs,coords=coords_rs,name=ds.name)
    return dsreshape,newdims,oldshape_trans

def reshape_2d_ds(inarr,ds_ori,oldshape_trans,newdims):
    inarr_rs    = inarr.reshape(oldshape_trans)
    coords_new  = {}
    for dname in newdims:
        coords_new[dname] = ds_ori[dname]
    #coords_new = ds_ori.transpose(*newdims).coords 
    da_inarr_rs = xr.DataArray(inarr_rs,coords=ds_ori.coords,dims=coords_new,name=ds_ori.name,)
    
    da_inarr_rs = da_inarr_rs.transpose(*ds_ori.dims)
    return da_inarr_rs

    #newdims,oldshape_trans


#%% ~ Classification/Grouping

def make_classes_nd(y,thresholds,exact_value=False,reverse=False,dim=0,debug=False,return_thres=False):
    """
    Makes classes based on given thresholds. Loops over values in ND number of datagroups
    and sets thresholds for each one

    Parameters
    ----------
    y : ARRAY
        Labels to classify. Defaults to first dimension if y is ND
    thresholds : ARRAY
        1D Array of thresholds to partition the data
    exact_value: BOOL, optional
        Set to True to use the exact value in thresholds (rather than scaling by
                                                          standard deviation)
        
    dim : INT
        Dimension to classify over
    Returns
    -------
    y_class : ARRAY [samples, datagroup, class]
        Classified samples, where the first dimension contains the class number
        (an integer representing each threshold)

    """
    
    # Make target array 2-D, and bring target dim to front (axis 0)
    if len(y.shape) > 1:
        if debug:
            print("Combining dimensions")
        reshape_flag  = True
        oldshape      = y.shape
        # Move target dimension to front, and combine other dims
        y,reorderdim    = dim2front(y,dim,combine=True,verbose=False,return_neworder=True) # [values, datagroup]
    else:
        reshape_flag = False
        y    = y[:,None]
    npts = y.shape[1] # Last axis is the number of points
    
    # Compute Thresholds for each datagroup (assume target dim = 0 here onwards)
    nthres = len(thresholds)
    if exact_value is False: # Scale thresholds by standard deviation
        y_std = np.std(y,axis=0) # Get standard deviation along first axis # [npts,]
        thresholds = np.array(thresholds)[:,None] * y_std[None,:] # [thres x npts]
    else:
        thresholds = np.array(thresholds)[:,None]
    
    y_class = np.zeros((y.shape[0],npts)) # [sample,datagroup]
    if nthres == 1: # For single threshold cases
        # Get the threshold
        thres             = thresholds[0,:]
        
        # Get masks of values above/below
        below_mask = y <= thres[None,:]
        above_mask = y >  thres[None,:]
        
        # Assign Classes
        y_class[below_mask] = 0
        y_class[above_mask] = 1
        
    else:
        for t in range(nthres+1): # Multi-threshold Case
            if t < nthres:
                thres = thresholds[t,:]
            else:
                thres = thresholds[-1,:]
                
            if reverse: # Assign class 0 to largest values
                tassign = nthres-t
            else:
                tassign = t
            
            if t == 0: # First threshold (Less than first value)
                mask = y <= thres[None,:]
            elif t == nthres: # Last threshold (Greater than last value)
                mask = y > thres[None,:]
            else: # Intermediate values (Between current and previous values)
                thres0 = thresholds[t-1,:]
                mask   = (y > thres0[None,:]) * (y <= thres[None,:])
            y_class[mask] = tassign
            
    if debug:
        idt = np.random.choice(y.shape[1])
        #idt     = 22 # Plot index (random)
        ytest   = y[:100,idt]
        fig,ax  = plt.subplots(1,1)
        ax.plot(ytest,color="k")
        for i in range(nthres):
            ax.axhline(thresholds[i,idt],color="red",ls='dashed')
        ax.scatter(np.arange(0,100),ytest,c=y_class[:100,idt])
        
    if reshape_flag:
        y_class=restoredim(y_class,oldshape,reorderdim)
    if return_thres:
        return y_class,thresholds
    return y_class

def checkpoint(checkpoints,invar,debug=True):
    """
    Groups values of invar between values specified in checkpoints and
    returns a list where each element is the indices of the group

    Parameters
    ----------
    checkpoints : ARRAY
        1-D Array of checkpoint/threshold values. Checks (z-1 < x <= z)
    invar : ARRAY
        1-D Array of values to check
    debug : TYPE, optional
        True to print messages (default)

    Returns
    -------
    ids_all : TYPE
        Indices of invar for each group

    """
    
    ids_all = []
    for z in range(len(checkpoints)+1):
        if z == 0: # <= First value
            if debug:
                print("Looking for indices <= %i"% (checkpoints[z]))
            ids = np.where(invar <= checkpoints[z])[0]
            
        elif z == len(checkpoints): # > Last value
            if debug:
                print("Looking for indices > %i"% (checkpoints[z-1]))
            ids = np.where(invar > checkpoints[z-1])[0]
            if len(ids)==0:
                continue
            else:
                print("Found %s"% str(np.array(invar)[ids]))
                ids_all.append(ids)
            return ids_all # Exit on last value
        else: # Check values between z-1, z
            ids = np.where((invar > checkpoints[z-1]) & (invar <= checkpoints[z]))[0]
            if debug and (z%100 == 0) or (z < 10) or (z>len(checkpoints)-10):
                print("Looking for indices %i < x <= %i" % (checkpoints[z-1],checkpoints[z]))
        
        
        if len(ids)==0:
            continue
        else:
            if debug and (z%100 == 0) or (z < 10) or (z>len(checkpoints)-10):
                print("Found %s"% str(np.array(invar)[ids]))
            ids_all.append(ids)
    return ids_all
    
def classify_bythres(var_in,thres,use_quantile=True,debug=False):
    """
    Parameters
    ----------
    var_in : DataArray [lat x lon]
        2-D input with the values to classify by
    thres  : ARRAY or LIST (Numeric)
        List of either quantiles or threshold values
    use_quantile : BOOL, optional
        True to compute quantiles over space. The default is True.
    debug : BOOL, optional
        True to do quick visualization of output. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if use_quantile: # Get Quantiles based on flattened Data
        var_flat    = var_in.data.flatten()
        thresvals   = np.nanquantile(var_flat,thres)
    else: # Or just used provided values
        thresvals   = thres

    # Make a map of the values
    classmap = xr.zeros_like(var_in)
    
    boolmap = []
    nthres  = len(thres) + 1
    for nn in range(nthres):
        print(nn)
        if nn == 0:
            print("x < %f" % (thresvals[nn]))
            #boolmap.append(var_in < thresvals[nn])
            classmap = xr.where(var_in < thresvals[nn],nn,classmap)
        elif nn == (nthres-1):
            print("x >= %f" % (thresvals[nn-1]))
            #boolmap.append(var_in >= thresvals[nn-1])
            classmap = xr.where(var_in >= thresvals[nn-1],nn,classmap)
        else:
            print("%f < x <= %f" % (thresvals[nn-1],thresvals[nn]))
            #boolmap.append( (var_in > thresvals[nn-1]) & (var_in <= thresvals[nn]))
            classmap = xr.where( (var_in > thresvals[nn-1]) & (var_in <= thresvals[nn]),nn,classmap)
        if debug:
            classmap.plot(),plt.title("nn=%i" % nn),plt.show()
    # Set NaN points to NaN
    classmap = xr.where(np.isnan(var_in),np.nan,classmap)
    if debug:
        classmap.plot(),plt.show()
    if use_quantile:
        return classmap,thresvals
    return classmap


def make_thres_labels(thresvals):
    # Make labels for a series of input values
    # Assumes equal to on the larger value...
    labels = []
    nthres = len(thresvals) + 1
    for nn in range(nthres):
        if nn == 0:
            lab = "x < %f" % (thresvals[nn])
        elif nn == (nthres-1):
            lab = "x >= %f" % (thresvals[nn-1])
        else:
            lab = "%f < x <= %f" % (thresvals[nn-1],thresvals[nn])
        labels.append(lab)
    return labels


#%% ~ Spatial Analysis/Wrangling

def lon360to180(lon360,var,autoreshape=False,debug=True):
    """
    Convert Longitude from Degrees East to Degrees West 
    Inputs:
        1. lon360 - array with longitude in degrees east
        2. var    - corresponding variable [lon x lat x time]
        3. autoreshape - BOOL, reshape variable autocmatically if size(var) > 3
    """
    
    # Reshape to combine dimensions
    dimflag = False 
    if autoreshape:
        var,vshape,dimflag=combine_dims(var,2,debug=True)
        
    kw = np.where(lon360 >= 180)[0]
    ke = np.where(lon360 < 180)[0]
    lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
    if var is None:
        return lon180
    var = np.concatenate((var[kw,...],var[ke,...]),0)
    if dimflag:
        var = var.reshape(vshape)
    
    return lon180,var

def lon180to360(lon180,var,autoreshape=False,debug=True):
    """
    Convert Longitude from Degrees West to Degrees East 
    Inputs:
        1. lon180 - array with longitude in degrees west
        2. var    - corresponding variable [lon x lat x time]
        3. autoreshape - BOOL, reshape variable autocmatically if size(var) > 3
    
    """
    
    # Reshape to combine dimensions
    dimflag = False 
    if autoreshape:
        var,vshape,dimflag=combine_dims(var,2,debug=True)
        
    kw = np.where(lon180 < 0)[0]
    ke = np.where(lon180 >= 0)[0]
    lon360 = np.concatenate((lon180[ke],lon180[kw]+360),0)
    if var is None:
        return lon360
    var = np.concatenate((var[ke,...],var[kw,...]),0)
    
    if dimflag:
        var = var.reshape(vshape)
    
    return lon360,var

def lon360to180_xr(ds,lonname='lon'):
    # Based on https://stackoverflow.com/questions/53345442/about-changing-longitude-array-from-0-360-to-180-to-180-with-python-xarray
    ds.coords[lonname] = (ds.coords[lonname] + 180) % 360 - 180
    ds = ds.sortby(ds[lonname])
    return ds

def lon360to180_ds(ds,lonname='longitude'):
    """Same as above but for datasets. Copied from stackexchange
    https://stackoverflow.com/questions/53121089/regridding-coordinates-with-python-xarray
    """
    newcoord = {lonname : ((ds[lonname] + 180) % 360) - 180}
    ds = ds.assign_coords(newcoord).sortby(lonname)
    return ds

def linear_crop(invar,lat,lon,ptstart,ptend,belowline=True,along_x=True,debug=False):
    
    """
    Remove points above/below a selected line. Taken from landicemask_comparison,
    and used for masking out Pacific Ocean points.
    
    Inputs
    ------
        1. invar (ARRAY: [lon x lat])   :: Input variable
        2. lat (ARRAY: [lat])           :: Latitudes
        3. lon (ARRAY: [lon])           :: Longitudes
        4. ptstart (LIST: [Lon,Lat])    :: Start point for crop line (upper left)
        5. ptend (LIST: [Lon,Lat])      :: End point for crop line (lower right)
        6. belowline (BOOL)             :: True to remove points below line, False=Above
        7. along_x (BOOL)               :: True to move zonally (False=meridionally)
        8. debug (BOOL)                 :: True to print outputs
    Output
    ------
        1. cropped_mask (ARRAY: [lon x lat])   :: Edited variable with points removed
    """
    
    # Get the indices for starting and ending points
    xstart,ystart = find_latlon(ptstart[0],ptstart[1],lon,lat)
    xend,yend     = find_latlon(ptend[0],ptend[1],lon,lat)
    
    # Copy variable
    cropped_mask         = invar.copy()
    
    # Starting X,Y for loop
    x0 = ptstart[0] 
    y0 = ptstart[1]
        
    # Moving zonally, cut out points
    if along_x:
        
        # Calculate the dx and dy moving zonally
        dx = (ptend[0]-ptstart[0])/ np.abs((xend-xstart))
        dy = (ptend[1]-ptstart[1])/ np.abs((xend-xstart)) # Abs() for decreasing values
        
        # Starting index for loop
        kx = xstart
        for i in range(np.abs(xend-xstart)):
            if belowline:
                kremove = np.where(lat <= y0)
                word = "less"
            else: # Remove points above the line
                kremove = np.where(lat >= y0)
                word = "greater"
            cropped_mask[kx,kremove] = np.nan
            if debug:
                print("Location is %i Lon, eliminating %i points with Lat %s than %i" % (x0,len(kremove),word,y0))
            # Add to Index
            kx += (1*np.sign(xend-xstart))
            y0 += dy
            x0 += dx
    else: # Move meridionally
        # Calculate the dx or dy moving meridionally
        dx = (ptend[0]-ptstart[0])/ np.abs((yend-ystart))
        dy = (ptend[1]-ptstart[1])/ np.abs((yend-ystart)) # Abs() for decreasing values
        print(dx,dy)
        # Starting X,y, and y-index
        ky = ystart
        for i in range(np.abs(yend-ystart)):
            if belowline:
                kremove = np.where(lon <= x0)
                word = "less"
            else: # Remove points above the line
                kremove = np.where(lon >= x0)
                word = "greater"
            cropped_mask[kremove,ky] = np.nan
            if debug:
                print("Location is %i Lat, eliminating %i points with Lon %s than %i" % (y0,len(kremove),word,x0))
            # Add to Index
            ky += (1*np.sign(yend-ystart))
            y0 += dy
            x0 += dx
    if debug: # Make a plot
        pts     = np.vstack([ptstart,ptend])
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
        pcm = ax.pcolormesh(lon,lat,cropped_mask[:,:].T)
        ax.plot(pts[:,0],pts[:,1],color="y",marker="x") 
        fig.colorbar(pcm,ax=ax)
    return cropped_mask

def calc_dx_dy(longitude,latitude,centered=False):
    ''' 
        This definition calculates the distance between grid points (in meters)
        that are in a latitude/longitude format.
        
        Function from: https://github.com/Unidata/MetPy/issues/288
        added "centered" option to double the distance for centered-difference
        
        Equations from:
        http://andrew.hedges.name/experiments/haversine/

        dy should be close to 55600 m
        dx at pole should be 0 m
        dx at equator should be close to 55600 m
        
        Accepts, 1D arrays for latitude and longitude
        
        Returns: dx, dy; 2D arrays of distances between grid points 
                                    in the x and y direction in meters 
    '''
    dlat = np.abs(latitude[1]-latitude[0])*np.pi/180
    if centered:
        dlat *= 2
    dy   = 2*(np.arctan2(np.sqrt((np.sin(dlat/2))**2),np.sqrt(1-(np.sin(dlat/2))**2)))*6371000
    dy   = np.ones((latitude.shape[0],longitude.shape[0]))*dy

    dx = np.empty((latitude.shape))
    dlon = np.abs(longitude[1] - longitude[0])*np.pi/180
    if centered:
        dlon *= 2
    for i in range(latitude.shape[0]):
        # Apply cos^2 latitude weight
        a = (np.cos(latitude[i]*np.pi/180)*np.cos(latitude[i]*np.pi/180)*np.sin(dlon/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a) )
        dx[i] = c * 6371000
    dx = np.repeat(dx[:,np.newaxis],longitude.shape,axis=1)
    return dx, dy

"""
------------------------------
|||  Statistical Analysis  ||| ****************************************************
------------------------------
"""

#%% ~ Regression
def regress_2d(A,B,nanwarn=1,verbose=True):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    Note that if A and B are of the same size, assumes axis 1 of A will be regressed to axis 0 of B
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    
    # Determine if A or B is 2D and find anomalies
    bothND = False # By default, assume both A and B are not 2-D.
    # Note: need to rewrite function such that this wont be a concern...
    
    # Accounting for the fact that I dont check for equal dimensions below..
    #B = B.squeeze()
    #A = A.squeeze() Commented out below because I still need to fix some things
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        if nanwarn == 1:
            print("NaN Values Detected...")
        
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        

        # A is [P x N], B is [N x M]
        elif len(A.shape) == len(B.shape):
            if verbose:
                print("Note, both A and B are 2-D...")
            bothND = True
            if A.shape[1] != B.shape[0]:
                print("WARNING, Dimensions not matching...")
                print("A is %s, B is %s" % (str(A.shape),str(B.shape)))
                print("Detecting common dimension")
                # Get intersecting indices 
                intersect, ind_a, ind_b = np.intersect1d(A.shape,B.shape, return_indices=True)
                if ind_a[0] == 0: # A is [N x P]
                    A = A.T # Transpose to [P x N]
                if ind_b[0] == 1: # B is [M x N]
                    B = B.T # Transpose to [N x M]
                print("New dims: A is %s, B is %s" % (str(A.shape),str(B.shape)))
                
            # Set axis for summing/averaging
            a_axis = 1 # Assumes dim 1 of A will be regressed to dim 0 of b
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis,keepdims=True)#[:,None] # Anomalize w.r.t. dim 1 of A
            Banom = B - np.nanmean(B,axis=b_axis,keepdims=True)# # Anonalize w.r.t. dim 0 of B
            
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom  = np.nansum(Aanom2,axis=a_axis,keepdims=True)     # Sum along dim 1 of A (lets say this is time)
        
        # Calculate Beta
        #if 
        if len(denom.shape)==1 or not bothND: # same as both not ND
            print("Adding singleton dimension to denom")
            denom = denom[:,None]
        beta = Aanom @ Banom / denom#[:,None] # Denom is [A[mode,time]@ B[time x space]], output is [mode x pts]
        
        b = (np.nansum(B,axis=b_axis,keepdims=True) - beta * np.nansum(A,axis=a_axis,keepdims=True))/A.shape[a_axis]
        # b is [mode x pts] [or P x M]
            
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
                
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
            
        # A is [P x N], B is [N x M]
        elif len(A.shape) == len(B.shape):
            if verbose:
                print("Note, both A and B are 2-D...")
            bothND = True
            if A.shape[1] != B.shape[0]:
                print("WARNING, Dimensions not matching...")
                print("A is %s, B is %s" % (str(A.shape),str(B.shape)))
                print("Detecting common dimension")
                # Get intersecting indices 
                intersect, ind_a, ind_b = np.intersect1d(A.shape,B.shape, return_indices=True)
                if ind_a[0] == 0: # A is [N x P]
                    A = A.T # Transpose to [P x N]
                if ind_b[0] == 1: # B is [M x N]
                    B = B.T # Transpose to [N x M]
                print("New dims: A is %s, B is %s" % (str(A.shape),str(B.shape)))
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)[None,:]

        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom  = np.sum(Aanom2,axis=a_axis,keepdims=True)
        if not bothND:
            
            denom = denom[:,None] # Broadcast
            
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        if bothND:
            b = (np.sum(B,axis=b_axis)[None,:] - beta * np.sum(A,axis=a_axis)[:,None])/A.shape[a_axis]
        else:
            b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    return beta,b

def regress2ts(var,ts,normalizeall=0,method=1,nanwarn=1,verbose=True):
    # var = [lon x lat x time], ts = [time]
    
    # Anomalize and normalize the data (time series is assumed to have been normalized)
    if normalizeall == 1:
        varmean = np.nanmean(var,2)
        varstd  = np.nanstd(var,2)
        var = (var - varmean[:,:,None]) /varstd[:,:,None]
        
    # Get variable shapes
    if len(var.shape) > 2:
        reshapeflag = True
        if verbose:
            print("Lon and lat are uncombined!")
        londim = var.shape[0]   
        latdim = var.shape[1]
    else:
        reshapeflag=False
    
    # 1st method is matrix multiplication
    if method == 1:
        
        # Combine the spatial dimensions 
        if len(var.shape)>2:
            var = np.reshape(var,(londim*latdim,var.shape[2]))
        
        
        # Find Nan Points
        # sumvar = np.sum(var,1)
        
        # # Find indices of nan pts and non-nan (ok) pts
        # nanpts = np.isnan(sumvar)
        # okpts  = np.invert(nanpts)
        
        # # Drop nan pt flons and reshape again to separate space and time dimensions
        # var_ok = var[okpts,:]
        #var[np.isnan(var)] = 0
        
        # Perform regression
        #var_reg = np.matmul(np.ma.anomalies(var,axis=1),np.ma.anomalies(ts,axis=0))/len(ts)
        var_reg,_ = regress_2d(ts,var,nanwarn=nanwarn)
        
        
        # Reshape to match lon x lat dim
        if reshapeflag:
            var_reg = np.reshape(var_reg,(londim,latdim))
    
    # 2nd method is looping point by point
    elif method == 2:
        
        
        # Preallocate       
        var_reg = np.zeros((londim,latdim))
        
        # Loop lat and long
        for o in range(londim):
            for a in range(latdim):
                
                # Get time series for that period
                vartime = np.squeeze(var[o,a,:])
                
                # Skip nan points
                if any(np.isnan(vartime)):
                    var_reg[o,a]=np.nan
                    continue
                
                # Perform regression 
                r = np.polyfit(ts,vartime,1)
                #r=stats.linregress(vartime,ts)
                var_reg[o,a] = r[0]
                #var_reg[o,a]=stats.pearsonr(vartime,ts)[0]
    
    return var_reg

#%% ~ Lead/Lag Analysis
def calc_lagcovar(var1,var2,lags,basemonth,detrendopt,yr_mask=None,debug=True,
                  return_values=False,spearman=False):
    """
    Calculate lag-lead relationship between two monthly time series with the
    form [mon x yr]. Lag 0 is set by basemonth
    
    Correlation will be calculated for each lag in lags (lead indicate by
    negative lags)
    
    Set detrendopt to 1 for a linear detrend of each time series.
    
    
    Inputs:
        1) var1: Monthly timeseries for variable 1 [mon x year]
        2) var2: Monthly timeseries for variable 2 [mon x year]
        3) lags: lags and leads to include
        4) basemonth: lag 0 month
        5) detrendopt: 1 for linear detrend of both variables
        6) yr_mask : ARRAY of indices for selected years
        7) debug : Print check messages
        8) return_values [BOOL] : Return the lagged values and base values
    
    Outputs:
        1) corr_ts: lag-lead correlation values of size [lags]
        2) yr_count : print the count of years
        3) varbase : [yrs] Values of monthly anomalies for reference month
        4) varlags : [lag][yrs] Monthly anomalies for each lag month
    
    Dependencies:
        numpy as np
        scipy signal,stats
    
    """
    
    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length
    totyr = var1.shape[1]
    
    # Get total number of year crossings from lag
    endmonth = basemonth + lagdim-1
    nlagyr   = int(np.ceil(endmonth/12)) #  Ignore zero lag (-1)
    
    if debug:
        print("Lags spans %i mon (%i yrs) starting from mon %i" % (endmonth,nlagyr,basemonth))
        
    # Get Indices for each year
    if yr_mask is not None:
        # Drop any indices that are larger than the limit
        # nlagyr-1 accounts for the base year...
        # totyr-1 accounts for indexing
        yr_mask_clean = np.array([yr for yr in yr_mask if (yr+nlagyr-1) < totyr])
        
        if debug:
            n_drop = np.setdiff1d(yr_mask,yr_mask_clean)
            print("Dropped the following years: %s" % str(n_drop))
        
        yr_ids  = [] # Indices to 
        for yr in range(nlagyr):
            
            # Apply year-lag to index
            yr_ids.append(yr_mask_clean + yr)
    
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.ceil(len(np.where(lags < 0)[0])/12))
    lagsize = int(np.ceil(len(np.where(lags > 0)[0])/12))
    
    # Detrend variables if option is set
    if detrendopt == 1:
        var1 = signal.detrend(var1,1,type='linear')
        var2 = signal.detrend(var2,1,type='linear')
    
    # Get base timeseries to perform the autocorrelation on
    if yr_mask is not None:
        varbase = var1[basemonth-1,yr_ids[0]] # Anomalies from starting year
    else: # Use old indexing approach
        base_ts = np.arange(0+leadsize,totyr-lagsize)
        varbase = var1[basemonth-1,base_ts]
        
    # Preallocate Variable to store correlations
    corr_ts = np.zeros(lagdim)
    
    # Set some counters
    nxtyr = 0
    addyr = 0
    modswitch = 0
    
    varlags = [] # Save for returning later
    for i in lags:

        lagm = (basemonth + i)%12
        
        if lagm == 0:
            lagm = 12
            addyr = 1         # Flag to add to nxtyr
            modswitch = i+1   # Add year on lag = modswitch
            
        if addyr == 1 and i == modswitch:
            if debug:
                print('adding year on '+ str(i))
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        if yr_mask is not None:
            varlag = var2[lagm-1,yr_ids[nxtyr]]
            if debug:
                print("For lag %i (m=%i), first (last) indexed year is %i (%i) " % (i,lagm,yr_ids[nxtyr][0],yr_ids[nxtyr][-1]))
        else:
            lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
            varlag = var2[lagm-1,lag_ts]
            if debug:
                print("For lag %i (m=%i), lag_ts is between %i and %i" % (i,lagm,lag_ts[0],lag_ts[-1]))
            
        #varbase = varbase - varbase.mean()
        #varlag  = varlag - varlag.mean()
        #print("Lag %i Mean is %i ")
        
        # Calculate correlation
        if spearman == 1:
            corr_ts[i] = stats.spearmanr(varbase,varlag)[0]
            #corr_ts[i] = stats.kendalltau(varbase,varlag)[0]
        elif spearman == 2:
            corr_ts[i] = stats.kendalltau(varbase,varlag)[0]
        else:
            corr_ts[i] = stats.pearsonr(varbase,varlag)[0]
        varlags.append(varlag)
        
    if return_values:
        return corr_ts,varbase,varlags
    if yr_mask is not None:
        return corr_ts,len(yr_ids[-1]) # Return count of years as well
    return corr_ts

def calc_lagcovar_nd(var1,var2,lags,basemonth,detrendopt):    
    
    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length # [mon x yr x npts]
    totyr = var1.shape[1]
    npts  = var1.shape[2]
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.ceil(len(np.where(lags < 0)[0])/12))
    lagsize  = int(np.ceil(len(np.where(lags > 0)[0])/12))
    
    # Detrend variables if option is set
    if detrendopt == 1:
        var1 = signal.detrend(var1,1,type='linear')
        var2 = signal.detrend(var2,1,type='linear')
    
    # Get base timeseries to perform the autocorrelation on
    base_ts = np.arange(0+leadsize,totyr-lagsize)
    varbase = var1[basemonth-1,base_ts,:]
        
    # Preallocate Variable to store correlations
    corr_ts = np.zeros((lagdim,npts))
    
    # Set some counters
    nxtyr = 0
    addyr = 0
    modswitch = 0
    
    for i in lags:
        
        lagm = (basemonth + i)%12
        
        if lagm == 0:
            lagm = 12
            addyr = 1         # Flag to add to nxtyr
            modswitch = i+1   # Add year on next lag = modswitch
            
        if addyr == 1 and i == modswitch:
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
        varlag = var2[lagm-1,lag_ts,:]
        
        # Calculate correlation
        corr_ts[i,:] = pearsonr_2d(varbase,varlag,0)       
    return corr_ts
    
def calc_lag_covar_ann(var1,var2,lags,dim,detrendopt,verbose=True):
    """
    Calculate lag **correlation**
    var1 is lagged, var2 remains the same (base)
    
    """
    # Move time to the first dimension (assume var1.shape==var2.shape)
    invars      = [var1,var2]
    oldshape    = var1.shape
    reshapevars = []
    for v in invars:
        vreshape,neworder = dim2front(v,dim,combine=True,return_neworder=True)
        reshapevars.append(vreshape)
    
    # Remove Nan Points (if any are found)
    
    # Get total number of lags
    var1,var2=reshapevars
    lagdim   = len(lags)
    if lagdim > var1.shape[0]:
        if verbose:
            print("\tWarning! maximum lag  (%i) exceeds the timeseries length (%i)" % (lagdim,var1.shape[0]))
    
    # Get timeseries length # [yr x npts]
    ntime  = var1.shape[0]
    npts   = var1.shape[1]
    
    # Detrend variables if option is set
    if detrendopt == 1:
        if verbose:
            print("\tWarning! Variable will be detrended linearly.")
        var1 = signal.detrend(var1,0,type='linear')
        var2 = signal.detrend(var2,0,type='linear')
    
    # Preallocate
    corr_ts        = np.zeros((lagdim,npts)) * np.nan
    window_lengths = []
    for l,lag in enumerate(lags):
        varlag   = var1[lag:,:]
        varbase  = var2[:(ntime-lag),:]
        window_lengths.append(varlag.shape[0])
        
        # Calculate correlation
        corr_ts[l,:] = pearsonr_2d(varbase,varlag,0)    
    
    # Replace back into old shape
    size_combined_dims = tuple(np.array(oldshape)[neworder][1:]) # Get other dims
    reshape_corr       = (lagdim,) + size_combined_dims
    corr_ts            = corr_ts.reshape(reshape_corr)
    
    return corr_ts,window_lengths

def calc_conflag(ac,conf,tails,n):
    """
    Calculate Confidence Intervals for autocorrelation function

    Parameters
    ----------
    ac : ARRAY [nlags,npts]
        Autocorrelation values by lag
    conf : NUMERIC
        Confidence level (ex. 0.95)
    tails : INT
        # of tails to consider
    n : INT
        Degrees of Freedom

    Returns
    -------
    cflags : ARRAY [nlags x 2 (upper/lower) x npts]
        Confidence interval for each lag

    """
    ND = False
    if len(ac.shape) > 1:
        ND = True
    
    if ND:
        nlags,npts = ac.shape
        cflags = np.zeros((nlags,2,npts)) # [Lag x Conf x Npts]
        
    else:
        nlags = len(ac)
        cflags = np.zeros((nlags,2)) # [Lag x Conf]
    
    for l in range(nlags):
        rhoin = ac[l,...]
        cfout = calc_pearsonconf(rhoin,conf,tails,n) # [conf x npts]
        cflags[l,...] = cfout
    return cflags

def tilebylag(kmonth,var,lags): 
    """
    Tile a monthly variable along a lag sequence,
    shifting to recenter lag0 = kmonth+1

    Parameters
    ----------
    kmonth : INT
        Index of month at lag 0 (ex. Jan, kmonth=0)
    var : ARRAY [12,]
        Monthly variable
    lags : ARRAY [nlags]
        Lags to tile along

    Returns
    -------
    vartile : ARRAY [nlags]
        Tiled and shifted variable

    """
    vartile = np.tile(np.array(var),int(np.floor(len(lags)/12))) 
    vartile = np.concatenate([np.roll(vartile,-kmonth),[var[kmonth]]])
    return vartile

def leadlag_corr(varbase,varlag,lags,corr_only=False):
    """
    Compute lead-lag correlation of two 1-D timeseries [time]
     - lags/leads are computed w.r.t. varbase
     - works with ufunc (see debug_leadlag_corr for examples)
    Inputs:
        varbase   (ARRAY, [time]) Base timeseries
        varlag    (ARRAY, [time]) Timeseries to lag/lead
        lags      (ARRAY, [lags]) Lags to compute lead/lag over
        corr_only (BOOL)          True to return just correlation
    Outputs:
        leadlags    (ARRAY, [lags]) Leads and Lags (flips and negates lags)
        leadlagcorr (ARRAY, [lags]) Correlation values
    
    """
    ntime = varbase.shape[0]
    nlags = len(lags)
    # Lags
    leadcorrs = []
    lagcorrs  = []
    for ii in range(nlags):
        lag     = lags[ii]
        lagcorr  = np.corrcoef(varbase[:(ntime-lag)],varlag[lag:])[0,1]
        leadcorr = np.corrcoef(varbase[lag:],varlag[:(ntime-lag)])[0,1]
        lagcorrs.append(lagcorr)
        leadcorrs.append(leadcorr)
    leadlagcorr = np.concatenate([np.flip(leadcorrs)[:-1],lagcorrs])
    leadlags    = np.concatenate([np.flip(-1*lags)[:-1],lags],)
    if corr_only:
        return leadlagcorr
    return leadlags,leadlagcorr



#%% ~ EOF Analysis
def eof_simple(pattern,N_mode,remove_timemean):
    """
    Simple EOF function based on script by Yu-Chiao Liang
    
    Inputs:
        1) pattern: Array of Space x Time [MxN], no NaNs
        2) N_mode:  Number of Modes to output
        3) remove_timemean: Set 1 to remove mean along N
    
    Outputs:
        1) eof: EOF patterns   [M x N_mode]
        2) pcs: PC time series [N x N_mode]
        3) varexp: % Variance explained [N_mode]
    
    Dependencies:
        import numpy as np
    
    """
    pattern1 = pattern.copy()
    nt = pattern1.shape[1] # Get time dimension size
    ns = pattern1.shape[0] # Get space dimension size
    
    if N_mode > nt:
        print("Warning, number of requested modes greater than length of time dimension. Adjusting to size of time.")
        N_mode = nt
    
    # Preallocate
    eofs = np.zeros((ns,N_mode))
    pcs  = np.zeros((nt,N_mode))
    varexp = np.zeros((N_mode))
    
    # Remove time mean if option is set
    if remove_timemean == 1:
        pattern1 = pattern1 - pattern1.mean(axis=1)[:,None] # Note, the None adds another dimension and helps with broadcasting
    
    # Compute SVD
    [U, sigma, V] = np.linalg.svd(pattern1, full_matrices=False)
    
    # Compute variance (total?)
    norm_sq_S = (sigma**2).sum()
    
    for II in range(N_mode):
        
        # Calculate explained variance
        varexp[II] = sigma[II]**2/norm_sq_S
        
        # Calculate PCs
        pcs[:,II] = np.squeeze(V[II,:]*np.sqrt(nt-1))
        
        # Calculate EOFs and normalize
        eofs[:,II] = np.squeeze(U[:,II]*sigma[II]/np.sqrt(nt-1))
    return eofs, pcs, varexp

def eof_filter(eofs,varexp,eof_thres,axis=0,return_all=False):
    """
    Discard modes above a certain variance explained percentage threshold
    Inputs:
        varexp    : [mode x mon]
        eofs      : [mode x mon x lat x lon]
        eof_thres : the percentange threshold (0.90=90%)
        axis      : axis of the mode dimension
    Outputs:
        eofs_filtered : [mode x mon x lat x lon] filtered eofs, with higher modes set to zero
        varexp_cumu   : [mode x mon] cumulative variance explained
        nmodes_needed : [mon] # of modes needed to reach threshold (first exceedence)
        varexps_filt  : [mode x mon] filtered variances explained by mode
    Note: Currently doesn't support DataArrays, see [calc_EOF_forcing_ERA5] for example.
    """
    varexp_cumu   = np.cumsum(varexp,axis=0)        # Cumulative sum of variance [Mode x Mon]
    above_thres   = varexp_cumu >= eof_thres        # Check exceedances [Mode x mon]
    nmodes_needed = np.argmax(above_thres,0)        # Get first exceedance
    
    eofs_filtered = eofs.copy()
    varexps_filt  = varexp.copy()
    for im in range(12):
        eofs_filtered[nmodes_needed[im]:,im,:,:] = 0 # Set modes above exceedence to zero
        varexps_filt[nmodes_needed[im]:,im] = 0
    if return_all:
        return eofs_filtered,varexp_cumu,nmodes_needed,varexps_filt
    # Here's a check"
    # print(np.sum(varexps_filt,0)) # Should be all below the variance threshold
    return eofs_filtered

#%% ~ Correlation
def pearsonr_2d(A,B,dim,returnsig=0,p=0.05,tails=2,dof='auto'):
    """
    Calculate Pearson's Correlation Coefficient for two 2-D Arrays
    along the specified dimension. Input arrays are anomalized.
    
    Option to perform students t-test.
    
    Inputs
    -------
    1) A : ARRAY
        First variable, 2D
    2) B : ARRAY
        Second variable, 2D (same axis arrangement as A)
    3) dim : INT
        Dimension to compute correlation along
    OPTIONAL ARGS
    4) returnsig : BOOL
        Return significance test result    
    5) p : FLOAT
        P-value for significance testing
    6) tails: INT
        Number of tails (1 or 2)
    7) dof: "auto" or INT
        Degress of freedom method. Set to "auto" for ndim - 2, or
        manually enter an integer value.

    Outputs
    --------
    1) rho : ARRAY
        Array of correlation coefficients
    OPTIONAL OUTPUTS (if returnsig=1)
    2) T : ARRAY
        T values from testing
    3) critval : FLOAT
        T - Critical value used as threshold
    4) sigtest : ARRAY of BOOLs
        Indicates points that passed significance threshold
    5) corrthres : FLOAT
        Correlation threshold corresponding to critval
    
    
    Calculates correlation between two matrices of equal size and also returns
    the significance test result
    
    Dependencies
        numpy as np
        scipy stats
    
    """
    
    # Find Anomaly
    Aanom = A - np.nanmean(A,dim)
    Banom = B - np.nanmean(B,dim)
    
    # Elementwise product of A and B
    AB = Aanom * Banom
    
    # Square A and B
    A2 = np.power(Aanom,2)
    B2 = np.power(Banom,2)
    
    # Compute Pearson's Correlation Coefficient
    rho = np.nansum(AB,dim) / np.sqrt(np.nansum(A2,dim)*np.nansum(B2,dim))
    
    if returnsig == 0:
        return rho
    else:
        
        # Determine DOF (more options to add later...)
        if dof == 'auto':
            # Assume N-2 dof
            n_eff = A.shape[dim]-2
        else:
            # Use manually supplied dof
            n_eff = dof
        
        # Compute p-value based on tails
        ptilde = p/tails
        
        # Compute T at each point
        T = rho * np.sqrt(n_eff / (1 - np.power(rho,2)))
        
        # Get threshold critical value
        critval = stats.t.ppf(1-ptilde,n_eff)
        
        # Perform test
        sigtest = np.where(np.abs(T) > critval)
        
        # Get critical correlation threshold
        corrthres = np.sqrt(1/ ((n_eff/np.power(critval,2))+1))
        
        return rho,T,critval,sigtest,corrthres

def calc_pearsonconf(rho,conf,tails,n):
    """
    rho   : pearson r [npts]
    conf  : confidence level
    tails : 1 or 2 tailed
    n     : Sample size
    """
    
    # Get z-critical
    alpha = (1-conf)/tails
    zcrit = stats.norm.ppf(1 - alpha)
    
    # Transform to z-space
    zprime = 0.5*np.log((1+rho)/(1-rho))
    
    # Calculate standard error
    SE     = 1/ np.sqrt(n-3)
    
    # Get Confidence
    z_lower = zprime-zcrit*SE
    z_upper = zprime+zcrit*SE
    
    # Convert back to r
    c_lower = np.tanh(z_lower)
    c_upper = np.tanh(z_upper)
    return c_lower,c_upper

def patterncorr(map1,map2):
    # From Taylor 2001,Eqn. 1, Ignore Area Weights
    # Calculate pattern correation between two 2d variables (lat x lon)
    
    # Get Non NaN values, Flatten, Array Size
    map1ok = map1.copy()
    map1ok = map1ok[~np.isnan(map1ok)].flatten()
    map2ok = map2.copy()
    map2ok = map2ok[~np.isnan(map2ok)].flatten()
    N      = len(map1ok)
    
    # Anomalize (remove spatial mean and calc spatial stdev)
    map1a  = map1ok - map1ok.mean()
    map2a  = map2ok - map2ok.mean()
    std1   = np.std(map1ok)
    std2   = np.std(map2ok)
    
    # Calculate
    R = 1/N*np.sum(map1a*map2a)/(std1*std2)
    return R

def patterncorr_nd(reference_map,target_maps,axis=0,return_N=False):
    # Vectorized version of patterncorr using array broadcasting
    # Computes along [axis] of target_maps (default=0)
    
    # Combine dimensions
    if axis != 0: # Move dim to the front
        target_maps = dim2front(target_maps,axis,verbose=True)
    N_space         = np.prod(reference_map.shape)
    reference_map   = reference_map.reshape(N_space) # [Space]
    target_maps     = target_maps.reshape((target_maps.shape[0],N_space)) # [Y x Space]
    
    # Remove NaNs (only use points where both are NOT NaN)
    okpts_ref                 = ~np.isnan(reference_map)
    okpts_targ                = ~np.isnan(target_maps.sum(0))
    okpts                     = okpts_ref * okpts_targ # Must be non-NaN in both
    reference_map,target_maps = reference_map[okpts],target_maps[:,okpts]
    N_space_ok                = reference_map.shape[0]
    
    # Anomalize
    refa     = reference_map - reference_map.mean()     # [Space]
    targa   = target_maps - target_maps.mean(1)[:,None] # [Y x Space]
    refstd  = np.std(refa)                              # [1]
    targstd = np.std(targa,1)                           # [Y x 1]
    
    # Compute
    R       = 1/N_space_ok * np.sum(refa[None,:] * targa,1) / (refstd * targstd) # [Y]
    if return_N:
        return R,N_space_ok
    return R    

def nancorr(ts1,ts2):
    # From https://stackoverflow.com/questions/31619578/numpy-corrcoef-compute-correlation-matrix-while-ignoring-missing-data
    return ma.corrcoef(ma.masked_invalid(ts1), ma.masked_invalid(ts2))

def calc_binwidth(invar,dim=0):
    """
    Use freedman-diaconis rule to compute bin width for invar along dimension/axis dim
                      IQR(x)
    Bin Width =  2 * -------
                      n^(1/3)
    """
    n   = invar.shape[dim]
    iqr = stats.iqr(invar,axis=dim,nan_policy='omit')
    return 2 * iqr / n**(1/3)

def expfit(acf,lags,lagmax):
    # Copief from reemergence/estimate_damping_fit/12.S992 Final Project
    expf3      = lambda t,b: np.exp(b*t)         # No c and A
    funcin     = expf3
    x = lags
    y = acf
    popt, pcov = sp.optimize.curve_fit(funcin, x[:(lagmax+1)], y[:(lagmax+1)],maxfev=5000)
    
    tau_inv = popt[0] # 1/tau (tau = timescale),. np.exp(tau_inv*t)
    acf_fit = expf3(lags,tau_inv)
    outdict = {'tau_inv':tau_inv, 'acf_fit':acf_fit}
    return outdict

def calc_monvar(ts,dim=0):
    # NOTE/WARNING: Currently just works if time is in the first dimension
    # Copied from viz_synth_stochmod_combine
    # Compute Monthly Variance for a timeseries, ignoring all NaNs
    _,tsmyr = calc_clim(ts,dim,returnts=1)
    monvar  = np.nanvar(tsmyr,axis=0)
    if monvar.shape[0] != 12:
        print("Warning, this function only supports the case where time is in the first dim.")
        return monvar
    return monvar

#%% ~ Significance Testing
## ND version (incomplete)
# def calc_dof(ts,dim=0):
#     """
#     Calculate effective degrees of freedom for autocorrelated timeseries.
#     Assumes time is first dim, but can specify otherwise.
    
#     ts :: ARRAY [time x otherdims] 1-D or 2-D Array
    
#     """
#     # Create Lagged Timeseries
#     n_tot = ts.shape[dim]
#     ts0   = np.take(ts,np.arange(0,n_tot-1),dim)
#     ts1   = np.take(ts,np.arange(1,n_tot),dim)
    
#     # Calculate Lag 1 Autocorrelation
#     #tscorr = lambda ts0,ts1 : np.corrcoef(ts0[0,:],ts1[0,:])[0,1]
#     #corrs  = np.apply_along_axis(tscorr,0)
    
#     #n_eff = 
#     return dof

def calc_dof(ts,ts1=None,calc_r1=True,ntotal=None,verbose=True,r1_in=None,r1_in_2=None):
    """
    Calculate effective degrees of freedom for autocorrelated timeseries.
    Assumes time is first dim, but can specify otherwise. Based on Eq. 31
    from Bretherton et al. 1998 (originally Bartlett 1935):
        
        N_eff = N * (1-r1*r2) / (1+r1*r2) 
        
    Inputs:
        ts          :: ARRAY [time] 1-D or 2-D Array, or lag 1 autocorrelation (r1) if calc_r1=False
        ts1         :: ARRAY [time] Another timeseries to correlate, or r1 if calc_r1=False
        calc_r1     :: BOOL    - Set to False if ts and ts1 are precalculated r1s
        ntotal      :: NUMERIC - Number of samples, must be given if calc_r1 is false (full DOF)
    Output:
        dof         :: Int Effective Degrees of Freedom
        
    """
    if calc_r1:
        if ntotal is None:
            n_tot = len(ts)
        else:
            n_tot = ntotal
    else:
        n_tot = len(ts)
    if verbose:
        print("Setting base DOF to %s" % str(n_tot))
    
    # Compute R1 for first timeseries
    if calc_r1:
        ts_base         = ts[:-1]
        ts_lag          = ts[1:]
        r1              = np.corrcoef(ts_base,ts_lag)[0,1]
    else:
        r1 = ts
    
    if r1_in is not None:
        print("Using provided r1 for timeseries 1")
        r1 = r1_in
    
    if np.any(r1<0):
        print("Warning, r1 is less than zero. Taking abs value!")
        r1 = np.abs(r1)
    
    if ts1 is None: # Square R1
        rho_in = r1**2
        
    else: # Compute R2 and compute product
        
        if calc_r1:
            ts1_base    = ts1[:-1]
            ts1_lag     = ts1[1:]
            r2          = np.corrcoef(ts1_base,ts1_lag)[0,1]
        else:
            r2          = ts1
            
        if r1_in_2 is not None:
            print("Using provided r1 for timeseries 2")
            r2 = r1_in_2
            
        if np.any(r2<0):
            print("Warning, r2 is less than zero. Taking abs value!")
            r2 = np.abs(r2)
    
        rho_in      = r1*r2
    
    # Compute DOF
    dof   = n_tot * (1-rho_in) / (1+rho_in)
    
    return dof

def ttest_rho(p,tails,dof,return_str=False):
    """
    Perform T-Test, given pearsonr, p (0.05), and tails (1 or 2), and degrees
    of freedom. The latter dof can be N-D
    
    Edit 12/01/2021, removed rho since it is not used
    
    """
    # Compute p-value based on tails
    ptilde = p/tails
    
    # Get threshold critical value
    if type(dof) is np.ndarray: # Loop for each point
        oldshape = dof.shape
        dof = dof.reshape(np.prod(oldshape))
        critval = np.zeros(dof.shape)
        for i in range(len(dof)): 
            critval[i] = stats.t.ppf(1-ptilde,dof[i])
        critval = critval.reshape(oldshape)
    else:
        critval = stats.t.ppf(1-ptilde,dof)
    
    # Get critical correlation threshold
    if type(dof) is np.ndarray:
        dof = dof.reshape(oldshape)
    corrthres = np.sqrt(1/ ((dof/np.power(critval,2))+1))
    if return_str:
        ttest_str = "p%03i_tails%i" % (p*100,tails)
        return corrthres,ttest_str
    return corrthres

def calc_stderr(x,dim,p=0.05,tails=2):
    """
    Parameters
    ----------
    x   : ARRAY of values
    dim : INT, dimension along which to compute the standard error

    Returns
    -------
    SE : FLOAT:  
    
    Calculates the standard error assuming a normal distribution + independent
    samples,
                        SE  = sigma / sqrt(n) * FAC, 
    where:
        sigma is the sample standard deviation
        n is the sample size
        FAC is the factor associated with confidence interval (ex. 1.96 = 95%)
    """
    sigma = np.nanstd(x,dim)            # Get sample standard dev
    n     = x.shape[dim]                # Number of samples      
    conf  = 1 - p/tails                 # Get confidence interval
    FAC   = stats.norm.ppf(conf)        # Get factor
    SE    = sigma / np.sqrt(n) * FAC    # Compute Standard Error
    return SE

def regress_ttest(in_var,in_ts,dof=None,p=0.05,tails=2,verbose=True):
    """
    Given a timeseries (in_ts) and variable (in_var), compute regression
    coefficients and perform t-test to get significance
    Note: only tested for single value DOF, need to check for map of dofs...
    h0: regression coeffs are significantly different from zero
    
    Inputs:
    -------
    invar (ARRAY: [Lon x Lat x Time])   : Input pattern to regress
    in_ts (ARRAY: [time])               : Timeseries to regress to
    dof   (NUMERIC)                     : Degrees of Freedom to use. Defaults to nt-2
    p     (NUMERIC)                     : p-value for significance testing; Default: 0.05
    tails (INT)                         : # of Tails for t-test (1 or 2); Default: 2
    
    Outputs: (all (ARRAY: [Lon x Lat] ezcept t_critval)
    --------
    regression_coeff : Map of Regression Coefficients
    intercept        : Map of Intercepts
    SSE              : Squared Sum of Errors
    se               : Residual Standard Error
    t_statistic      : T-statistic at each point
    t_critval        : Critical T-value
    sigmask          : Mask where t_statistic > t_critval
    
    """
    
    # Step (1), get needed dimensions
    nt          = in_ts.shape[0]
    nlon,nlat,_ = in_var.shape # Assume [lon x lat x time]
    invar_rs    = in_var.reshape(nlon*nlat,nt)
    
    # Step (2), Remove NaNs
    nandict     = find_nan(invar_rs,1,return_dict=True,verbose=verbose) # Sum along time in 1
    invar_rs    = nandict['cleaned_data']
    
    # Define function to replace NaN
    def replace(x):
        outvar = np.zeros((nlon*nlat))
        outvar[nandict['ok_indices']] = x
        return outvar.reshape(nlon,nlat)
    
    # A1. Compute the Slopes
    m,b = regress_2d(in_ts,invar_rs) # [1 x pts]
    
    # A2. Calculate SSE and residual standard error
    # https://www.geo.fu-berlin.de/en/v/soga-r/Basics-of-statistics/Hypothesis-Tests/Inferential-Methods-in-Regression-and-Correlation/Inferences-About-the-Slope/index.html
    yhat    = in_ts[None,:] * m.T  + b.T # Re-make the model
    epsilon = invar_rs - yhat # Residual
    SSE     = (epsilon**2).sum(1) # Errors are generally large along NAC
    if dof is None:
        if verbose:
            print("Using DOF len(time) - 2...")
        dof     = nt-2 # Note you can set DOF to be different here. I think 2 is just 2 parameters for linear regr
    se      = np.sqrt(SSE/ (dof)) # Residual Standard Error. 
    
    # A3. Compute the t-statistic
    rss_x = np.sqrt( np.sum( (in_ts - in_ts.mean()) **2))# Root Sum Square of x
    denom = se / rss_x
    tstat = m.squeeze() / denom
    
    # A4. Get Critical T
    ptilde  = p/tails
    critval = stats.t.ppf(1-ptilde,dof)
    if tails == 2:
        critval_lower = stats.t.ppf(ptilde,dof)
    
    # Make significance Mask
    if tails == 2:
        sigmask = (tstat > critval) | (tstat < critval_lower)
    else:
        sigmask = tstat > critval
    
    
    sigmask = replace(sigmask)
    
    # Return all values
    outdict = {}
    outdict["regression_coeff"] = replace(m.squeeze())
    outdict["intercept"] = replace(b.squeeze())
    outdict["SSE"] = replace(SSE)
    outdict["se"] = replace(se)
    outdict["t_statistic"] = replace(tstat)
    outdict["t_critval"] = critval
    outdict["sigmask"] = sigmask
    if tails == 2:
        outdict['t_critval_lower'] = critval_lower
    
    return outdict


def calc_pval_rho(rho,dof):
    # Compute p-value (two-tailed) given pearson R correaltion and Degrees of freedom
    # From https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Mostly_Harmless_Statistics_(Webb)/12%3A_Correlation_and_Regression/12.01%3A_Correlation/12.1.02%3A_Hypothesis_Test_for_a_Correlation
    tstat = rho * np.sqrt(dof/(1-rho**2))
    pval  = sp.stats.t.sf(np.abs(tstat), dof)*2  # two-sided pvalue = Prob(abs(t)>tt)
    return pval


def mcsampler(ts_full,sample_len,mciter,preserve_month=True,scramble_year=False,target_timeseries=None):
    # Taken from ensobase.utils on 2025.10.30
    # Given a monthly timeseries [time] and sample length (int), take [mciter] # of random samples.
    # if preserve_month = True, preserve the 12-month sequence as a chunk
    # if scramble_year = True, randomize years that you are selecting from (do not preserve year order)
    # if target_timeseries is not None: also select random samples from list of timeseries (must be same length as ts_full)
    
    # Function Start
    ntime_full        = len(ts_full)
    
    # 1 -- month agnostic (subsample sample length, who cares when)
    if not preserve_month:
        
        print("Month with not be preserved.")
        istarts    = np.arange(ntime_full-sample_len)
        
        sample_ids = []
        samples    = []
        for mc in range(mciter):
            # ts_full[istarts[-1]:(istarts[-1]+sample_len)] Test last possible 
            iistart = np.random.choice(istarts)
            idsel   = np.arange(iistart,iistart+sample_len) 
            msample = ts_full[idsel]
            
            
            sample_ids.append(idsel)
            samples.append(msample) # [iter][sample]
            
        samples = np.array(samples) # [iter x sample]
        # Returns 
            
    elif preserve_month:
        # 2 -- month aware (must select starting points of January + maintain the chunk, preserving the month + year to year autocorrelation)
        if not scramble_year:
            
            # Only start on the year  (to preserve month sequence)
            istarts    = np.arange(0,ntime_full-sample_len,12)
            
            # -------------------- Same as Above
            sample_ids = []
            samples    = []
            for mc in range(mciter):
                # ts_full[istarts[-1]:(istarts[-1]+sample_len)] Test last possible 
                iistart = np.random.choice(istarts)
                idsel   = np.arange(iistart,iistart+sample_len) 
                msample = ts_full[idsel]
                
                sample_ids.append(idsel)
                samples.append(msample) # [var][iter][sample]
            samples = np.array(samples) # [var x iter x sample]
            # -------------------- 
            
        # 3 -- month aware, year scramble (randomly select the year of each month, but preserve each month)
        elif scramble_year: # Scrample Year and Month
            
            # Reshape to the year and month
            nyr_full        = int(ntime_full/12)
            ts_yrmon        = ts_full.reshape(nyr_full,12)
            ids_ori         = np.arange(ntime_full)
            ids_ori_yrmon   = ids_ori.reshape(ts_yrmon.shape)
            
            nyr_sample      = int(sample_len/12)
            sample_ids      = []
            samples         = []
            for mc in range(mciter): # For each loop
                
                # Get start years
                startyears = np.random.choice(np.arange(nyr_full),nyr_sample)
                # Select random years equal to the sample length and combine
                idsel      = ids_ori_yrmon[startyears,:].flatten() 
                # ------
                msample    = ts_full[idsel]
                sample_ids.append(idsel)
                samples.append(msample) # [var][iter][sample]
            samples = np.array(samples) # [var x iter x sample]
            # -----
    
    outdict = dict(sample_ids = sample_ids, samples=samples)    
    if target_timeseries is not None:
        
        sampled_timeseries = []
        for ts in target_timeseries:
            if len(ts) != len(ts_full):
                print("Warning... timeseries do not have the same length")
            randsamp = [ts[sample_ids[mc]] for mc in range(mciter)]
            randsamp = np.array(randsamp)
            sampled_timeseries.append(randsamp) # [var][iter x time]
        outdict['other_sampled_timeseries'] = sampled_timeseries
    
    return outdict


#%% ~ Other

def covariance2d(A,B,dim,normalize=False):
    """
    Calculate Covariancefor two 2-D Arrays
    along the specified dimension. Input arrays are anomalized.
        
    Inputs
    -------
    1) A : ARRAY
        First variable, 2D
    2) B : ARRAY
        Second variable, 2D (same axis arrangement as A)
    3) dim : INT
        Dimension to compute correlation along
        
    Outputs
    -------    
    1) cov : ARRAY
        Covariance
    
    """
    # Find Anomaly
    Aanom = A - np.mean(A,dim)
    Banom = B - np.mean(B,dim)
    
    # Elementwise product of A and B
    AB = Aanom * Banom
    
    # Calculate covariance
    cov = np.sum(AB,dim)/A.shape[dim]
    
    # Normalize to return correlation if desired
    if normalize:
        cov = cov / (np.nanstd(A,dim) * np.nanstd(B,dim))
    return cov

def make_ar1(r1,sigma,simlen,t0=0,savenoise=False,usenoise=None):
    """
    Create AR1 timeseries given the lag 1 corr-coef [r1],
    the amplitude of noise (sigma), and simluation length.
    
    Adapted from slutil.return_ar1_model() on 2022.04.12

    Parameters
    ----------
    r1        [FLOAT] : Lag 1 correlation coefficient
    sigma     [FLOAT] : Noise amplitude (will be squared)
    simlen    [INT]   : Simulation Length
    t0        [FLOAT] : Starting value (optional, default is 0.)
    savenoise [BOOL]  : Output noise timeseries as well
    usenoise  [ARRAY] : Use provided noise timeseries

    Returns
    -------
    rednoisemodel : [ARRAY: (simlen,)] : AR1 model
    noisets       : [ARRAY: (simlen,)] : Used noise timeseries
    """
    # Create Noise
    if usenoise is None:
        noisets       = np.random.normal(0,1,simlen)
        noisets       *= (sigma**2) # Scale by sigma^2
    else:
        noisets = usenoise
    # Integrate Model
    rednoisemodel = np.zeros(simlen) * t0 # Multiple by initial value
    for t in range(1,simlen):
        rednoisemodel[t] = r1 * rednoisemodel[t-1] + noisets[t]
    if savenoise:
        return rednoisemodel,noisets
    return rednoisemodel

def calc_r1_sigma(ts):
    """
    Given a timeseries [ts]:
        1. Estimate the lag-1 autocorrelation (r1)
        2. Recover the noise amplitude assuming an AR(1) model + stationary ts
            var(ts) = r1^2 var(ts) + sigma^2
            
            sigma^2 = var(ts) (1-r1^2)
            
            See 12.S992 Class Notes
            NOTE: not sure about the noise amplitude... should check some refs
    """
    r1      = np.corrcoef(ts[:(-1)],ts[1:])[0,1]
    sigma   = np.sqrt( np.var(ts) * (1-r1**2) )
    #sigma   = (1-r1**2)
    return r1,sigma

def montecarlo_ar1(ts1,ts2,mciter,infuncs):
    """
    
    Perform [mciter] MonteCarlo simulations for a set of timeseries [ts1,ts2] 
    by generating AR(1) models and using specified operations/functions [infuncs]
    
    Parameters
    ----------
    ts1,ts2 : ARRAY [time]
        Target timeseries for the analysis. They are assumed to have the same time
        dimension.
    mciter : INT
        Number of iterations.
    infuncs : LIST of funcs
        List of functions, each that take ts1 and ts2 as input and output something
        (using lambda ts1,ts2 = function(ts1,ts2,[other_args...])).
        Example: [func1, func2, ..., funcN]
    
    Returns
    -------
    output : LIST of outputs
        List of [mciter] outputs in the order of each function.
        Example: [output1[mciter],output2[mciter],..., outputN[mciter]]
        
    See analyze_amoc_index_local.py coherence squared calculations for an example
        
    """
    
    r1,sigma1 = calc_r1_sigma(ts1)
    r2,sigma2 = calc_r1_sigma(ts2)
    ntime     = len(ts1)
    
    output = [[] for ff in range(len(infuncs))]
    
    for mc in tqdm.tqdm(range(mciter)):
        
        ar1 = make_ar1(r1,sigma1,ntime)
        ar2 = make_ar1(r2,sigma2,ntime)
        
        # Apply Functions
        for f,ifunc in enumerate(infuncs):
            mcout = ifunc(ar1,ar2)
            output[f].append(mcout)
    return output


def patterncorr(map1,map2,verbose=True):
    # From Taylor 2001,Eqn. 1, Ignore Area Weights
    # Calculate pattern correation between two 2d variables (lat x lon)
    
    
    # Get Non NaN values, Flatten, Array Size
    nan1 = ~np.isnan(map1) 
    nan2 = ~np.isnan(map2)
    if np.any(nan1 != nan2):
        if verbose:
            print("Warning, both maps have different NaN points." + "\n" +
                  "Calculation will only include points that are non-NaN in both.")
        nanboth = nan1*nan2
        nan1 = nanboth
        nan2 = nanboth
    
    map1ok = map1.copy()
    map1ok = map1ok[nan1].flatten()
    map2ok = map2.copy()
    map2ok = map2ok[nan2].flatten()
    N      = len(map1ok)
    
    # Anomalize
    map1a = map1ok - map1ok.mean()
    map2a = map2ok - map2ok.mean()
    std1  = np.std(map1ok)
    std2  = np.std(map2ok)
    
    # calculate
    R = 1/N*np.sum(map1a*map2a)/(std1*std2)
    return R

"""
------------------------------
|||  Spectral Analysis  ||| ****************************************************
------------------------------
"""
#%% ~ Spectral Analysis and Filtering
def lp_butter(varmon,cutofftime,order,btype='lowpass'):
    """
    Design and apply a low-pass filter (butterworth)

    Parameters
    ----------
    varmon : 
        Input variable to filter (monthly resolution)
    cutofftime : INT
        Cutoff value in months
    order : INT
        Order of the butterworth filter
    btype : STR
        Type of filter {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}

    Returns
    -------
    varfilt : ARRAY [time,lat,lon]
        Filtered variable
    
    """
    # Input variable is assumed to be monthy with the following dimensions:
    flag1d=False
    if len(varmon.shape) > 1:
        nmon,nlat,nlon = varmon.shape
    else:
        flag1d = True
        nmon = varmon.shape[0]
    
    # Design Butterworth Lowpass Filter
    filtfreq = nmon/cutofftime
    nyquist  = nmon/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype=btype)
    
    # Reshape input
    if flag1d is False: # For 3d inputs, loop thru each point
        varmon = varmon.reshape(nmon,nlat*nlon)
        # Loop
        varfilt = np.zeros((nmon,nlat*nlon)) * np.nan
        for i in range(nlon*nlat):
            varfilt[:,i] = filtfilt(b,a,varmon[:,i])
        varfilt=varfilt.reshape(nmon,nlat,nlon)
    else: # 1d input
        varfilt = filtfilt(b,a,varmon)
    return varfilt
    

def calc_specvar(freq,spec,thresval,dtthres,droplast=True
                 ,lowerthres=0,return_thresids=False,trapz=True):
    """
    Calculate variance of spectra BELOW a certain threshold
    
    Inputs:
        freq     [ARRAY]   : frequencies (1/sec)
        spec     [ARRAY]   : spectra (Power/cps) [otherdims ..., freq]
        thresval [FLOAT]   : Threshold frequency (in units of dtthres)
        dtthres  [FLOAT]   : Units of thresval (in seconds)
        droplast [BOOL]    : True,start from lowest freq (left riemann sum)
        lowerthres [FLOAT] : Low-freq limit (in units of dtthres) Default=0
        return_thresids [BOOL] : Set to True to just return the threshold indices
        trapz [BOOL] : Just use the trapezoidal integration method
    """
    # Get indices of frequencies less than the threshold
    thresids = (freq*dtthres >= lowerthres) * (freq*dtthres <= thresval)
    if return_thresids:
        return thresids
        
    # Limit to values
    specthres = spec[...,thresids]
    freqthres = freq[thresids]
    
    if trapz:
        specsum = np.trapz(specthres,x=freqthres,axis=-1)
    else:
        # Compute the variance (specval*df)
        if droplast:
            specval    = specthres[...,:-1]#np.abs((specthres[1:] - specthres[:-1]))/dtthres
        else:
            specval    = specthres[...,1:]
        df       = ((freqthres[1:] - freqthres[:-1]).mean(0))
        specsum  = np.sum((specval*df),-1)
    return specsum

def point_spectra(ts, nsmooth=1, opt=1, dt=None, clvl=[.95], pct=0.1):
    # Compute the power spectra to a single timeseries
    if dt is None:  # Divides to return output in 1/sec
        dt = 3600*24*30
    sps = ybx.yo_spec(ts, opt, nsmooth, pct, debug=False, verbose=False)

    P, freq, dof, r1 = sps
    coords = dict(freq=freq/dt)
    da_out = xr.DataArray(P*dt, coords=coords, dims=coords, name="spectra")
    return da_out


def get_freqdim(ts, dt=None, opt=1, nsmooth=1, pct=0.10, verbose=False, debug=False):
    # Get the frequency dimension from a spectra calculation
    if dt is None:
        dt = 3600*24*30
    sps = ybx.yo_spec(ts, opt, nsmooth, pct, debug=False, verbose=verbose)
    if sps is None:
        return None
    return sps[1]/dt


def calc_confspec(alpha,nu):
    """
    Based on code written by Tom Farrar for 12.805 @ MIT (see the func. confid).
    Copied from the 12.805 tbx.py on 2025.04.22
    
    Compute the upper and lower confidence limits for a chi-square variate.
    
    
    Parameters
    ----------
    alpha : numeric
        Significance value (For example, alpha=0.05 gives 95% confidence interval.)
    nu : numeric
        Number of degrees of freedom

    Returns
    -------
    lower: lower bound of confidence interval
    upper: upper bound of confidence interval
    
    """
    # requires:
    # from scipy import stats
    upperv=stats.chi2.isf(1-alpha/2,nu)
    lowerv=stats.chi2.isf(alpha/2,nu)
    lower=nu / lowerv
    upper=nu / upperv
    
    return (lower,upper)

def plot_conflog(loc,bnds,ax=None,color='k',cflabel=None):
    """
    Plot confidence intervals for spectral density on a log scale
    Copied from the 12.805 tbx.py on 2025.04.22
    
    params
    ------
        1) loc : array
            [x-coordinate,y-coordinate] for confidence interval point
        2) bnds : array
            [lower,upper] bounds of confidence interval (nu/chi_alpha)
        Optional...
        3) ax : matplotlib axes to plot on
        4) color : string - color of confidence interval
        5) cflabel : string - label for the confidence interval
    
    outputs
        1) ax : matplotlib axes with confidence interval plotted
        2) bar : matplotlib object with error bars
    """

    if ax is None:
        ax = plt.gca()
        
    if cflabel is None:
        cflabel = "95% Confidence"
    
    confx,confy = loc
    lower,upper=bnds
    errbounds = confy*np.array([[lower],[upper]])
    #print(errbounds)
    bars = ax.errorbar([confx],[confy],
                       yerr=errbounds,
                       marker='o',capsize=3,
                       color=color,label=cflabel)
    return ax,bars


"""
-------------------------------
|||  Indexing and Querying  ||| ****************************************************
-------------------------------
"""

#%% ~ Indexing and Querying
def find_latlon(lonf,latf,lon,lat,verbose=True):
    """
    Find lat and lon indices
    """
    if((np.any(np.where(lon>180)) & (lonf < 0)) or (np.any(np.where(lon<0)) & (lonf > 180))):
        print("Potential mis-match detected between lonf and longitude coordinates")
    
    klon = np.abs(lon - lonf).argmin()
    klat = np.abs(lat - latf).argmin()
    
    if verbose:
        msg1 = "Closest lon to %.2f was %.2f" % (lonf,lon[klon])
        msg2 = "Closest lat to %.2f was %.2f" % (latf,lat[klat])
        print(msg1)
        print(msg2)
    return klon,klat

def find_tlatlon(ds,lonf,latf,verbose=True,return_index=False):

    # Get minimum index of flattened array
    kmin      = np.argmin( (np.abs(ds.TLONG-lonf) + np.abs(ds.TLAT-latf)).values)
    klat,klon = np.unravel_index(kmin,ds.TLAT.shape)
    
    # Print found coordinates
    if verbose:
        foundlat = ds.TLAT.isel(nlat=klat,nlon=klon).values
        foundlon = ds.TLONG.isel(nlat=klat,nlon=klon).values
        print("Closest lon to %.2f was %.2f" % (lonf,foundlon))
        print("Closest lat to %.2f was %.2f" % (latf,foundlat))
    if return_index:
        return klon,klat
    return ds.isel(nlon=klon,nlat=klat)

def find_nan(data,dim,val=None,return_dict=False,verbose=True):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim].
    
    Inputs:
        1) data        : 2d array, which will be summed along last dimension
        2) dim         : dimension to sum along. 0 or 1.
        3) val         : value to search for (default is NaN)
        4) return_dict : Set to True to return dictionary with clearer arguments...
    Outputs:
        1) okdata : data with nan points removed
        2) knan   : boolean array with indices of nan points
        3) okpts  : indices for non-nan points
    """
    
    # Sum along select dimension
    if len(data.shape) > 1:
        datasum = np.sum(data,axis=dim)
    else:
        datasum = data.copy()
    
    # Find non nan pts
    if val is None:
        knan  = np.isnan(datasum)
    else:
        knan  = (datasum == val)
    okpts = np.invert(knan)
    
    if len(data.shape) > 1:
        if dim == 0:
            okdata = data[:,okpts]
            clean_dim = 1
        elif dim == 1:    
            okdata = data[okpts,:]
            clean_dim = 0
    else:
        okdata = data[okpts]
    if verbose:
        print("Found %i NaN Points along axis %i." % (data.shape[clean_dim] - okdata.shape[clean_dim],clean_dim))
    if return_dict: # Return dictionary with clearer arguments
        nandict = {"cleaned_data" : okdata,
                   "nan_indices"  : knan,
                   "ok_indices"   : okpts,
                   }
        return nandict
    return okdata,knan,okpts


def selpt_ds(ds,lonf,latf,lonname='lon',latname='lat'):
    return ds.sel({lonname:lonf,latname:latf},method='nearest')


def remap_nan(lon,lat,okdata,okpts,lonfirst=True):
    """
    Remaps [okdata] with combined spatial dimension 
    into original [lon x lat] dimensions

    Parameters
    ----------
    lon : ARRAY [lon,]
        Lon values
    lat : ARRAY [lat,]
        Lat values
    okdata : ARRAY [space, otherdims]
        Values to remap
    okpts : ARRAY [space]
        Indices where the original values belonged
    lonfirst : TYPE, BOOL
        Unfold to Lon x Lat (as opposed to Lat x Lon). The default is True.

    Returns
    -------
    None.
    """
    
    nlon      = len(lon)
    nlat      = len(lat)
    otherdims = okdata.shape[1:]
    newshape  = (nlon*nlat,) +  otherdims
    if lonfirst:
        unfold    = (nlon,nlat,) +  otherdims
    else:
        unfold    = (nlat,nlon,) +  otherdims
    
    outvar = np.zeros(newshape)*np.nan
    outvar[okpts,...] = okdata
    outvar = outvar.reshape(unfold)
    return outvar

def sel_region(var,lon,lat,bbox,reg_avg=0,reg_sum=0,warn=1,autoreshape=False,returnidx=False,awgt=None):
    """
    
    Select Region
    
    Inputs
        1) var: ARRAY, variable with dimensions [lon x lat x otherdims]
        2) lon: ARRAY, Longitude values
        3) lat: ARRAY, Latitude values
        4) bbox: ARRAY, bounding coordinates [lonW lonE latS latN]
        5) reg_avg: BOOL, set to 1 to return regional average
        6) reg_sum: BOOL, set to 1 to return regional sum
        7) warn: BOOL, set to 1 to print warning text for region selection
        8) awgt: INT, type of area weighting to apply (default is None, 1=cos(lat),2=cos^2(lat))
    Outputs:
        1) varr: ARRAY: Output variable, cut to region
        2+3), lonr, latr: ARRAYs, new cut lat/lon
    
    Assume longitude is always searching eastward...
    Assume var is of the form [lon x lat x otherdims]
    bbox is [lonW lonE latS latN]
    
    
    """    
    # Reshape to combine dimensions
    dimflag = False 
    if autoreshape:
        var,vshape,dimflag=combine_dims(var,2,debug=True)
    
    # Find indices
    klat = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
    if bbox[0] < bbox[1]:
        klon = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
    elif bbox[0] > bbox[1]:
        if warn == 1:
            print("Warning, crossing the prime meridian!")
        klon = np.where((lon <= bbox[1]) | (lon >= bbox[0]))[0]
    
    if returnidx:
        return klon,klat
    
    lonr = lon[klon]
    latr = lat[klat]
    
    #print("Bounds from %.2f to %.2f Latitude and %.2f to %.2f Longitude" % (latr[0],latr[-1],lonr[0],lonr[-1]))
    
    # Index variable
    varr = var[klon[:,None],klat[None,:],...]
    
    if reg_avg==1:
        if awgt is not None:
            varr = area_avg(varr,bbox,lonr,latr,awgt)
        else:
            varr = np.nanmean(varr,(0,1))
        return varr
    elif reg_sum == 1:
        varr = np.nansum(varr,(0,1))
        return varr
    
    # Reshape variable automatically
    if dimflag:
        newshape = np.hstack([[len(lonr),len(latr)],vshape[2:]])
        varr     = varr.reshape(newshape)
    
    return varr,lonr,latr

def sel_region_cv(tlon,tlat,invar,bbox,debug=False,return_mask=False):
    """
    Select region for a curvilinear/2D points. Currently tested with
    CESM1 POP grid output
    
    Inputs
    ------
    1) tlon  : ARRAY [lat x lon]
        Longitude values at each point (*assumed degrees East,  0-360)
    2) tlat  : ARRAY [lat x lon]
        Latitude values at each point
    3) invar : ARRAY [lat x lon x otherdims]
        Input variable to select from
    4) bbox  : LIST [lonW,lonE,latS,latN]
        Bounding coordinates. Can be degrees E/W for longitude
    5) debug : BOOL (optional)
        Set to True to plot the mask selection
    6) return_mask : BOOL (optional)
        Set to True to return BOOL mask of selection
    
    Outputs
    -------
    1) sellon : ARRAY [npts]
        Longitude values of points within region
    2) sellat : ARRAY [npts]
        Latitude values of points within region
    3) selvar : ARRAY [npts]
        Data values of points within region
    4) selmask : ARRAY [lat x lon]
        Boolean mask, True for points within region
    
    """
    
    # Check crossings
    for i in range(2):
        if bbox[i] < 0:
            bbox[i] += 360 # Switch to degrees east
    if bbox[0] > bbox[1]:
        cross_pm = True
    else:
        cross_pm = False

    # Select longitude
    if cross_pm:
        # lonW --> 360, 0 --> lowE
        masklon = ( (tlon >= bbox[0]) * (tlon <= 360)) + (tlon <= bbox[1])
    else:
        masklon = ((tlon >= bbox[0]) * (tlon <= bbox[1]))
        
    masklat = ((tlat >= bbox[2]) * (tlat <= bbox[3]))
    masksel = masklon * masklat

    # Plot selection
    if debug:
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
        ax.coastlines()
        ax.scatter(tlon,tlat,0.02,alpha=0.2,color="k")
        ax.scatter(tlon[masklon],tlat[masklon],0.02,alpha=0.2,color="r")
        ax.scatter(tlon[masklat],tlat[masklat],0.02,alpha=0.2,color="b")
        ax.scatter(tlon[masksel],tlat[masksel],0.02,alpha=0.8,color="y",marker="x")
        ax.set_title("Selection Masks \n Lons (Blue) | Lats (Red) | Region (Yellow)")

    # Return selected variables
    selvar = invar[masksel,...]
    sellon = tlon[masksel]
    sellat = tlat[masksel]
    if return_mask:
        return sellon,sellat,selvar,masksel
    return sellon,sellat,selvar


def sel_region_xr(ds,bbox):
    """
    Selects region given bbox = [West Bnd, East Bnd, South Bnd, North Bnd]
    
    Parameters
    ----------
    ds : xr.DataArray or Dataset
        Assumes "lat" and "lon" variables are [present]
    bbox : LIST
        Boundaries[West Bnd, East Bnd, South Bnd, North Bnd]
        
    Returns
    -------
        Subsetted datasetor dataarray
    """
    return ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))


def sel_region_xr_cv(ds2,bbox,debug=False):
    # Select region with curvilinear coordinates TLONG and TLAT
    # Copied from preprocess_by_level (but removed the vname requirement)
    # Note, assumes tlon is degrees east and converts if not
    # Get mesh
    tlat = ds2.TLAT.values
    tlon = ds2.TLONG.values
    
    # Adjust to degrees east
    if np.any(tlon < 0):
        print("Converting longitude to degrees east")
        tlon = np.where(tlon<0,tlon+360,tlon)
    
    # Make Bool Mask
    latmask = (tlat >= bbox[2]) * (tlat <= bbox[3])
    
    # Three Cases
    # Case 1. Both are degrees west
    # Case 2. Crossing prime meridian (0,360)
    # Case 3. Crossing international date line (180,-180)
    # Case 4. Both are degrees east
    if np.any(np.array(bbox)[:2] < 0):
        print("Degrees West Detected")
        
        if np.all(np.array(bbox[:2]) < 0): # Case 1 Both are degrees west
            print("Both are degrees west")
            lonmask = (tlon >= bbox[0]+360) * (tlon <= bbox[1]+360)
            
        elif (bbox[0] < 0) and (bbox[1] >= 0): # Case 2 (crossing prime meridian)
            print("Crossing Prime Meridian")
            lonmaskE = (tlon >= bbox[0]+360) * (tlon <= 360) # [lonW to 360]
            if bbox[1] == 0:
                lonmaskW = lonmaskE
            else:
                lonmaskW = (tlon >= 0)           * (tlon <= bbox[1])       # [0 to lonE]
            
            lonmask = lonmaskE | lonmaskW
        elif (bbox[0] > 0) and (bbox[1] < 0): # Case 3 (crossing dateline)
            print("Crossing Dateline")
            lonmaskE = (tlon >= bbox[0]) * (tlon <= 180) # [lonW to 180]
            lonmaskW = (tlon >= 180)     * (tlon <= bbox[1]+360) # [lonW to 180]
            lonmask = lonmaskE * lonmaskW
    else:
        print("Everything is degrees east")
        lonmask = (tlon >= bbox[0]) * (tlon <= bbox[1])
    
    regmask = lonmask*latmask

    # Select the box
    if debug:
        plt.pcolormesh(lonmask*latmask),plt.colorbar(),plt.show()
    
    # Make a mask
    #ds2 = ds2[vname]#.isel(z_t=1)
    
    ds2.coords['mask'] = (('nlat', 'nlon'), regmask)
    
    st = time.time()
    ds2 = ds2.where(ds2.mask,drop=True)
    print("Loaded in %.2fs" % (time.time()-st))
    return ds2



def get_bbox(ds):
    # Get bounding box of a dataset from "lon" and "lat" dimensions
    bbox = [ds.lon.values[0],
            ds.lon.values[-1],
            ds.lat.values[0],
            ds.lat.values[-1]]
    return bbox

def resize_ds(ds_list):
    # Given a list of datasets, (lon, lat, etc)
    # Resize all of them to the smallest bounding box
    # Note this was made to work with degrees west, have not handeled crrossing dateline
    bboxes  = np.array([get_bbox(ds) for ds in ds_list]) # [ds.bound]
    
    bbxsmall = np.zeros(4)
    bbxsmall[0] = np.max(bboxes[:,0]) # Easternmost Westbound
    bbxsmall[1] = np.min(bboxes[:,1]) # Westernmost Eastbound
    bbxsmall[2] = np.max(bboxes[:,2]) # Northerhmost Southbound
    bbxsmall[3] = np.min(bboxes[:,3]) # Southernmost Northbound
    
    ds_resize = [sel_region_xr(ds,bbxsmall) for ds in ds_list]
    return ds_resize

def get_posneg(varr,idx,return_id=False):
    """
    Get positive and negative years of an varianble
    based on some climate index
    
    Inputs:
        1) varr : ARRAY[lon x lat x time] - Target Variable
        2) idx  : ARRAY[time] - Target Index
        3) return_id : BOOL - Set to true to just return indices
    Output:
        1) varrp : ARRAY[lon x lat x time] - Positive years
        2) varrn : ARRAY[lon x lat x time] - Negative years
        3) varrz : ARRAY[lon x lat x time] - Zero years
    
        if return_id:
            1) kp : ARRAY[time] - Positive Indices
            2) kn : ARRAY[time] - Negative Indices
            3) Kz : ARRAY[time] - Zero Indices
    """
    
    kp = idx > 0
    kn = idx < 0
    kz = idx == 0
    
    if return_id:
        return kp,kn,kz
    
    varrp = varr[:,:,kp]
    varrn = varr[:,:,kn]
    varrz = varr[:,:,kz]
    
    return varrp,varrn,varrz

def get_posneg_sigma(varr,idxin,sigma=1,normalize=True,return_id=False):
    """
    Given index, normalize (optional) and take the values
    [sigma] standard deviations above and below the mean
    
    Inputs:
        1) varr : ARRAY[lon x lat x time] - Target Variable
        2) idx  : ARRAY[time] - Target Index
        3) sigma : NUMERIC - n standard deviation to use as threshold
        4) normalize : BOOL - Option to  
        3) return_id : BOOL - Set to true to just return indices
    Output:
        1) varrp : ARRAY[lon x lat x time] - Positive years
        2) varrn : ARRAY[lon x lat x time] - Negative years
        3) varrz : ARRAY[lon x lat x time] - Zero years
    
        if return_id:
            1) kp : ARRAY[time] - Positive Indices
            2) kn : ARRAY[time] - Negative Indices
            3) Kz : ARRAY[time] - Zero Indices
    """
    idx = idxin.copy()
   
    if normalize: # Normalize the index so sigma = 1
        idx -= np.nanmean(idx) # Anomalize
        idx /= np.nanstd(idx)
    
    kp = idx > sigma # Extreme Positive
    kn = idx < -sigma # Extreme Negative
    kz = ~kp * ~kn
    
    if return_id:
        return kp,kn,kz
    
    varrp = varr[:,:,kp]
    varrn = varr[:,:,kn]
    varrz = varr[:,:,kz]
    return varrp,varrn,varrz

def get_topN(arr,N,bot=False,sort=False,absval=False):
    """
    Get the indices for the top N values of an array.
    Searches along the last dimension. Option to sort output.
    Set [bot]=True for the bottom 5 values
    
    Parameters
    ----------
    arr : TYPE
        Input array with partition/search dimension as the last axis
    N : INT
        Top or bottom N values to find
    bot : BOOL, optional
        Set to True to find bottom N values. The default is False.
    sort : BOOL, optional
        Set to True to sort output. The default is False.
    absval : BOOL, optional
        Set to True to apply abs. value before sorting. The default is False.
        
    Returns
    -------
    ids : ARRAY
        Indices of found values
    """
    
    if absval:
        arr = np.abs(arr)
    if bot is True:
        ids = np.argpartition(arr,N,axis=-1)[...,:N]
    else:
        ids = np.argpartition(arr,-N,axis=-1)[...,-N:]
         # Parition up to k, and take first k elements
    if sort:
        if bot:
            return ids[np.argsort(arr[ids])] # Least to greatest
        else:
            return ids[np.argsort(-arr[ids])] # Greatest to least
    return ids


def maxid_2d(invar):
    """
    Find indices along each dimension for a 2D matrix. Ignores NaN values.
    
    Parameters
    ----------
    invar : ARRAY [x1, x2] - 2D ARRAY/Matrix
    Returns
    -------
    idx1  : INT            - Index along first axis x1
    idx2  : INT            - Index along second axis x2
    """
    x1,x2     = invar.shape
    idmax     = np.nanargmax(invar.flatten())
    idx1,idx2 = np.unravel_index(idmax,invar.shape)
    return idx1,idx2

def sort_by_axis(sortarr,targarrs,axis=0,lon=None,lat=None):
    """
    Sort a list of arrays [targarrs] given values from [sortarr] along [axis] from smallest to largest
    Note this is currently untested...
    
    Parameters
    ----------
    sortarr : np.array
        Array containing values to sort by along [axis]
    targarrs : list of np.arrays
        List containing target arrays to sort (same axis)
    axis : INT, optional
        Axis along which to sort. The default is 0.
    lon : np.array, optional
        Longitude Values. The default is None.
    lat : np.array, optional
        Latitude values. The default is None.
        
    Returns
    -------
    sortid : np.array
        Array containing indices that would sort array from smallest to largest
    sorttarg : list of np.arrays
        Sorted list of arrays
    coords_str : list of STR ["lon,lat"] (%.2f) for corresponding points
    coords_val : list of lists [lon,lat] in float for corresponding points
    """
    
    
    sortid   = np.argsort(sortarr,axis=axis)
    sortid   = [sid for sid in sortid if ~np.isnan(sortarr[sid])]# Drop NaN values
    sorttarg = [np.take(arrs,sortid,axis=axis) for arrs in targarrs]
    if (lon is not None) and (lat is not None):
        nlon,nlat=len(lon),len(lat)
        xx,yy  = np.unravel_index(sortid,(nlon,nlat))
        coords_str = [[ "%.2f, %.2f" % (lon[xx[i]], lat[yy[i]])] for i in range(len(sortid))] # String Formatted Version
        coords_val = [[lon[xx[i]], lat[yy[i]]] for i in range(len(sortid))] # Actual values
        return sortid,sorttarg,coords_str,coords_val
    return sortid,sorttarg


def get_nearest(value,searcharr,dim=0,return_rank=False):
    # Given a [value], find the closest corresponding number in [searcharr] along dimension [dim]
    # Optionally return the full ranking of values from closest to furthest
    diff = np.abs(searcharr - value)
    if return_rank:
        return np.argsort(diff,axis=dim)
    return np.argmin(diff,axis=dim)


def indexwindow(invar,m,monwin,combinetime=False,verbose=False):
    """
    index a specific set of months/years for an odd sliding window
    given the following information (see inputs)
    
    drops the first and last years when a the dec-jan boundary
    is crossed, according to the direction of crossing
    time dimension is thus reduced by 2 overall
    
    inputs:
        1) invar [ARRAY: yr x mon x otherdims] : variable to index
        2) m [int] : index of central month in the window
        3) monwin [int]: total size of moving window of months
        4) combinetime [bool]: set to true to combine mons and years into 1 dimension
    
    output:
        1) varout [ARRAY]
            [yr x mon x otherdims] if combinetime=False
            [time x otherdims] if combinetime=True
    
    """
    
    if monwin > 1:  
        winsize = int(np.floor((monwin-1)/2))
        monid = [m-winsize,m,m+winsize]
    
    varmons = []
    msg = []
    for m in monid:

        if m < 0: # Indexing months from previous year
            
            msg.append("Prev Year")
            varmons.append(invar[:-2,m,:])
            
        elif m > 11: # Indexing months from next year
            msg.append("Next Year")
            varmons.append(invar[2:,m-12,:])
            
        else: # Interior years (drop ends)
            msg.append("Interior Year")
            varmons.append(invar[1:-1,m,:])
    if verbose:
        print("Months are %s with years %s"% (str(monid),str(msg)))       
    # Stack together and combine dims, with time in order
    varout = np.stack(varmons) # [mon x yr x otherdims]
    varout = varout.transpose(1,0,2) # [yr x mon x otherdims]
    if combinetime:
        varout = varout.reshape((varout.shape[0]*varout.shape[1],varout.shape[2])) # combine dims
    return varout
    
def match_time_month(var_in,ts_in,timename='time'):
    # Crops the start and end times for var_in and ts_in (xr.DataArrays/Datasets)
    # Note works for datetime64[ns] format in xr.DataArray
    # See ensobase/calculate_enso_response.py for working example
    
    if len(var_in[timename]) != len(ts_in[timename]): # Check if they match
        
        # Warning: Only checking Year and Date
        vstart = str(np.array((var_in[timename].data[0])))[:7]
        tstart = str(np.array((ts_in[timename].data[0])))[:7]
        
        if vstart != tstart:
            print("Start time (v1=%s,v2=%s) does not match..." % (vstart,tstart))
            if vstart > tstart:
                print("Cropping to start from %s" % vstart)
                ts_in = ts_in.sel( 
                    {timename : slice(vstart+"-01",None)}
                    )
            elif vstart < tstart:
                print("Cropping to start from %s" % tstart)
                var_in = var_in.sel(
                    {timename : slice(tstart+"-01",None)}
                    )
        
        vend = str(np.array((var_in[timename].data[-1])))[:7]
        tend = str(np.array((ts_in[timename].data[-1])))[:7]
        
        
        if vend != tend:
            
            print("End times (v1=%s,v2=%s) does not match..." % (vend,tend))
            
            if vend > tend:
                print("\nCropping to start from %s" % tend)
                var_in = var_in.sel(time=slice(None,tend+"-31"))
            elif vend < tend:
                print("\nCropping to start from %s" % vend)
                ts_in = ts_in.sel(time=slice(None,vend+"-31"))
                
        print(len(var_in.time) == len(ts_in.time))  
    return var_in,ts_in

def getfirstnan(x):
    # Find indices if first NaN in a 1-D Array
    # Designed to be used with xarray.ufunc (see xrfunc and smio/scrap/investigate_blowup_sm for reference)
    idout =  np.where(np.isnan(x))[0] #[0][0]
    if len(idout) == 0: # No NaN Found
        return len(x)
    else:
        return idout[0]

"""
-----------------------------------
|||  Interpolation & Regridding ||| ****************************************************
-----------------------------------
"""
#%% ~ Interpolation and Regridding
def coarsen_byavg(invar,lat,lon,deg,tol,latweight=True,verbose=True,newlatlon=None,usenan=False):
    """
    Coarsen an input variable to specified resolution [deg]
    by averaging values within a search tolerance for each new grid box.
    
    Dependencies: numpy as np

    Parameters
    ----------
    invar : ARRAY [TIME x LAT x LON]
        Input variable to regrid
    lat : ARRAY [LAT]
        Latitude values of input
    lon : ARRAY [LON]
        Longitude values of input
    deg : INT
        Resolution of the new grid (in degrees)
    tol : TYPE
        Search tolerance (pulls all lat/lon +/- tol)
    
    OPTIONAL ---
    latweight : BOOL
        Set to true to apply latitude weighted-average
    verbose : BOOL
        Set to true to print status
    newlatlon : ARRAY
        Array of desired grid output (lon, lat)
    usenan  : BOOL
        Set to True to use NaN operations
    

    Returns
    -------
    outvar : ARRAY [TIME x LAT x LON]
        Regridded variable       
    lat5 : ARRAY [LAT]
        New Latitude values of input
    lon5 : ARRAY [LON]
        New Longitude values of input

    """

    # Make new Arrays
    if newlatlon is None:
        lon5 = np.arange(0,360+deg,deg)
        lat5 = np.arange(-90,90+deg,deg)
    else:
        lon5,lat5 = newlatlon
        
    # Check lon360
    # outlonflag = False
    # inlonflag  = False
    # if np.any(lon5) < 0:
    #     outlonflag = True
    # if np.any(lon) < 0:
    #     inlonflag = True
    # match = outlonflag == inlonflag
    
    
    # Set up latitude weights
    if latweight:
        _,Y = np.meshgrid(lon,lat)
        wgt = np.cos(np.radians(Y)) # [lat x lon]
        invar *= wgt[None,:,:] # Multiply by latitude weight
    
    # Get time dimension and preallocate
    nt = invar.shape[0]
    outvar = np.zeros((nt,len(lat5),len(lon5)))
    
    # Loop and regrid
    i=0
    for o in range(len(lon5)):
        for a in range(len(lat5)):
            lonf = lon5[o]
            latf = lat5[a]
            
            lons = np.where((lon >= lonf-tol) & (lon <= lonf+tol))[0]
            lats = np.where((lat >= latf-tol) & (lat <= latf+tol))[0]
            
            varf = invar[:,lats[:,None],lons[None,:]]
            
            if latweight:
                wgtbox = wgt[lats[:,None],lons[None,:]]
                if usenan: # NOTE: Need to check if the NAN part is still valid...
                    varf = np.nansum(varf/np.nansum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
                else:
                    varf = np.sum(varf/np.sum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
            else:
                if usenan:
                    varf = np.nanmean(varf,(1,2))
                else:
                    varf = varf.mean((1,2))
            outvar[:,a,o] = varf.copy()
            i+= 1
            msg="\rCompleted %i of %i"% (i,len(lon5)*len(lat5))
            print(msg,end="\r",flush=True)
    return outvar,lat5,lon5

def getpt_pop(lonf,latf,ds,searchdeg=0.5,returnarray=1,debug=False):
    
    
    """ 
    
    Quick script to find and average values on a POP grid, for a DataArray [ds].
    
    Inputs:
        1) lonf = Longitude to find 
        2) latf = Latitude to find
        3) ds   = DataArray with TLONG, TLAT variables and coordinates 'nlon','nlat'
        4) searchdeg = Search Tolerance +/-, in degrees (default is 0.5)
        5) return array = set to 1 to return numpy array instead of DataArray
    
        
    Outputs:
        1) pmean = Mean value for that point
        
    Dependencies
        numpy as np
        xarray as xr
    
    """
    
    # Do same for curivilinear grid
    if lonf < 0:
        lonfc = lonf + 360 # Convert to 0-360 if using negative coordinates
    else:
        lonfc = lonf
        
    # Find the specified point on curvilinear grid and average values
    selectmld = ds.where((lonfc-searchdeg < ds.TLONG) & (ds.TLONG < lonfc+searchdeg)
                    & (latf-searchdeg < ds.TLAT) & (ds.TLAT < latf+searchdeg),drop=True).load()
    if debug:
        print("Found %i points" % (len(selectmld)))
        
    pmean = selectmld.mean(('nlon','nlat'))
    
    if returnarray ==1:
        pmean = np.squeeze(pmean.values)
        return pmean
    else:
        return pmean

def get_pt_nearest(ds,lonf,latf,tlon_name="TLONG",tlat_name="TLAT",debug=True,
                   use_arr=False,tlon=None,tlat=None,returnid=False):
    # ds: Either Dataset or np.array with [time x tlat x tlon]
    #     Adjust the name of TLON with tlong_name (and same for TLAT)
    #     It an array is supplied, use_arr=True and supply tlon and tlat manually!
    # latf: target latitude
    # lonf: target longitude
    # returnid: Set this to True to return the actual indices
    
    # Different version of above but query just the nearest point using nearest neightbor
    if tlon_name is None:
        tlon_name = "TLONG"
    if tlat_name is None:
        tlat_name = "TLAT"
    x1name    = "nlat"
    x2name    = "nlon"
    if tlon is None:
        tlon      = ds[tlon_name].values
    if tlat is None:
        tlat      = ds[tlat_name].values

    # Find minimum in tlon
    # Based on https://stackoverflow.com/questions/58758480/xarray-select-nearest-lat-lon-with-multi-dimension-coordinates
    londiff   =  np.abs(tlon - lonf)
    latdiff   =  np.abs(tlat - latf)
    locdiff   =  londiff+latdiff
    
    # Get Point Indexes
    ([x1], [x2]) = np.where(locdiff == np.min(locdiff))
    #plt.pcolormesh(np.maximum(londiff,latdiff)),plt.colorbar(),plt.show()
    #plt.pcolormesh(locdiff),plt.colorbar(),plt.show()

    if debug:
        print("Nearest point to (%.2f,%.2f) is (%.2f,%.2f) at index (%i,%i)" % (lonf,latf,tlon[x1,x2],tlat[x1,x2],x1,x2))
    
    if returnid:
        return x1,x2
    else:
        if use_arr:
            return ds[:,x1,x2] # Assume Time x Lat x Lon
        else:
            return ds.isel(**{x1name : x1, x2name : x2})


def quick_interp2d(inlons,inlats,invals,outlons=None,outlats=None,method='cubic',
                   debug=False):
    """
    Do quick 2-D interpolation of datapoints using scipy.interpolate.griddata.
    Works with output from sel_region_cv
    
    Parameters
    ----------
    inlons : ARRAY [npts]
        Longitude values of selected points
    inlats : ARRAY [npts]
        Latitude values of selected points
    invals : ARRAY [npts]
        Data values of seleted points
    outlons : 1D-ARRAY [nlon], optional. Default is 1deg between min/max LON
        Longitude values of desired output
    outlats : 1D-ARRAY [nlat], optional
        Latitude values of desired output. Default is 1deg between min/max LAT
    method : STR, optional
        Interpolation method for scipy.griddata. The default is 'cubic'.

    Returns
    -------
    newx : TYPE
        DESCRIPTION.
    newy : TYPE
        DESCRIPTION.
    outvals : TYPE
        DESCRIPTION.

    """
    
    # Make new grid
    if outlons is None:
        newx = np.arange(np.nanmin(inlons),np.nanmax(inlons),1)
    if outlats is None:
        newy = np.arange(np.nanmin(inlats),np.nanmax(inlats),1)
    xx,yy = np.meshgrid(newx,newy)
    
    # Do interpolation
    outvals = scipy.interpolate.griddata((inlons,inlats),invals,(xx,yy),method=method,)
    return newx,newy,outvals

def repair_timestep(ds,t):
    """Given a ds with a timestep with all NaNs, replace value with linear interpolation"""
    
    # Get steps before/after and dimensions
    val_before = ds.isel(time=t-1).values # [zz x tlat x tlon]
    val_0      = ds.isel(time=t).values   # [zz x tlat x tlon]
    val_after  = ds.isel(time=t+1).values # [zz x tlat x tlon]
    orishape   = val_before.shape
    newshape   = np.prod(orishape)
    
    # Do Linear Interp
    x_in       = [0,2]
    y_in       = np.array([val_before.flatten(),val_after.flatten()]) # [2 x otherdims]
    interpfunc = sp.interpolate.interp1d(x_in,y_in,axis=0)
    val_fill   = interpfunc([1]).squeeze()
    
    # Reshape and replace into repaired data array copy
    val_fill   = val_fill.reshape(orishape)
    #ds_fill   = xr.zeros_like(ds.isel(time=t))
    
    ds_new     = ds.copy()
    ds_new.loc[{'time':ds.time.isel(time=t)}] = val_fill
    return ds_new


"""
-------------------------
|||  Climate Analysis ||| ****************************************************
-------------------------
"""
#%% ~ Climate Analysis

def calc_AMV(lon,lat,sst,bbox,order,cutofftime,awgt,runmean=False):
    """
    Calculate AMV Index for detrended/anomalized SST data [LON x LAT x Time]
    given bounding box [bbox]. Applies cosine area weighing

    Parameters
    ----------
    lon : ARRAY [LON]
        Longitude values
    lat : ARRAY [LAT]
        Latitude Values
    sst : ARRAY [LON x LAT x TIME]
        Sea Surface Temperature
    bbox : ARRAY [LonW,LonE,LonS,LonN]
        Bounding Box for Area Average
    order : INT
        Butterworth Filter Order
    cutofftime : INT
        Filter Cutoff, expressed in same timesteps as input data
    awgt : INT (0,1,2)
        Type of Area weighting
        0 = No weight
        1 = cos
        2 = cos^2
    
    runmean : BOOL
        Set to true to do simple running mean
        
    Returns
    -------
    amv: ARRAY [TIME]
        AMV Index (Not Standardized)
    
    aa_sst: ARRAY [TIME]
        Area Averaged SST
    
    # Dependencies
    functions: area_avg
    
    numpy as np
    from scipy.signal import butter,filtfilt
    """
    # Take the weighted area average
    aa_sst   = area_avg(sst,bbox,lon,lat,awgt)

    # Design Butterworth Lowpass Filter
    filtfreq = len(aa_sst)/cutofftime
    nyquist  = len(aa_sst)/2
    cutoff   = filtfreq/nyquist
    b,a      = butter(order,cutoff,btype="lowpass")
    
    # fs = 1/(12*30*24*3600)
    # xtk     = [fs/100,fs/10,fs,fs*12,fs*12*30]
    # xtklabel = ['century','decade','year','mon',"day"]

    # w, h = freqz(b, a, worN=1000)
    # plt.subplot(2, 1, 1)
    # plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    # plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    # plt.axvline(cutoff, color='k')
    # plt.xlim(fs/1200, 0.5*fs)
    # plt.xscale('log')
    # plt.xticks(xtk,xtklabel)
    # plt.title("Lowpass Filter Frequency Response")
    # plt.xlabel('Frequency [Hz]')
    # plt.grid()

    
    # Compute AMV Index
    if runmean:
        amv = np.convolve(aa_sst,np.ones(cutofftime)/cutofftime,mode='same')
    else:
        amv = filtfilt(b,a,aa_sst)
    return amv,aa_sst
    
def calc_AMVquick(var_in,lon,lat,bbox,order=5,cutofftime=10,anndata=False,
                  runmean=False,dropedge=0,monid=None,nmon=12,
                  mask=None,return_unsmoothed=False,verbose=True):
    """
    
    Wrapper for quick AMV calculation.
    
    Inputs:
        1) sst  [lon x lat x time] Array (monthly data)
        2) lon  [lon] Array
        3) lat  [lat] Array
        4) bbox [lonW, lonE, latS, latN] Vector
        OPTIONAL
        5) order (optional, int), order of butterworth filter
        6) cutofftime (optional, int), filter cutoff time in years
        7) anndata - set to 1 if input data is already annual (skip resampling)
        8) runmean [BOOL] set to True to take running mean    
        9) monid : [LIST] Indices of months involved in AMV calculation. 
            Default = Ann average. Requires anndata = 0 to work!
            (ex. monid = [11,0,1] --> DJF AMV Index/Pattern)
        10) nmon : [INT] Number of months for each timestep (for monthly data)
        11) mask : ARRAY [lon x lat] --> Mask to Apply
        
    Outputs:
        1) amvidx     [time]      Array - AMV Index
        2) amvpattern [lon x lat] Array - AMV Spatial
    
    Dependencies:
        from amv.proc:
            sel_region
            ann_avg
            calc_AMV
            regress2ts
        
        numpy as np
    
    """
    
    if monid is None:
        monid = np.arange(0,nmon,1)
    
    # Resample to monthly data 
    if anndata == False:
        sst      = np.copy(var_in)
        annsst   = ann_avg(sst,2,monid=monid,nmon=nmon)
        
    else:
        annsst   = var_in.copy()
    
    # Calculate Index
    if mask is not None: 
        if verbose:
            print("Masking SST for Index Calculation!")
        amvidx,aasst   = calc_AMV(lon,lat,annsst*mask[:,:,None],bbox,order,cutofftime,1,runmean=runmean)
    else:
        amvidx,aasst   = calc_AMV(lon,lat,annsst,bbox,order,cutofftime,1,runmean=runmean)
    
    # Drop boundary points if option is set
    amvidxout = amvidx.copy() # Copy for later (without dropped e ddges)
    if dropedge > 0:
        
        amvidx = amvidx[dropedge:-dropedge]
        annsst = annsst[:,:,dropedge:-dropedge]
    
    # Normalize index
    idxnorm    = amvidx / np.nanstd(amvidx)
    
    # Regress back to SST for spatial pattern
    amvpattern = regress2ts(annsst,idxnorm,nanwarn=0,verbose=verbose)
    if return_unsmoothed:
        return amvidxout,amvpattern,aasst
    return amvidxout,amvpattern

def calc_DMI(sst,lon,lat):
    """
    Calculate the Dipole Mode Index over the Indian Ocean
    Using difference in SSTs over western/eastern boxes from
    Saji et al. 1999
    
    Inputs:
        1) sst : ARRAY[lon x lat x time] - SST anomalies
        2) lon : ARRAY[lon] - Longitudes
        3) lat : ARRAY[lat] - Latitudes
    
    Output:
        1) DMI : ARRAY[time] - Dipole Mode Index
    """
    # Regions to calculate index over (Saji et al. 1999)
    wIO = [50,70,-10,10] # Western Indian Ocean Box
    eIO = [90,110,-10,0]  # Eastern Box
    
    wsst = sel_region(sst,lon,lat,wIO,reg_avg=1)
    esst = sel_region(sst,lon,lat,eIO,reg_avg=1)
    
    DMI = wsst - esst
    return DMI
    
def calc_remidx_simple(ac,kmonth,monthdim=-2,lagdim=-1,
                       minref=6,maxref=12,tolerance=3,debug=False,return_rei=False):
    
    # Select appropriate month
    ac_in          = np.take(ac,np.array([kmonth,]),axis=monthdim).squeeze()
    
    # Compute the number of years involved (month+lags)/12
    nyrs           = int(np.floor((ac_in.shape[lagdim] + kmonth) /12))
    
    # Move lagdim to the front
    ac_in,neworder = dim2front(ac_in,lagdim,return_neworder=True)
    
    # Make an array
    #remidx     = np.zeros((nyrs,),ac_in.shape[1:])    # [year x otherdims]
    maxmincorr = np.zeros((2,nyrs,)+ac_in.shape[1:])  # [max/min x year x otherdims]
    
    if debug:
        maxids = []
        minids = []
    for yr in range(nyrs):
        
        # Get indices of target lags
        minid = np.arange(minref-tolerance,minref+tolerance+1,1) + (yr*12)
        maxid = np.arange(maxref-tolerance,maxref+tolerance+1,1) + (yr*12)
        
        # Drop indices greater than max lag
        maxlag = (ac_in.shape[0]-1)
        minid  = minid[minid<=maxlag]
        maxid  = maxid[maxid<=maxlag]
        if len(minid) == 0:
            continue
        
        if debug:
            print("For yr %i"% yr)
            print("\t Min Ids are %i to %i" % (minid[0],minid[-1]))
            print("\t Max Ids are %i to %i" % (maxid[0],maxid[-1]))
        
        # Find minimum
        mincorr  = np.min(np.take(ac_in,minid,axis=0),axis=0)
        maxcorr  = np.max(np.take(ac_in,maxid,axis=0),axis=0)
        
        maxmincorr[0,yr,...] = mincorr.copy()
        maxmincorr[1,yr,...] = maxcorr.copy()
        #remreidx[yr,...]     = (maxcorr - mincorr).copy()
        
        if debug:
            maxids.append(maxid)
            minids.append(minid)
        
        # if debug:
        #     ploti = 0
        #     fig,ax = plt.subplots(1,1)
        #     acplot = ac_in.reshape(np.concatenate([maxlag,],ac_in.shape[1:].prod()))
            
        #     # Plot the autocorrelation function
        #     ax.plot(np.arange(0,maxlag),acplot[:,ploti])
            
        #     # Plot the search indices
        #     ax.scatter(minid,acplot[minid,ploti],marker="x")
        #     ax.scatter(maxid,acplot[maxid,ploti],marker="+")
            
            
    if debug:
        return maxmincorr,maxids,minids
    if return_rei:
        rei = maxmincorr[1,...] - maxmincorr[0,...]
        return rei
    return maxmincorr

def calc_remidx_xr(ac,minref=6,maxref=12,tolerance=3,debug=False,
                   return_rei=False):
    # Rewritten to work with just an input of acf [lags].
    # For use case, see [calc_remidx_general.py]
    # Rather than explicitly computing # years based on starting month, just assume 12 lags = 1 year
    
    # Compute the number of years involved (month+lags)/12
    # ex: if nlags = 37
    #     if kmonth = 0, nyrs is just 37/12 = ~3 years (computes re-emergence for each year)
    #     if kmonth = 11, nyrs is starts from Y1 (Dec) andgoes to (37+11)/12 (48), to Y4 Dec Re-emergence
    nlags          = ac.shape[0]
    nyrs           = int(np.floor((nlags) /12))
    if debug:
        maxids = []
        minids = []
    maxmincorr = np.zeros((2,nyrs,))  # [max/min x year]
    for yr in range(nyrs):
        
        # Get indices of target lags
        minid = np.arange(minref-tolerance,minref+tolerance+1,1) + (yr*12)
        maxid = np.arange(maxref-tolerance,maxref+tolerance+1,1) + (yr*12)
        
        # Drop indices greater than max lag
        maxlag = (ac.shape[0]-1)
        minid  = minid[minid<=maxlag]
        maxid  = maxid[maxid<=maxlag]
        if len(minid) == 0:
            continue
        
        if debug:
            print("For yr %i"% yr)
            print("\t Min Ids are %i to %i" % (minid[0],minid[-1]))
            print("\t Max Ids are %i to %i" % (maxid[0],maxid[-1]))
        
        # Find minimum
        mincorr  = np.min(np.take(ac,minid,axis=0),axis=0)
        maxcorr  = np.max(np.take(ac,maxid,axis=0),axis=0)
        
        maxmincorr[0,yr,...] = mincorr.copy()
        maxmincorr[1,yr,...] = maxcorr.copy()
        #remreidx[yr,...]     = (maxcorr - mincorr).copy()
        
        if debug:
            maxids.append(maxid)
            minids.append(minid)
        
        if debug:
            ploti = 0
            fig,ax = plt.subplots(1,1)
            acplot=ac
            #acplot = ac.reshape(np.concatenate([maxlag,],ac.shape[1:].prod()))
            
            # Plot the autocorrelation function
            ax.plot(np.arange(0,maxlag+1),acplot)
            
            # Plot the search indices
            ax.scatter(minid,acplot[minid],marker="x")
            ax.scatter(maxid,acplot[maxid],marker="+")
            
    # if return_max:
    #     return maxmincorr[1,...]
    if debug:
        return maxmincorr,maxids,minids
    if return_rei:
        rei = maxmincorr[1,...] - maxmincorr[0,...]
        return rei
    return maxmincorr

def calc_T2(rho,axis=0,ds=False,verbose=False):
    """
    Calculate Decorrelation Timescale (DelSole 2001)
    Inputs:
    rho  : [ARRAY] Autocorrelation Function [lags x otherdims]
    axis : [INT], optional, Axis to sum along (default = 0)
    """
    # if ds:
    #     return (1+2*(rho**2).sum(axis))
    # if np.take(rho,0,axis=axis).squeeze() == 1: # (Take first lag)
    if np.any(rho == 1.):
        if verbose:
            print("Replacing %i values of Corr=1.0 with 0." % (np.sum(rho==1)))
        rho_in = np.where(rho == 1.,0,rho)
    else:
        rho_in = rho
        
    return (1+2*np.nansum(rho_in**2,axis=axis))

def remove_enso(invar,ensoid,ensolag,monwin,reduceyr=True,verbose=True,times=None):
    """
    Remove ENSO via regression of [ensoid] to [invar: time,lat,lon] for each month
    and principle component.
    
    Copied from stochmod/scm on 2024.06.24
    
    Parameters
    ----------
    invar : ARRAY [time x lat x lon]
        Input variable to remove ENSO from
    ensoid : ARRAY [time x month x pc]
        ENSO Index.
    ensolag : INT
        Lag to apply between ENSO Index and Input variable
    monwin : INT
        Size of centered moving window to calculate ENSO for
    reduceyr : BOOL, optional
        Reduce/drop time dimension to account for ensolag. The default is True.
    times : ARRAY[time]
        Array of times to work with
    Returns
    -------
    vout : ARRAY [time x lat x lon]
        Variable with ENSO removed
    ensopattern : ARRAY [month x lat x lon x pc]
        ENSO Component that was removed
    
    """
    allstart = time.time()

    # Standardize the index (stdev = 1)
    ensoid = ensoid/np.std(ensoid,(0,1))
    
    # Shift ENSO by the specified enso lag
    ensoid = np.roll(ensoid,ensolag,axis=1) # (ex: Idx 0 (Jan) is now 11)

    # Reduce years if needed
    if reduceyr: # Since enso leads, drop years from end of period
        dropyr = int(np.fix(ensolag/12) + 1)
        ensoid=ensoid[:-dropyr,:,:]

    # Reshape to combine spatial dimensions
    ntime,nlat,nlon = invar.shape
    nyr             = int(ntime/12)
    vanom           = invar.reshape(nyr,12,nlat*nlon) # [year x mon x space]
    
    # Reduce years if option is set
    if reduceyr:
        vanom=vanom[dropyr:,:,:] # Drop year from beginning of period for lag
        if times is not None:
            times = times.reshape(nyr,12)[dropyr:,:]
    vout            = vanom.copy()
    
    # Variable to save enso pattern
    nyr,_,pcrem = ensoid.shape # Get reduced year, pcs length
    ensopattern = np.zeros((12,nlat,nlon,pcrem))
    
    # Looping by PC...
    for pc in range(pcrem):
        # Looping for each month
        for m in range(12):
            #print('m loop start, vout size is %s'% str(vout[winsize:-winsize,m,:].shape))
            
            # Set up indexing
            if monwin > 1:  
                winsize = int(np.floor((monwin-1)/2))
                monid = [m-winsize,m,m+winsize]
                if monid[2] > 11: # Reduce end month if needed
                    monid[2] -= 12
            else:
                winsize = 0
                monid = [m]
                
            if reduceyr:
                ensoin = indexwindow(ensoid[:,:,[pc]],m,monwin,combinetime=True).squeeze()
                varin  = indexwindow(vanom,m,monwin,combinetime=True)
                nyr   = int(ensoin.shape[0]/monwin) # Get new year dimension
            else:
                # Index corresponding timeseries
                ensoin = ensoid[:,monid,pc] # [yr * mon]  Lagged ENSO
                varin = vanom[:,monid,:] # [yr * mon * space] Variable
                
                # Re-combine time dimensions
                ensoin = ensoin.reshape(nyr*monwin)
                varin = varin.reshape(nyr*monwin,nlat*nlon)
            
            # Check for points that are all nan
            # delete_points = []
            # for i in range(varin.shape[0]):
            #     if np.all(np.isnan(varin[i,:])):
            #         print("All nan at i=%i for month %i, removing from calculation" % (i,m)) 
            #         delete_points.append(i)
            
            #ensoin = np.delete(ensoin,delete_points)
            #varin  = np.delete(varin,delete_points,axis=0)
                    
            # Regress to obtain coefficients [space] # [1xshape]
            varreg,_    = regress_2d(ensoin.squeeze(),varin,nanwarn=1)
            varreg      = varreg.squeeze()
            
            # Write to enso pattern
            ensopattern[m,:,:,pc] = varreg.reshape(nlat,nlon).copy()
            
            # Expand and multiply out and take mean for period [space,None]*[None,time] t0o [space x time]
            ensocomp = (varreg[:,None] * ensoin[None,:]).squeeze()
            
            # Separate year and mon and take mean along months
            ensocomp = ensocomp.reshape(nlat*nlon,nyr,monwin).mean(2)
            
            # Remove the enso component for the specified month
            vout[winsize:-winsize,m,:] -= ensocomp.T
            
            if verbose:
                print("Removed ENSO Component for PC %02i | Month %02i (t=%.2fs)" % (pc+1,m+1,time.time()-allstart))
            # < End Mon Loop>
        # < End PC Loop>
        # Get the correct window size, and reshape it to [time x lat x lon]
    vout = vout[winsize:-winsize,:,:].reshape(nyr*12,nlat,nlon)
    if times is not None:
        times = times[winsize:-winsize,:].flatten()
        return vout,ensopattern,times
    return vout,ensopattern


def calc_HF(sst,flx,lags,monwin,verbose=True,posatm=True,return_cov=False,
            var_denom=None,return_dict=False):
    """
    damping,autocorr,crosscorr=calc_HF(sst,flx,lags,monwin,verbose=True)
    Calculates the heat flux damping given SST and FLX anomalies using the
    formula:
        lambda = [SST(t),FLX(t+l)] / [SST(t),SST(t+l)]
    
    
    Inputs
    ------
        1) sst     : ARRAY [year x time x lat x lon] 
            sea surface temperature anomalies
        2) flx     : ARRAY [year x time x lat x lon]
            heat flux anomalies
        3) lags    : List of INTs
            lags to calculate for (0-N)
        4) monwin  : INT (odd #)
            Moving window of months centered on target month
            (ex. For Jan, monwin=3 is DJF and monwin=1 = J)
        
        --- OPTIONAL ---
        5) verbose : BOOL
            set to true to display print messages
        6) posatm : BOOL
            check to true to set positive upwards into the atmosphere
        
        7) return_cov : BOOL
            True to return covariance values
        8) var_denom : ARRAY [year x time x lat x lon]
            Rather than autovariance, replace SST(t) with vardenom(t).
        9) return_dict : BOOL
            True to return dictionary
        
    Outputs
    -------     
        1) damping   : ARRAY [month x lag x lat x lon]
            Heat flux damping values
        2) autocorr  : ARRAY [month x lag x lat x lon]
            SST autocorrelation
        3) crosscorr : ARRAY [month x lag x lat x lon]
            SST-FLX cross correlation
    """
    # Reshape variables [time x lat x lon] --> [yr x mon x space]
    nyr,nmon,nlat,nlon = sst.shape
    
    sst = sst.reshape(nyr,12,nlat*nlon)
    flx = flx.reshape(sst.shape)
    if var_denom is not None:
        var_denom = var_denom.reshape(sst.shape)
    #sst = sst.reshape(int(ntime/12),12,nlat*nlon)
    #flx = flx.reshape(sst.shape)
    
    # Preallocate
    nlag      = len(lags)
    damping   = np.zeros((12,nlag,nlat*nlon)) # [month, lag, lat, lon]
    autocorr  = np.zeros(damping.shape)
    crosscorr = np.zeros(damping.shape)
    
    covall    = np.zeros(damping.shape)
    autocovall    = np.zeros(damping.shape)
    
    st = time.time()
    for l in range(nlag):
        lag = lags[l]
        for m in range(12):
            
            lm = m-lag # Get Lag Month
            
            # Restrict to time ----
            flxmon = indexwindow(flx,m,monwin,combinetime=True,verbose=False)
            sstmon = indexwindow(sst,m,monwin,combinetime=True,verbose=False)
            if var_denom is None:
                sstlag = indexwindow(sst,lm,monwin,combinetime=True,verbose=False)
            else:
                sstlag = indexwindow(var_denom,lm,monwin,combinetime=True,verbose=False)
            
            # Compute Correlation Coefficients ----
            crosscorr[m,l,:] = pearsonr_2d(flxmon,sstlag,0) # [space]
            autocorr[m,l,:]  = pearsonr_2d(sstmon,sstlag,0) # [space]
            
            # Calculate covariance ----
            cov     = covariance2d(flxmon,sstlag,0)
            autocov = covariance2d(sstmon,sstlag,0)
            
            # Compute damping
            damping[m,l,:] = cov/autocov
            
            # Save covariances
            covall[m,l,:]     = cov
            autocovall[m,l,:] = autocov
            
            print("Completed Month %02i for Lag %s (t = %.2fs)" % (m+1,lag,time.time()-st))
            
    # Reshape output variables
    damping     = damping.reshape(12,nlag,nlat,nlon)  
    autocorr    = autocorr.reshape(damping.shape)
    crosscorr   = crosscorr.reshape(damping.shape)
    covall      = covall.reshape(damping.shape)
    autocovall  = autocovall.reshape(damping.shape)
    
    # Check sign
    if posatm:
        if np.nansum(np.sign(crosscorr)) < 0:
            print("WARNING! sst-flx correlation is mostly negative, sign will be flipped")
            crosscorr*=-1
            covall*=-1
    
    if return_dict:
        outdict = {
            'damping':damping,
            'autocorr':autocorr,
            'crosscorr':crosscorr,
            'autocovall':autocovall,
            'covall':covall
            }
        return outdict
    if return_cov:
        return damping,autocorr,crosscorr,autocovall,covall
    return damping,autocorr,crosscorr



def check_flx(da_flx,flxname=None,return_flag=True,bbox_gs=None):
    # Check to see if it is positive into the atmosphere (and flip it not..)
    if flxname is not None:
        da_in = da_flx[flxname]
    else:
        da_in = da_flx
    
    if bbox_gs is None:
        bbox_gs = [-80,-60,20,40]
    flx_gs   = sel_region_xr(da_in,bbox_gs) # Take Gulf Stream Region
    if 'time' in flx_gs.dims:
        flx_savg = flx_gs.groupby('time.season').mean('time') # Take Seasonal Avg
        flx_wint = flx_savg.sel(season='DJF') # Select Winter
    elif 'month' in flx_gs.dims:
        flx_wint = flx_gs.isel(month=[0,1,2]).mean('month')
    elif 'mon' in flx_gs.dims:
        flx_wint = flx_gs.isel(mon=[0,1,2]).mean('mon')
    else:
        flx_wint = flx_gs # Not selecting time dimension (already simplfied)
    wintsum  = flx_wint.sum(['lat','lon']).data.item() # Sum over winter
    if wintsum < 0:
        print("Warning, wintertime avg values are NEGATIVE over the Gulf Stream.")
        print("\tSign will be flipped to be Positive Upwards (into the atmsophere)")
        #if flxname is not None:
        da_flx = da_flx * -1
        #da_in = da_in * -1
    return da_flx
    
        

#%% ~ Dimension Gymnastics
"""
----------------------------
||| Dimension Gymnastics ||| **********************************************
----------------------------
"""

def combine_dims(var,nkeep,debug=True):
    """
    Keep first n dimensions of var, and 
    reshape to combine the rest

    Parameters
    ----------
    var : ARRAY
        Array to reshape
    nkeep : INT
        Number of dimensions to avoid reshape
    debug : BOOL (optional)
        Prints warning message if reshaping occurs
        
    
    Returns
    -------
    var : ARRAY
        Reshaped Variable
    vshape : TYPE
        Original shape of the variable
    dimflag : BOOL
        True if variable is reshaped

    """
    dimflag = False
    vshape  = var.shape
    if (len(var.shape) > nkeep+1):
        dimflag=True
        if debug:
           print("Warning, variable has more than %i dimensions, combining last!"% (nkeep+1))
        otherdims = np.prod(vshape[nkeep:])
        newshape = np.hstack([vshape[:nkeep],[otherdims]])
        var = var.reshape(vshape[0],vshape[1],otherdims)
    return var,vshape,dimflag

def dim2front(x,dim,verbose=True,combine=False,flip=False,return_neworder=False):
    """
    Move dimension in position [dim] to the front
    
    Parameters
    ----------
    x    [NDARRAY] : Array to Reorganize
    dim      [INT] : Target Dimension
    combine [BOOL] : Set to True to combine all dims
    flip    [BOOL] : Reverse the dimensions
    
    Returns
    -------
    y [ND Array]   : Output

    """
    if dim <0:
        dim = len(x.shape) + dim
        
    neworder = np.concatenate([[dim,],
                         np.arange(0,dim),
                         np.arange(dim+1,len(x.shape))
                         ])
    y = x.transpose(neworder)
    if combine:
        y = y.reshape(y.shape[0],np.prod(y.shape[1:]))
    if flip:
        y = y.transpose(np.flip(np.arange(0,len(y.shape))))
    if verbose:
        print("New Order is : %s"%str(neworder))
        print("Dimensions are : %s"%str(y.shape))
    if return_neworder:
        return y,neworder
    return y

def restoredim(y,oldshape,reorderdim):
    """
    Revert y back to its oldshape, given its current
    reordered state (from dim2front)
    
    Parameters
    ----------
    y          : ARRAY       : Array to restore 
    oldshape   : LIST of INT : Old dimension sizes, in order
    reorderdim : LIST of INT : Reordering of dimensions performed by dim2front
    
    Returns
    -------
    y          : ARRAY       : Array restored to the old shape
    """
    # Get Uncombined Size and reshape
    newshape = [oldshape[i] for i in reorderdim]
    y = y.reshape(newshape)
    
    # Now try to remap things to get the old order
    restore_order = [newshape.index(dim) for dim in oldshape]
    return y.transpose(restore_order) # Back to original order

def flipdims(invar):
    """ Reverse dim order of an n-dimensional array"""
    return invar.transpose(np.flip(np.arange(len(invar.shape))))

"""
---------------------
|||  Convenience ||| ****************************************************
---------------------
"""
#%% ~ Unsorted
def numpy_to_da(invar,times,lat,lon,varname,savenetcdf=None):
    """
    from cvd-12860 tutorials
    Usage: da = numpy_to_da(invar,lon,lat,time,varname)
    
    Converts a NumPy array into an xr.DataArray with the same
    coordinates as the provided arrays.
    
    Parameters
    ----------
    invar : 3D ARRAY[time x lat x lon]
        Input variable
    lon :   1D ARRAY[lon]
        Longitude
    lat : 1D ARRAY[lat]
        Latitude
    time : 1D ARRAY[time]
        Time
    varname : STR
        Name of the variable
    savenetcdf : STR 
        If string argument is provided, saves as netcdf to the
        path indicated by the string. Default is None.
    
    Returns
    -------
    da : xr.DataArray
    """
    da = xr.DataArray(invar,
                dims={'time':times,'lat':lat,'lon':lon},
                coords={'time':times,'lat':lat,'lon':lon},
                name = varname
                )
    if savenetcdf is None:
        return da
    else:
        st = time.time()
        da.to_netcdf(savenetcdf,
                 encoding={varname: {'zlib': True}})
        print("Saving netCDF to %s in %.2fs"% (savenetcdf,time.time()-st))
        return da
    
def cftime2str(times):
    "Convert array of cftime objects to string (YYYY-MM-DD)"
    newtimes = []
    for t in range(len(times)):
        newstr = "%04i-%02i-%02i" % (times[t].year,times[t].month,times[t].day)
        newtimes.append(newstr)
    return np.array(newtimes)

def convert_datenum(matlab_datenum,datestr=False,fmt="%d-%b-%Y %H:%M:%S",autoreshape=True,return_datetimeobj=False,verbose=False):
    """
    Usage: python_datenum = convert_datenum(matlab_datenum,datestr=False,
                                fmt="%d-%b-%Y %H:%M%S",autoreshape=True,
                                return_datetimeobj=False,verbose=False)
    
    Converts an array of Matlab Datenumbers to either an array of np.datetime64 or strings (if datestr=True).
    This considers the offset for different origin date for matlab (Jan 1, year 0) vs. Python (Jan 1, 1970).
    If you prefer to work with datetime objects, set return_datetimeobj to True.
    
    Inputs
    ------
    1) matlab_datenum     : N-D ARRAY of floats
        Array containing matlab datenumbers to convert
    2) datestr            : BOOL
        Set to True to convert output to human-readable strings. Default returns np.datetime64[ns]
    3) fmt                : STR
        Format out output datestring. ex. of default format is "30-Jan-1990 23:59:00"
    4) autoreshape        : BOOL
        By default, the script flattens then reshapes the array to its original dimensions.
        Set to False to just return the flattened array.
    5) return_datetimeobj : BOOL
        By default, the script returns np.datetime64. To just return datetime, set this to true.
        NOTE: The dimensions will be flattened and autoreshape will not work.
    6) verbose            : BOOL
        Set to true to print messages
    
    Output
    ------
    1) python_datetime : N-D ARRAY of np.datetime64[64] or strings
        Array containing the result of the conversion.
    
    Copied from cvd_utils on 2024.01.18
    Dependencies
        import pandas as pd
        import numpy as np
        import datetime as datetime
        
    """
    # Preprocess inputs (flattening n-d arrays)
    in_arr = np.array(matlab_datenum)
    dims   = in_arr.shape
    if len(dims)  > 1:
        if verbose:
            print("Warning! flattening %i-D array with dims %s"%(len(dims),str(dims)))
        in_arr = in_arr.flatten()

    # Calculate offset (In Python, reference date is Jan 1, year 1970 UTC, see "Unix Time")
    # Additional 366 days because reference date in matlab is January 0, year 0000 
    offset = datetime.datetime(1970, 1, 1).toordinal() + 366

    # Convert to Python datetime, considering offset
    # Note that Matlab uses "days since", hence unit="D"
    python_datetime  = pd.to_datetime(in_arr-offset, unit='D')

    # Convert to datestring, and optionally convert to numpy array
    if datestr:
        if verbose:
            print("Converting to datestr")
        python_datetime = python_datetime.strftime(fmt)
        if return_datetimeobj:
            return python_datetime
        # Otherwise convert to string array
        python_datetime = np.array(python_datetime,dtype=str)
    else: # Convert to numpy array with datetime objects
        if return_datetimeobj:
            return python_datetime
        # Otherwise convert to np.datetime64 array
        python_datetime = np.array(python_datetime)

    # Reshape array if necessary (current works only for numpy arrays)
    if len(dims) > 1 and autoreshape:
        if verbose:
            print("Reshaping array to original dimensions!")
        # Reshape array to original dimensions
        python_datetime=python_datetime.reshape(dims) 
    return python_datetime

def noleap_tostr(timestep):
    timestr = "%04i-%02i-%02i" % (timestep.item().year,timestep.item().month,timestep.item().day)
    return timestr

def ds_dropvars(ds,keepvars):
    '''Drop variables in ds whose name is not in the list [keepvars]'''
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in keepvars]
    ds = ds.drop(remvar)
    return ds

def make_encoding_dict(ds,encoding_type='zlib'):
    if type(ds) == xr.core.dataarray.DataArray:
        vname         = ds.name
        encoding_dict = {vname : {encoding_type:True}}
    else:
        keys          = list(ds.keys())
        values        = ({encoding_type:True},) * len(keys)
        encoding_dict = { k:v for (k,v) in zip(keys,values)}
    return encoding_dict

def npz_to_dict(npz,drop_pickle=True):
    """Convert loaded npz file to dict"""
    keys    = npz.files
    if drop_pickle:
        if "allow_pickle" in keys:
            id_pickle = keys.index("allow_pickle")
            keys.pop(id_pickle)# Drop allow_pickle
    newdict = {keys[k]: npz[keys[k]] for k in range(len(keys))}
    return newdict

def format_ds(da,latname='lat',lonname='lon',timename='time',lon180=True,verbose=True):
    """
    Format dataset to match 'lat' from -90 to 90, 'lon' from -180 to 180 and 'time'
    and have consistent specified names. Taken from GulfStream_TBI/grad_funcs on 2023.08.29
    
    Parameters
    ----------
    da          : [xr.DataArray] DataArray to Format
    latname     : STR. Name of latitude dimension. The default is 'lat'.
    lonname     : STR. Name of longitude dimension The default is 'lon'.
    timename    : STR. Name of time dimension. The default is 'time'.
    lon180      : BOOL. True to flip to -180 to 180, False to keep at 0 to 360
    
    Returns
    -------
    da : [xr.DataArray], formatted DataArray

    """
    # Rename lat, lon time
    format_dict = {}
    if latname != "lat":         # Rename Lat
        if verbose:
            print("Renaming lat")
        format_dict[latname] = 'lat'
    if lonname != "lon":         # Rename Lon
        if verbose:
            print("Renaming lon")
        format_dict[lonname] = 'lon'
    if timename != "time":       # Rename time
        if verbose:
            print("Renaming time")
        format_dict[timename] = 'time'
    if len(format_dict) > 0:
        da = da.rename(format_dict)
    
    # Rename variables
    latname = "lat"
    lonname = "lon"
    
    # Flip Latitude to go from -90 to 90
    if (da[latname][1] - da[latname][0]) < 0:
        if verbose:
            print("Flipping Latitude to go from South to North")
        format_dict['lat_original'] = da[latname].values
        da = da.isel(**{latname:slice(None,None,-1)})
        
    # Flip longitude to go from -180 to 180
    if lon180:
        if np.any(da[lonname]>180):
            if verbose:
                print("Flipping Longitude to go from -180 to 180")
            format_dict['lon_original'] = da[lonname].values
            newcoord = {lonname : ((da[lonname] + 180) % 360) - 180}
            da       = da.assign_coords(newcoord).sortby(lonname)
    else:
        if np.any(da[lonname]<0):
            # Note need to test this
            if verbose:
                print("Flipping Longitude to go from 0 to 360")
            format_dict['lon_original'] = da[lonname].values
            newcoord = {lonname : ((da[lonname] + 360) % 360)}
            da       = da.assign_coords(newcoord).sortby(lonname)
        
    # Check if time is divisble by 12 (and send warning if now)
    if (len(da.time)%12):
        print("Warning! Time dimension is not evenly divisible by 12...")
        print("\tMight be missing months of data.")
    
    # Transpose the datase
    da = da.transpose('time','lat','lon')
    
    return da


def format_ds_dims(ds,start_1=True):
    
    # Format Dimensions of Dataset according to specifications for the
    # Stochastic model/re-emergence project...
    
    # Format the dimensions of a dataset
    dimnames = list(ds.dims)
    rename_dict = {}
    if 'ensemble' in dimnames:
        print("Rename <ensemble> --> <ens>")
        rename_dict['ensemble']='ens'
    if 'run' in dimnames:
        print("Rename <run> --> <ens>")
        rename_dict['run'] = 'ens'
    if 'month' in dimnames:
        print("Rename <month> --> <mon>")
        rename_dict['month']='mon'
    ds = ds.rename(rename_dict)
    
    # Format the numbering to start from 1
    if start_1:
        print("Checking dimension indices so that they start numbering from 1...")
        dimnames_new = list(ds.dims)
        formatdims   = ['ens','mode','mon']
        for dd in formatdims:
            if dd in dimnames_new:
                if ds[dd][0].item() == 0:
                    print("\tRenumbering %s starting from 1" % dd)
                    ndim = len(ds[dd])
                    ds[dd] = np.arange(1,ndim+1,1)
    return ds

def savefig_pub(savename,fig=None,
                dpi=1200,transparent=False,format='eps'):
    """
    Save a figure for publication. EPS format.
    Grabs current figure by default.
    Set transparency, DPI, format.
    """
    if fig is None:
        fig = plt.gcf()
    plt.savefig(savename,dpi=dpi,bbox_inches='tight',format=format,
                transparent=transparent)
    
    return None

def check_sum_ds(add_list,sum_ds,lonf=50,latf=-30,t=0,fmt="%.2f"):
    """
    Check sum (sum_ds) of a list of dataarrays at a given point/time.
    
    Parameters
    ----------
    add_list : List of xr.DataArrays that were summed. have lat, lon, time dims
    sum_ds   : DataArray containing the summed result
    lonf,latf: NUMERIC, optional. Lon/Lat indices to check at. The default is 50,-30
    t        : INT, optional. Time indices to check. default is 0

    fmt      : STR, optional. Font Format String.The default is "%.2f".
    
    Returns
    -------
    chkstr : STR. Sum Check String
    """
    # Get Values
    out_pt  = sum_ds.sel(lon=lonf,lat=latf,method='nearest').isel(time=t).values
    list_pt = [ds.sel(lon=lonf,lat=latf,method='nearest').isel(time=t).values for ds in add_list]
    vallist = list_pt + [np.array(list_pt).sum(),out_pt] # List of values
    
    # Make Format String
    fmtstr = ""
    for ii in range(len(list_pt)):
        fmtstr += "%s + " % fmt
    fmtstr = fmtstr[:-2] # Drop last addition sign
    fmtstr += "= %s (obtained %s)" % (fmt,fmt)
    
    # Make Check String
    chkstr  = fmtstr % tuple(vallist)
    print(chkstr)
    return chkstr

def get_xryear(ystart="0000",nmon=12):
    # Get an xarray year (dummy operation). Can indicate startyear or month
    return xr.cftime_range(start='0000',periods=nmon,freq="MS",calendar="noleap")

def rep_ds(ds,repdim,dimname):
    # repeat [ds] a number of times along a selected dimension [repdim], named [dimname]
    nreps  = len(repdim) # Number of reps
    dsrep  = [ds for n in range(nreps)] 
    dsrep  = xr.concat(dsrep,dim=dimname)
    dsrep  = dsrep.reindex(**{ dimname :repdim})
    return dsrep

def nanargmaxds(ds,dimname):
    # Take nanargmax for ds, asmking out nan slices with zero
    # First used in reemergence/monvar_analysis
    mask   = xr.where(~np.isnan(ds),1,np.nan) # Make Mask
    mask   = mask.prod(dimname,skipna=False)
    tempds = xr.where(np.isnan(ds),0,ds) # Set to Zero
    argmax = tempds.argmax(dimname) # Take NanArg
    return argmax * mask

def nanargminds(ds,dimname):
    # Take nanargmin for ds, asmking out nan slices with zero
    # First used in reemergence/monvar_analysis
    mask   = xr.where(~np.isnan(ds),1,np.nan) # Make Mask
    mask   = mask.prod(dimname,skipna=False)
    tempds = xr.where(np.isnan(ds),0,ds) # Set to Zero
    argmin = tempds.argmin(dimname) # Take NanArg
    return argmin * mask

def make_mask(ds_all,nanval=np.nan):
    # From compare_reipy, given a list of dataarrays, make mask
    # that propagates NaNs across all datasets
    if ~np.isnan(nanval):
        ds_all = [xr.where((ds == nanval),np.nan,ds) for ds in ds_all]
    mask    = [xr.where(~np.isnan(ds),1,np.nan) for ds in ds_all]
    mask    = xr.concat(mask,dim='exp')
    mask  = mask.prod('exp',skipna=False)
    return mask


def check_latlon_ds(ds_list,refid=0):
    # Checks "lat" and "lon" in list of reference datasets/dataarrays
    # that are of the same size. Compares it to the lat from the reference
    # ds (whose index/position in the list is indicated by refid)
    lats = [ds.lat.data for ds in ds_list]
    lons = [ds.lon.data for ds in ds_list]
    latref = lats[refid]
    lonref = lons[refid]
    nds    = len(ds_list)
    for dd in range(nds):
        if ~np.all(latref==lats[dd]):
            print("Warning: lat for ds %02i is not matching! Reassigning...")
            ds_list[dd]['lat'] = latref
        if ~np.all(lonref==lons[dd]):
            print("Warning: lon for ds %02i is not matching! Reassigning...")
            ds_list[dd]['lon'] = lonref
    return ds_list


def splittime_ds(ds_in,cutyear):
    # Crop ds_time by cutyear into chunk 1 (earlier) and 2 (later) periods
    # Assumes monthly data and that name of time dimension is "time"
    # See use case in reemergence/era5_acf_sensitivty_analysis
    cutyear = int(cutyear)
    chunk1  = ds_in.sel(time=slice(None,'%04i-12-31' % (cutyear-1)))
    chunk2  = ds_in.sel(time=slice('%04i-01-01' % (cutyear),None))
    return chunk1,chunk2

def get_dtmon(nyrs=None,leap=False):
    # Get number of seconds in a month (unfortunately hard-coded, look into
    # calendar package)
    # Tiling does not support leap year currently...
    ndays = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    if leap:
        ndays[1] = 29
    dtmon = ndays * 60*60*24
    if nyrs is not None: # Tile to timeseries length
        return np.tile(dtmon,nyrs)
    return dtmon

def stdsqsum_da(invar,dim):
    """
    Take the Square root of the sum of squares for [invar] along dimension [dim]. 
    Copied from reemergence/analysis/viz_inputs_paper_draft.py
    """
    return np.sqrt((invar**2).sum(dim))

def printtime(st,print_str="Completed"):
    # Given start time, print the elapsed time in seconds
    print("%s in %.2fs" % (print_str,time.time()-st))

def darkname(figname):
    #Append "_dark" to end of figure name
    return addstrtoext(figname,"_dark")

def selmon_ds(ds,selmon):
    "Select Months [selmon] in a DataArray/DataSet"
    return ds.sel(time=ds.time.dt.month.isin(selmon))

"""
-----------------
|||  Labeling ||| ****************************************************
-----------------
"""
#%% ~ Labeling

def make_locstring(lon,lat,pres=None,lon360=False,fancy=True):
    if lon360 and lon < 0:
        lon += 360
        
    if pres == True:
            
        locfn    = "lon%.4f_lat%.4f" % (lon,lat)
        loctitle = "Lon: %.4f, Lat: %.4f" % (lon,lat)
        if fancy:
            if lon < 0:
                lonsign = "W"
            else:
                lonsign = "E"
            if lat < 0:
                latsign = "S"
            else:
                latsign = "N"
            loctitle = u"%i$\degree$%s, %i$\degree$%s" % (np.abs(lon),lonsign,
                                                         np.abs(lat),latsign)
    else:

        locfn    = "lon%03i_lat%02i" % (lon,lat)
        loctitle = "Lon: %i, Lat: %i" % (lon,lat)
        
        if fancy:
            if lon < 0:
                lonsign = "W"
            else:
                lonsign = "E"
            if lat < 0:
                latsign = "S"
            else:
                latsign = "N"
            loctitle = u"%i$\degree$%s, %i$\degree$%s" % (np.abs(lon),lonsign,
                                                         np.abs(lat),latsign)
            
    return locfn,loctitle

def make_locstring_bbox(bbox,lon360=False):
    bbox_in = bbox.copy()
    if lon360:
        for ii in range(4):
            if bbox_in[ii] < 0:
                bbox_in[ii] += 360
    locfn       = "lon%03ito%03i_lat%03ito%03i" % (bbox_in[0],bbox_in[1],bbox_in[2],bbox_in[3])
    loctitle    = "Lon: %i to %i, Lat: %i to %i" % (bbox_in[0],bbox_in[1],bbox_in[2],bbox_in[3])
    return locfn,loctitle

def makedir(expdir):
    """
    Check if "expdir" exists, and creates a directory if it doesn't

    Parameters
    ----------
    expdir : TYPE
        DESCRIPTION.

    """
    checkdir = os.path.isdir(expdir)
    if not checkdir:
        print(expdir + " Not Found! \n\tCreating Directory...")
        os.makedirs(expdir)
    else:
        print(expdir+" was found!")
    return None

def checkfile(fn,verbose=True):
    """
    Check if "fn" exists, and returns False if not.
    
    Parameters
    ----------
    fn : STR
        File to check, including the absolute or relative path

    """
    checkdir = os.path.isfile(fn)
    if not checkdir:
        exists_flag = False
    else:
        exists_flag = True
    if verbose:
        if exists_flag:
            print(fn + " was found!")
        else:
            print(fn + " Not found!")
    return exists_flag
    
def get_monstr(nletters=3):
    """
    Get Array containing strings of first 3 letters of reach month
    """
    if nletters is None:
        return [cal.month_name[i][:] for i in np.arange(1,13,1)]
    else:
        return [cal.month_name[i][:nletters] for i in np.arange(1,13,1)]

def mon2str(selmon,index=True):
    mons3 = get_monstr()
    """Return String with First Letter of Each Month"""
    if index:
        ''.join([mons3[a][0] for a in selmon])
    else: # Actual month givem so convert to index
        selmon = np.array(selmon) - 1 
        ''.join([mons3[a][0] for a in selmon])
    return 

def addstrtoext(name,addstr,adjust=0):
    """
    Add [addstr] to the end of a string with an extension [name.ext]
    Result should be "name+addstr+.ext"
    -4: 3 letter extension. -3: 2 letter extension
    """
    return name[:-(4+adjust)] + addstr + name[-(4+adjust):]

def get_stringnum(instring,keyword,nchars=1,verbose=True,return_pos=False):
    """
    Finds [keyword] in input string [instring] and grabs [nchars] after the string
    """
    keystart = instring.find(keyword) # Get start of word
    numstart = keystart + len(keyword) # Start grabbing from end of keyword
    grabstr  = instring[numstart:numstart+nchars]
    if verbose:
        print("Grabbed <%s> from end of <%s> at position %i" % (grabstr,
                                                                instring[keystart:keystart+len(keyword)],
                                                                numstart))
    if return_pos:
        return grabstr,numstart
    return grabstr

def fix_febstart(ds):
    # Copied from preproc_CESM.py on 2022.11.15
    if ds.time.values[0].month != 1:
        print("Warning, first month is %s. Fixing."% ds.time.values[0])
        # Get starting year, must be "YYYY"
        startyr = str(ds.time.values[0].year)
        while len(startyr) < 4:
            startyr = '0' + startyr
        nmon = ds.time.shape[0] # Get number of months
        # Corrected Time
        correctedtime = xr.cftime_range(start=startyr,periods=nmon,freq="MS",calendar="noleap")
        ds = ds.assign_coords(time=correctedtime) 
    return ds



