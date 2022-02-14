#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:12:31 2020

Data Processing Functions

test from stormtrack (moved 20200815)
@author: gliu
"""

import numpy as np
import xarray as xr
import calendar as cal
from scipy import signal,stats
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend
import os
import time

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

# Functions
def regress_2d(A,B,nanwarn=1,verbose=True):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    bothND = False # By default, assume both A and B are not 2-D.
    # Note: need to rewrite function such that this wont be a concern...
    
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
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.nansum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.nansum(B,axis=b_axis) - beta * np.nansum(A,axis=a_axis))/A.shape[a_axis]
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
        denom = np.sum(Aanom2,axis=a_axis)
        if bothND:
            denom = denom[:,None] # Broadcast
            
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        if bothND:
            b = (np.sum(B,axis=b_axis)[None,:] - beta * np.sum(A,axis=a_axis)[:,None])/A.shape[a_axis]
        else:
            b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    return beta,b


def area_avg(data,bbox,lon,lat,wgt):
    
    """
    Function to find the area average of [data] within bounding box [bbox], 
    based on wgt type (see inputs)
    
    Inputs:
        1) data: target array [lon x lat x otherdims]
        2) bbox: bounding box [lonW, lonE, latS, latN]
        3) lon:  longitude coordinate
        4) lat:  latitude coodinate
        5) wgt:  number to indicate weight type
                    0 = no weighting
                    1 = cos(lat)
                    2 = sqrt(cos(lat))
    
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
    if wgt != 0:
        
        # Make Meshgrid
        _,yy = np.meshgrid(lon[kw:ke+1],lat[ks:kn+1])
        
        
        # Calculate Area Weights (cosine of latitude)
        if wgt == 1:
            wgta = np.cos(np.radians(yy)).T
        elif wgt == 2:
            wgta = np.sqrt(np.cos(np.radians(yy))).T
        
        # Remove nanpts from weight, ignoring any pt with nan in otherdims
        nansearch = np.sum(sel_data,2) # Sum along otherdims
        wgta[np.isnan(nansearch)] = 0
        
        # Apply area weights
        #data = data * wgtm[None,:,None]
        sel_data  = sel_data * wgta[:,:,None]

    
    # Take average over lon and lat
    if wgt != 0:

        # Sum weights to get total area
        sel_lat  = np.sum(wgta,(0,1))
        
        # Sum weighted values
        data_aa = np.nansum(sel_data/sel_lat,axis=(0,1))
    else:
        # Take explicit average
        data_aa = np.nanmean(sel_data,(0,1))
    
    return data_aa


def eof_simple(pattern,N_mode,remove_timemean):
    """
    Simple EOF function based on script by Yu-Chiao
    
    
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


def calc_lagcovar(var1,var2,lags,basemonth,detrendopt):
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
    
    Outputs:
        1) corr_ts: lag-lead correlation values of size [lags]
    
    Dependencies:
        numpy as np
        scipy signal,stats
    
    """
    
    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length
    totyr = var1.shape[1]
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.ceil(len(np.where(lags < 0)[0])/12))
    lagsize = int(np.ceil(len(np.where(lags > 0)[0])/12))
    
    
    # Detrend variables if option is set
    if detrendopt == 1:
        var1 = signal.detrend(var1,1,type='linear')
        var2 = signal.detrend(var2,1,type='linear')
    
    # Get base timeseries to perform the autocorrelation on
    base_ts = np.arange(0+leadsize,totyr-lagsize)
    varbase = var1[basemonth-1,base_ts]
        
    # Preallocate Variable to store correlations
    corr_ts = np.zeros(lagdim)
    
    # Set some counters
    nxtyr = 0
    addyr = 0
    modswitch = 0
    
    for i in lags:

        lagm = (basemonth + i)%12
        
        if lagm == 0:
            lagm = 12
            addyr = 1         # Flag to add to nxtyr
            modswitch = i+1   # Add year on lag = modswitch
            
        if addyr == 1 and i == modswitch:
            #print('adding year on '+ str(i))
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
        varlag = var2[lagm-1,lag_ts]
        
        # Calculate correlation
        corr_ts[i] = stats.pearsonr(varbase,varlag)[0]
            
        # if lagm == basemonth:
        #     print(i)
        #     print(corr_ts[i])
                      
    return corr_ts


def getpt_pop(lonf,latf,ds,searchdeg=0.5,returnarray=1):
    
    
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
                    & (latf-searchdeg < ds.TLAT) & (ds.TLAT < latf+searchdeg),drop=True)
    
    pmean = selectmld.mean(('nlon','nlat'))
    
    if returnarray ==1:
        pmean = np.squeeze(pmean.values)
        return pmean
    else:
        return pmean
    
    
def find_latlon(lonf,latf,lon,lat):
    """
    Find lat and lon indices
    """
    if((np.any(np.where(lon>180)) & (lonf < 0)) or (np.any(np.where(lon<0)) & (lonf > 180))):
        print("Potential mis-match detected between lonf and longitude coordinates")
    
    klon = np.abs(lon - lonf).argmin()
    klat = np.abs(lat - latf).argmin()
    
    msg1 = "Closest lon to %.2f was %.2f" % (lonf,lon[klon])
    msg2 = "Closest lat to %.2f was %.2f" % (latf,lat[klat])
    print(msg1)
    print(msg2)
    
    return klon,klat

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
    
    var = np.concatenate((var[ke,...],var[kw,...]),0)
    
    if dimflag:
        var = var.reshape(vshape)
    
    return lon360,var


def find_nan(data,dim):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim]
    
    Inputs:
        1) data: 2d array, which will be summed along last dimension
        2) dim: dimension to sum along. 0 or 1
    Outputs:
        1) okdata: data with nan points removed
        2) knan: boolean array with indices of nan points
        3) okpts: indices for non-nan points

    """
    
    # Sum along select dimension
    if len(data.shape) > 1:
        datasum = np.sum(data,axis=dim)
    else:
        datasum = data.copy()
    
    
    # Find non nan pts
    knan  = np.isnan(datasum)
    okpts = np.invert(knan)
    
    if len(data.shape) > 1:
        if dim == 0:
            okdata = data[:,okpts]
        elif dim == 1:    
            okdata = data[okpts,:]
    else:
        okdata = data[okpts]
        
    return okdata,knan,okpts

def year2mon(ts):
    """
    Separate mon x year from a 1D timeseries of monthly data
    """
    ts = np.reshape(ts,(int(np.ceil(ts.size/12)),12))
    ts = ts.T
    return ts



def detrend_dim(invar,dim):
    
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
    tdim = newvar.shape[0]
    otherdims = newvar.shape[1::]
    proddims = np.prod(otherdims)
    newvar = np.reshape(newvar,(tdim,proddims))
    
    # Find non nan points
    varok,knan,okpts = find_nan(newvar,0)
    
    # Ordinary Least Squares Regression
    tper = np.arange(0,tdim)
    m,b = regress_2d(tper,varok)
    
    # Detrend
    ymod = (m[:,None]*tper + b[:,None]).T
    dtvarok = varok - ymod
    
    # Replace into variable of original size
    dtvar  = np.zeros(newvar.shape) * np.nan
    linmod = np.copy(dtvar)
    beta   = np.zeros(okpts.shape) * np.nan
    intercept = np.copy(beta)
    
    dtvar[:,okpts] = dtvarok
    linmod[:,okpts] = ymod
    beta[okpts] = m
    intercept[okpts] = b
    
    # Reshape to original size
    dtvar  = np.reshape(dtvar,((tdim,)+otherdims))
    linmod = np.reshape(linmod,((tdim,)+otherdims))
    beta = np.reshape(beta,(otherdims))
    intercept = np.reshape(beta,(otherdims))
    
    # Tranpose to original order
    oldshape = [dtvar.shape.index(x) for x in varshape]
    dtvar = np.transpose(dtvar,oldshape)
    linmod = np.transpose(linmod,oldshape)
    
    return dtvar,linmod,beta,intercept


def regress2ts(var,ts,normalizeall=0,method=1,nanwarn=1):
    
    
    # Anomalize and normalize the data (time series is assumed to have been normalized)
    if normalizeall == 1:
        varmean = np.nanmean(var,2)
        varstd  = np.nanstd(var,2)
        var = (var - varmean[:,:,None]) /varstd[:,:,None]
        
    # Get variable shapes
    londim = var.shape[0]
    latdim = var.shape[1]
    
    # 1st method is matrix multiplication
    if method == 1:
        
        # Combine the spatial dimensions 

        var = np.reshape(var,(londim*latdim,var.shape[2]))
        
        
        # Find Nan Points
        # sumvar = np.sum(var,1)
        
        # # Find indices of nan pts and non-nan (ok) pts
        # nanpts = np.isnan(sumvar)
        # okpts  = np.invert(nanpts)
    
        # # Drop nan pts and reshape again to separate space and time dimensions
        # var_ok = var[okpts,:]
        #var[np.isnan(var)] = 0
        
        
        # Perform regression
        #var_reg = np.matmul(np.ma.anomalies(var,axis=1),np.ma.anomalies(ts,axis=0))/len(ts)
        var_reg,_ = regress_2d(ts,var,nanwarn=nanwarn)
        
        
        # Reshape to match lon x lat dim
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

def xrdeseason(ds):
    """ Remove seasonal cycle, given an Dataarray with dimension 'time'"""
    return ds.groupby('time.month') - ds.groupby('time.month').mean('time')


def calc_clim(ts,dim,returnts=0):
    """
    Given monthly timeseries with time in axis [dim], calculate the climatology...
    
    Returns: climavg,tsyrmon (if returnts=1)
    
    """
    tsshape = ts.shape
    ntime   = ts.shape[dim] 
    newshape =    tsshape[:dim:] +(int(ntime/12),12) + tsshape[dim+1::]
    
    tsyrmon = np.reshape(ts,newshape)
    climavg = np.nanmean(tsyrmon,axis=dim)
    
    if returnts==1:
        return climavg,tsyrmon
    else:
        return climavg
    
    
    
def calc_lagcovar_nd(var1,var2,lags,basemonth,detrendopt):    
    
    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length
    totyr = var1.shape[1]
    npts  = var1.shape[2]
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.ceil(len(np.where(lags < 0)[0])/12))
    lagsize = int(np.ceil(len(np.where(lags > 0)[0])/12))
    
    
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
            modswitch = i+1   # Add year on lag = modswitch
            
        if addyr == 1 and i == modswitch:
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
        varlag = var2[lagm-1,lag_ts,:]
        
        # Calculate correlation
        corr_ts[i,:] = pearsonr_2d(varbase,varlag,0)       
    return corr_ts

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
  
def ttest_rho(p,tails,dof):
    """
    Perform T-Test, given pearsonr, p (0.05), and tails (1 or 2), and degrees
    of freedom
    
    Edit 12/01/2021, removed rho since it is not used
    
    """
    # Compute p-value based on tails
    ptilde = p/tails
    
    # Get threshold critical value
    critval = stats.t.ppf(1-ptilde,dof)
    
    # Get critical correlation threshold
    corrthres = np.sqrt(1/ ((dof/np.power(critval,2))+1))
    return corrthres
    
    
  
def covariance2d(A,B,dim):
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
    
    return cov


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
        8) awgt: INT, type of area weighting to apply (default is None, no area weight)
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

    """
    
    """
    
    # Dependencies
    functions: area_avg
    
    numpy as np
    from scipy.signal import butter,filtfilt
    """
    
    # Take the weighted area average
    aa_sst = area_avg(sst,bbox,lon,lat,awgt)

    # Design Butterworth Lowpass Filter
    filtfreq = len(aa_sst)/cutofftime
    nyquist  = len(aa_sst)/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
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
                  mask=None):
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
        print("Masking SST for Index Calculation!")
        amvidx,_   = calc_AMV(lon,lat,annsst*mask[:,:,None],bbox,order,cutofftime,1,runmean=runmean)
    else:
        amvidx,_   = calc_AMV(lon,lat,annsst,bbox,order,cutofftime,1,runmean=runmean)
    
    # Drop boundary points if option is set
    amvidxout = amvidx.copy() # Copy for later (without dropped edges)
    if dropedge > 0:
        
        amvidx = amvidx[dropedge:-dropedge]
        annsst = annsst[:,:,dropedge:-dropedge]
        
    # Normalize index
    idxnorm    = amvidx / np.nanstd(amvidx)
    
    # Regress back to SST for spatial pattern
    amvpattern = regress2ts(annsst,idxnorm,nanwarn=0)
    
    
    return amvidxout,amvpattern

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
    fit = np.polyfit(x,y,deg=deg)
    # Prepare matrix (x^n, x^n-1 , ... , x^0)
    #inputs = np.array([np.power(x,d) for d in range(len(fit))])
    inputs = np.array([np.power(x,d) for d in reversed(range(len(fit)))])
    
    # Calculate model
    model = fit.T.dot(inputs)
    
    # Remove trend
    ydetrend = y - model.T
    return ydetrend,model

def coarsen_byavg(invar,lat,lon,deg,tol,latweight=True,verbose=True):
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
    lon5 = np.arange(0,360+deg,deg)
    lat5 = np.arange(-90,90+deg,deg)
    
    
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
                varf = np.sum(varf/np.sum(wgtbox,(0,1)),(1,2)) # Divide by the total weight for the box
            else:
                varf = varf.mean((1,2))
            outvar[:,a,o] = varf.copy()
            i+= 1
            msg="\rCompleted %i of %i"% (i,len(lon5)*len(lat5))
            print(msg,end="\r",flush=True)
    return outvar,lat5,lon5

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

def calc_pearsonconf(rho,conf,tails,n):
    """
    rho   : pearson r
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
    SE = 1/ np.sqrt(n-3)
    
    # Get Confidence
    z_lower = zprime-zcrit*SE
    z_upper = zprime+zcrit*SE
    
    # Convert back to r
    c_lower = np.tanh(z_lower)
    c_upper = np.tanh(z_upper)
    return c_lower,c_upper
    
def make_locstring(lon,lat):
    locfn    = "lon%i_lat%i" % (lon,lat)
    loctitle = "Lon: %i Lat: %i" % (lon,lat)
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
        
def get_monstr(nletters=None):
    """
    Get Array containing strings of first 3 letters of reach month
    """
    mons = [cal.month_name[i][:nletters] for i in np.arange(1,13,1)]
    return mons
        
        
def calc_conflag(ac,conf,tails,n):
    """
    Calculate Confidence Intervals for autocorrelation function

    Parameters
    ----------
    ac : ARRAY [nlags]
        Autocorrelation values by lag
    conf : NUMERIC
        Confidence level (ex. 0.95)
    tails : INT
        # of tails to consider
    n : INT
        Degrees of Freedom

    Returns
    -------
    cflags : ARRAY [nlags]
        Confidence interval for each lag

    """
    cflags = np.zeros((len(ac),2))
    for l in range(len(ac)):
        rhoin = ac[l]
        cfout = calc_pearsonconf(rhoin,conf,tails,n)
        cflags[l,:] = cfout
    return cflags


def nan_inv(invar):
    """
    Invert boolean array with NaNs

    Parameters
    ----------
    invar : ARRAY
        BOOLEAN Array with NaNs

    Returns
    -------
    inverted : ARRAY
        Inverted Boolean Array
    """
    vshape = invar.shape
    inverted  = np.zeros((vshape)) * np.nan
    inverted[invar == True]  = False
    inverted[invar == False] = True
    return inverted
    
    
    
    



#%%

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


def calc_savg(invar,debug=False,return_str=False):
    """
    Calculate Seasonal Average of input with time in the last dimension
    
    Inputs:
        1) invar : ARRAY[...,time], N-D monthly variable where time is the last axis
        2) debug : BOOL, Set to True to Print the Dimension Sizes
        3) return_str : BOOL, Set to True to return the Month Strings (DJF,MAM,...)
    Outputs:
        1) savgs : LIST of ARRAYS, [winter, spring, summer, fall]
        2) snames : LIST of STR, season names, (returns if return_str is true)
    """
    snames = ("DJF"   ,"MAM"  ,"JJA"  ,"SON")
    sids   = ([11,0,1],[2,3,4],[5,6,7],[8,9,10])
    savgs = []
    for s in range(4):
        savgs.append(np.nanmean(invar[...,sids[s]],-1)) # Take mean along last dimension
        if debug:
            print(savgs[s].shape)
    if return_str:
        return savgs,snames
    return savgs

#%% X-array processing

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
    
def maxabs(invar,axis=None):
    # Return max absolute value for 1 variable
    if axis is None:
        invar = invar.flatten()
        axis  = 0
    return np.nanmax(np.abs(invar),axis=axis)


def flipdims(invar):
    # Reverse dim order of an n-dimensional array
    return invar.transpose(np.flip(np.arange(len(invar.shape))))
    
#%% File Utilities

def addstrtoext(name,addstr):
    """
    Add [addstr] to the end of a string with an extension [name.ext]
    Result should be "name+addstr+.ext"
    """
    return name[:-4] + addstr + name[-4:]


    