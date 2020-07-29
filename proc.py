#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:12:31 2020

Data Processing Functions

@author: gliu
"""

import numpy as np
import xarray as xr
from scipy import signal,stats


def ann_avg(ts,dim):
    """
    # Take Annual Average of a monthly time series
    where time is axis "dim"
    
    """
    tsshape = ts.shape
    ntime   = ts.shape[dim] 
    newshape =    tsshape[:dim:] +(int(ntime/12),12) + tsshape[dim+1::]
    annavg = np.reshape(ts,newshape)
    annavg = np.nanmean(annavg,axis=dim+1)
    return annavg


# Functions
def regress_2d(A,B):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
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
            
            
            # Set axis for summing/averaging
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
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.sum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    
    return beta,b


def area_avg(data,bbox,lon,lat,wgt):
    
    """
    Function to find the area average of [data] within bounding box [bbox], 
    based on wgt type (see inputs)
    
    Inputs:
        1) data: target array [lat x lon x otherdims]
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
            print('adding year on '+ str(i))
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
        varlag = var2[lagm-1,lag_ts]
        
        # Calculate correlation
        corr_ts[i] = stats.pearsonr(varbase,varlag)[0]
            
        if lagm == basemonth:
            print(i)
            print(corr_ts[i])
                      
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


def lon360to180(lon360,var):
    """
    Convert Longitude from Degrees East to Degrees West 
    Inputs:
        1. lon360 - array with longitude in degrees east
        2. var    - corresponding variable [lon x lat x time]
    """
    
    kw = np.where(lon360 >= 180)[0]
    ke = np.where(lon360 < 180)[0]
    lon180 = np.concatenate((lon360[kw]-360,lon360[ke]),0)
    var = np.concatenate((var[kw,:,:],var[ke,:,:]),0)
    
    return lon180,var


def find_nan(data,dim):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim]
    
    Inputs:
        1) data: 2d array, which will be summed along last dimension
        2) dim: dimension to search along. 0 or 1.
    Outputs:
        1) okdata: data with nan points removed
        2) knan: boolean array with indices of nan points
        

    """
    
    # Sum along select dimension
    datasum = np.sum(data,axis=dim)
    
    
    # Find non nan pts
    knan  = np.isnan(datasum)
    okpts = np.invert(knan)
    
    if dim == 0:
        okdata = data[:,okpts]
    elif dim == 1:    
        okdata = data[okpts,:]
    
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


def regress2ts(var,ts,normalizeall,method):
    
    
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
        var_reg,_ = regress_2d(ts,var)
        
        
        # Reshape to match lon x lat dim
        var_reg = np.reshape(var_reg,(londim,latdim))
    
    
    
    
    # 2nd method is looping point by poin  t
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
