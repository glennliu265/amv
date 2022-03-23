#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

TLONG LAT selection, Trying to underestand curvilinear grid
Created on Tue Mar 22 14:19:32 2022

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
import scipy

#%% Load in curvilinear grids
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
fn      = "tlat_tlon.npz"

ld = np.load(datpath+fn,allow_pickle=True)
print(ld.files)

tlon = ld['tlon']
tlat = ld['tlat']
nlat,nlon = tlon.shape

testvar  = np.sin(np.deg2rad(tlon)) + np.sin(np.deg2rad(tlat))#np.random.normal(0,1,(nlat,nlon))

testvar = testvar[...,None]
#%% First lets try Plotting the points
proj  = ccrs.PlateCarree()
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax.coastlines()
ax.scatter(tlon,tlat,0.02,alpha=0.2)

#%% Ok, lets now try to index/select specific region

bbox = [-80,0,0,65]


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
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
        ax.coastlines()
        ax.scatter(tlon,tlat,0.02,alpha=0.2,color="k")
        ax.scatter(tlon[masklon],tlat[masklon],0.02,alpha=0.2,color="r")
        ax.scatter(tlon[masklat],tlat[masklat],0.02,alpha=0.2,color="b")
        ax.scatter(tlon[masksel],tlat[masksel],0.02,alpha=0.8,color="y",marker="x")
        ax.set_title("Selection Masks \n Lons (Blue) | Lats (Red) | Region (Yellow)")

    # Return selected variables
    selvar = testvar[masksel,...]
    sellon = tlon[masksel]
    sellat = tlat[masksel]
    if return_mask:
        return sellon,sellat,selvar,masksel
    return sellon,sellat,selvar


sellon,sellat,selvar,masksel = sel_region_cv(tlon,tlat,testvar,bbox,return_mask=True,debug=True)


#%%

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


outval = quick_interp2d(sellon,sellat,selvar)

fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(12,4))

# Plot original values
ax = axs[0]
ax.coastlines()
ax.scatter(tlon[masksel],tlat[masksel],c=testvar[masksel,0],s=7.5)
ax.set_title("Scatterplot Original")

ax = axs[1]
ax.coastlines()
pcm  = ax.pcolormesh(outval[0],outval[1],outval[2][...,0])
ax.set_title("Interpolated Data")
    
    
    
    
    

points = np.vstack([sellon,sellat]) # [dim x point]
newx = np.arange(320,360,1)
newy = np.arange(0,66,1)
xx,yy = np.meshgrid(newx,newy)

# Quick Interpolation (not accurate over large distances)
test = scipy.interpolate.griddata((sellon,sellat),selvar,(xx,yy),method='cubic',)

#%% Try plotting the variable

# Plot selection
fig,ax = plt.subplots(1,1,subplot_kw={'projection':proj})
ax.coastlines()
#ax.pcolormesh(sellon,sellat,selvar)
#ax.scatter(sellon,sellat,selvar)
ax.scatter(tlon[masksel],tlat[masksel],s=0.202,alpha=0.7,color="k")
#ax.scatter(tlon,tlat,0.02,alpha=0.2,color="k")
#ax.scatter(tlon[masklon],tlat[masklon],0.02,alpha=0.2,color="r")
#ax.scatter(tlon[masklat],tlat[masklat],0.02,alpha=0.2,color="b")
#ax.scatter(tlon[masksel],tlat[masksel],0.02,alpha=0.8,color="y",marker="x")
#ax.pcolormesh(tlon,tlat,testvar)


#plt.pcolormesh(tlon,tlat,testvar)
#plt.pcolormesh(tlat,tlon,testvar)
