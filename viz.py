#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Scripts for visualization
Created on Wed Jul 29 18:02:18 2020

@author: gliu
"""
import numpy as np

import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def quickstatslabel(ts):
    """ Quickly generate label of mean ,stdev ,and maximum for a figure title/text
    """
    statsstring = "( Mean:%.2f | Stdev:%.2f | Max:%.2f )" % (np.nanmean(ts),np.nanstd(ts),np.nanmax(np.abs(ts)))
    return statsstring


def quickstats(ts):
    """ Yields nanmean, nanstd, and and absolute max of a timeseries
    """
    tmean = np.nanmean(ts)
    tstd  = np.nanstd(ts)
    tmax  = np.nanmax(np.abs(ts))
    return tmean,tstd,tmax


def init_map(bbox,ax=None):
    """
    Quickly initialize a map for plotting
    """
    # Create Figure/axes
    #fig = plt.gcf() 
    
    #ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.gca()
    #ax = plt.axes(projection=ccrs.PlateCarree())
        
    
    ax.set_extent(bbox)
    
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE,facecolor='k')
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    return ax

def ensemble_plot(var,dim,ax=None,color='k',ysymmetric=1,ialpha=0.1,plotrange=1):
    if ax is None:
        ax = plt.gca()
    
    # Move ensemble dimension the back [time x dim]
    if dim == 0:
        var = var.T
    
    tper = np.arange(0,var.shape[0]) # Preallocate time array
    nens = var.shape[1]
    
    tmax = np.around(np.nanmax(np.abs(var)))
    
    maxens = np.nanmax(var,1)
    minens = np.nanmin(var,1)
    
    
    # Plot for each ensemble member
    for e in range(nens):
        ax.plot(tper,var[:,e],color=color,alpha=ialpha)
        
    # Plot 1 more line for labeling
    ln1 = ax.plot(tper,var[:,-1],color=color,alpha=ialpha,label='Indv. Member')
    ln2 = ax.plot(tper,np.nanmean(var,1),color=color,linewidth=1.5,label="Ens. Avg.")
    
    # Plot maximum and minimum values
    if plotrange == 1:
        ln3 = ax.plot(tper,maxens,color=color,linestyle="dashed",linewidth=0.5,label="Max/Min")
        ax.plot(tper,minens,color=color,linestyle="dashed",linewidth=0.5)
    
    if ysymmetric == 1:
        ax.set_ylim([-1*tmax,tmax])
        
    
    # Set Legend
    if plotrange == 1:
        lns = ln1 + ln2 + ln3
    else:
        lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns,labs,loc=0,ncol=2)
    return ax