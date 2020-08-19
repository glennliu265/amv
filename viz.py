#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Scripts for visualization
Created on Wed Jul 29 18:02:18 2020

@author: gliu
"""

## Dependencies
import numpy as np
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc


import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
#%% Functions


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
    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.LAND,facecolor='k',zorder=-1)
    
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.top_labels = gl.right_labels = False
    

    
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    
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

def plot_annavg(var,units,figtitle,ax=None,ymax=None,stats='mon'):
    """
    Inputs:
        1) var = monthly variable (1D array)
        2) ax  = axis (defaut, get current axis)
        3) units = ylabel 
        4) figtitle = title of figure
        5) ymax = ylimits (default is None)
        6) stats = 'mon' or 'ann' to indicate stats calc
    
    Dependencies
    
        numpy as np
        matplotlib.pyplot as plt
        quickstatslabel from amv.viz
        ann_avg from amv.proc
    
    """
    if ax is None:
        ax = plt.gca()
    
    # Make time variables
    tper = np.arange(0,len(var),1)
    yper = np.arange(0,len(var),12)
    
    # Ann Avg
    varann = proc.ann_avg(var,0)
    
    if stats == "mon":
        statlabel=quickstatslabel(var)
    elif stats == "ann":
        statlabel=quickstatslabel(varann)
    
    # Plot and add legend
    ax.plot(tper,var)
    ax.plot(yper,varann,color='k',label='Ann. Avg')
    ax.legend()
    
    # Set axis labels
    ax.set_xlabel("Months")
    ax.set_ylabel(units)
    ax.set_title("%s %s" % (figtitle,statlabel))
    
    # Set ymax
    if ymax is not None:
        ax.set_ylim([-1*ymax,ymax])
        ax.set_yticks(np.linspace(-1*ymax,ymax,4))

    return ax

def viz_kprev(h,kprev,locstring=""):
    
    """
    Quick visualization of mixed layer cycle (h)
    and the corresponding detrainment index found using the
    kprev function in scm (or prep_mld scripts)
    
    Inputs:
        1) h - MLD cycle (array of size 12)
        2) kprev - Detraining months (array of size 12)
        3) string indicate location (Lon/Lat)
    
    """
    
    # Create Connector lines ([entrainmon,detrainmon],[MLD,MLD])
    connex = [((im+1,kprev[im]),(h[im],h[im])) for im in range(12) if kprev[im] != 0]
    
    # Indicate entraining months
    foundmon = kprev[kprev!=0]
    foundmld = h[kprev!=0]
    
    # Append month to the end
    plotmon = np.arange(1,14,1)
    plotmld = np.concatenate((h,[h[0]]))
    
    # Start Plot
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    plt.style.use('seaborn-bright')
    
    # Plot the MLD cycle
    ax.plot(plotmon,plotmld,color='k',label='MLD Cycle')
    
    # Plot the connectors
    [ax.plot(connex[m][0],connex[m][1]) for m in range(len(connex))]
    [ax.annotate("%.2f"%(connex[m][0][1]),(connex[m][0][1],connex[m][1][1])) for m in range(len(connex))]
    # Plot Markers
    ax.scatter(foundmon,foundmld,marker="x")
    
    ax.set(xlabel='Month',
           ylabel='Mixed Layer Depth',
           xlim=(1,12),
           title="Mixed Layer Depth Seasonal Cycle \n" + locstring
           )

    ax.set_xticks(range(1,14,1))
    
    return fig,ax


def plot_AMV(amv,ax=None):
    
    """
    Plot amv time series
    
    Dependencies:
        
    matplotlib.pyplot as plt
    numpy as np
    """
    if ax is None:
        ax = plt.gca()
    
    
    htimefull = np.arange(len(amv))
    
    ax.plot(htimefull,amv,color='k')
    ax.fill_between(htimefull,0,amv,where=amv>0,facecolor='red',interpolate=True,alpha=0.5)
    ax.fill_between(htimefull,0,amv,where=amv<0,facecolor='blue',interpolate=True,alpha=0.5)

    return ax

def plot_AMV_spatial(var,lon,lat,bbox,cmap,cint=[0,],clab=[0,],ax=None,pcolor=0):

    fig = plt.gcf()
    
    if ax is None:
        ax = plt.gca()
        ax = plt.axes(projection=ccrs.PlateCarree())
        
    # Add cyclic point to avoid the gap
    var,lon1 = add_cyclic_point(var,coord=lon)
    

    
    # Set  extent
    ax.set_extent(bbox)
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE,facecolor='k')
    
    
    if len(cint) == 1:
        # Automaticall set contours to max values
        cmax = np.nanmax(np.abs(var))
        cmax = np.round(cmax,decimals=2)
        cint = np.linspace(cmax*-1,cmax,9)
    
    
    
    if pcolor == 0:

        # Draw contours
        cs = ax.contourf(lon1,lat,var,cint,cmap=cmap)
    
    
    
        # Negative contours
        cln = ax.contour(lon1,lat,var,
                    cint[cint<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())
    
        # Positive Contours
        clp = ax.contour(lon1,lat,var,
                    cint[cint>=0],
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())    
                          
        #ax.clabel(cln,colors=None)
    else:
        
        cs = ax.pcolormesh(lon1,lat,var,vmin = cint[0],vmax=cint[-1],cmap=cmap)
        
                                
                
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')

    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    gl.xlabel_style={'size':8}
    gl.ylabel_style={'size':8}
    if len(clab) == 1:
        cbar= fig.colorbar(cs,ax=ax,fraction=0.046, pad=0.04,format="%.1e")
        cbar.ax.tick_params(labelsize=8)
    else:
        cbar = fig.colorbar(cs,ax=ax,ticks=clab,fraction=0.046, pad=0.04,format="%.1e")
        cbar.ax.tick_params(labelsize=8)
    #cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in cint], fontsize=10, weight='bold')
    
    return ax

def plot_box(bbox,ax=None,return_line=False,leglab="Bounding Box",color='k',linestyle='solid',linewidth=1):
    
    """
    Plot bounding box
    Inputs:
        1) bbox [1D-ARRAY] [lonW,lonE,latS,latN]
        Optional Arguments...
        2) ax           [axis] axis to plot onto
        3) return_line  [Bool] return line object for legend labeling
        4) leglabel     [str]  Label for legend
        5) color        [str]  Line Color, default = black
        6) linestyle    [str]  Line style, default = solid
        7) linewidth    [#]    Line width, default = 1  
    
    
    """
    if ax is None:
        ax = plt.gca()
    # Plot North Boundary
    ax.plot([bbox[0],bbox[1]],[bbox[3],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_')
    # Plot East Boundary
    ax.plot([bbox[1],bbox[1]],[bbox[3],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_')
    # Plot South Boundary
    ax.plot([bbox[1],bbox[0]],[bbox[2],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_')
    # Plot West Boundary
    ax.plot([bbox[0],bbox[0]],[bbox[2],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_')
    
    if return_line == True:
        linesample =  ax.plot([bbox[0],bbox[0]],[bbox[2],bbox[3]],color=color,ls=linestyle,lw=linewidth,label=leglab)
        return ax,linesample
    return ax