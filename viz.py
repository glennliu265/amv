#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    -----------------------
    |||  Visualization  ||| ****************************************************
    -----------------------
        General functions for visualizing data. Mostly works with matplotlib and cartopy.
        
    Created on Wed Jul 29 18:02:18 2020
    @author: gliu
    
        ~ Labeling
    reorder_legend      : Reorder items in a legend
    add_ylabel          : Add y label (for subplots or geoaxis)
    label_barplots      : Add text labels to bar plots (can be inside bar)
    return_mon_label    : Return month labels for a given month
    set_xlim_auto       : Automatically set x-limits to x-tick max/min
    add_ticks           : Helper function to add ticks and gridlines
    
    
        ~ Subplot Management
    init_2rowodd        : Center row with odd or even subplots
    label_sp            : Add text label to each subplot
    
        ~ Cartopy/Mapping
    geosubplots         : Make subplots with geoaxes
    init_map            : Quickly initialize a map for plotting
    plot_box            : Plot bounding box
    get_box_coords      : Get coordinates of bounding box for orthomap plot
    add_coast_grid      : Add land and gridlines (with fill)
    init_fig            : Initialize a figure with geoaxis
    init_blabels        : Initialize dict indicating bounding box labels
    init_orthomap       : Initialize orthographic map over North Atlantic
    
        ~ Time Series/1-D Plots
    quickstatslabel     : Quickly generate label of mean ,stdev ,and maximum for a figure title/text
    quickstats          : Yields nanmean, nanstd, and and absolute max of a timeseries
    plot_annavg          : Plot the seasonal cycle and annual average
    ensemble_plot       : Plot timeseries with max/min and mean of ensemble members
    plot_mean_stdev     : Plot mean and +/- stdev
    init_monplot        : initialize seasonal cycle monthly plot
    
        ~ Power Spectra/Spectral Analysis
    twin_freqaxis       : Twin x-axis on top and label with periods
    make_axtime         : Label twinned axis with text markers (Millenniun, Century, etc.)
    add_yrlines         : Add lines at particular frequencies/periods
    plot_freqxpower     : Quick Frequency x Power plot  (log x   , linear y)
    plot_freqlin        : Linear Frequency x Power plot (linear x, linear y)
    plot_freqlog        : Log-Log plot                  (log x   , log y)

        ~ Spatial/2-D Plots/Contours
    plot_contoursign    : Contour line plot with solid as positive and dashed as negative
    return_clevels      : Return contouring steps
    plot_mask           : Plot stippling based on mask given

        ~ Specialized/Misc.
    viz_kprev           : Visualize mixed-layer cycle with detrainment times (stochmod)
    summarize_params    : 3-panel plot of MLD, Damping, and Forcing at a point (stochmod)
    plot_AMV            : Visualize AMV time series with red/blue fill
    plot_AMV_spatial    : Visualize AMV Pattern
    init_acplot         : Initialized monthly lagged autocorrelation plot
    prep_monlag_labels  : Add month laevls below lag for autocorrelation plots

        ~ Quick Visualization (qv) series
    qv_seasonal         : Plot the seasonal cycle of 2D variable
    hcbar               : Make Horizontal Colorbar
    
"""


# Import
import sys
import cmocean
import string

import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tqdm import tqdm
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import LogLocator
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import matplotlib.path as mpath

# Custom Functions
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc
#%% Functions


# ~~~~~~~~~~~~~~~~~~~~~~
#%% Labeling
# ~~~~~~~~~~~~~~~~~~~~~~

def return_mon_label(m,nletters='all'):
    """
    Return month labels for a given month m
    Inputs
    ------
    1) m : INT
        Number of the month
    2) nletters INT or 'all'
        Number of letters to return
    
    """
    mons = ["January","February","March","April","May","June",
            "July","August","September","October","November","December"]
    if nletters == 'all':
        return mons[m-1]
    else:
        return mons[m-1][:nletters]

def label_barplots(labels,adjustx=2,adjusty=0,ax=None,rects=None,
                   fontcolor='k',fontsize=12):
    """
    labels [ARRAY] : String labels for each barplot
    adjustx [NUMERIC] : How much to move right
    adjusty [NUMERIC] : How much to move up
    rects [ARRAY of matplotlib.patches.Rectangle] : Barplots to label.
    """
    if ax is None:
        ax    = plt.gca()
    if rects is None:
        rects = ax.patches
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / adjustx, height + adjusty,
            label, ha="center", va="bottom",
            color=fontcolor,fontsize=fontsize
        )
    return ax,rects

def set_xlim_auto(ax,xticks):
    """ Automatically set x-limits to limits of xticks """
    ax.set_xlim([xticks[0],xticks[-1]])
    return None

def reorder_legend(ax,order=None):
    """
    Reorder legend items based on code from:
    https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined

    Parameters
    ----------
    ax : TYPE
        Axes containing the objects with labels
    order : TYPE, optional
        Desired new order (using original indices). Default is to flip.

    Returns
    -------
    legend : TYPE
        DESCRIPTION.

    """
    handles, labels = ax.get_legend_handles_labels()
    if order is None:
        order = np.flip(np.arange(0,len(handles))) # Flip order
    legend = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    return legend
    
def add_ylabel(label,ax=None,x=-0.10,y=0.5,fontsize=12,rotation='vertical'):
    if ax is None:
        ax = plt.gca()
    txt = ax.text(x, y, label, va='bottom', ha='center',rotation=rotation,
            rotation_mode='anchor',transform=ax.transAxes,fontsize=fontsize)
    return txt

def add_ticks(ax=None,add_grid=True,grid_lw=0.5,grid_ls="dotted",grid_col="gray",
              bottom=True,top=False,left=True,right=False,facecolor="white",
              spinecolor="k",tickcolor="k",ticklabelcolor="k",fontsize=12,minorx=True,minory=True):
    """
    Add gridlines and ticks (+minorticks) to perimeter of whole plot 
    (control labeling with bottom, top, left, right)
    
    Parameters (all optional)
    ----------
    ax         : matplotlib axes to format. The default is None.
    add_grid   : BOOL. True to add gridlines
    grid_lw    : NUMERIC. Linewidth of gridlines The default is 0.75.
    grid_ls    : STR. Linestyle of gridlines. The default is "dotted".
    grid_col   : STR. Color of gridlines
    bottom     : BOOL. True to label bottom ticks.
    top        : BOOL. True to label top ticks
    left       : BOOL. True to label left ticks
    right      : BOOL. True to label right ticks
    facecolor  : STR  Matplotlib named color or RGB tuple for background
    spinecolor : STR  Border Color
    tickcolor  : STR Tick color
    ticklabelcolor : STR tick label color
    
    
    Returns
    -------
    ax : matplotlib.axes. Formatted axes.
    """
    if ax is None:
        ax = plt.gca()
    if add_grid:
        ax.grid(True,ls=grid_ls,lw=grid_lw,color=grid_col,alpha=0.75,zorder=1) # Add Grid
    ax.tick_params(bottom=True,top=True,left=True,right=True,which='both',color=tickcolor,labelcolor=ticklabelcolor)
    ax.tick_params(labelbottom=bottom,labeltop=top,labelleft=left,labelright=right,which='both',labelsize=fontsize)
    if minorx:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    if minory:
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    #ax.minorticks_on()
    ax.set_facecolor(facecolor)
    ax.spines[:].set_color(spinecolor)
    return ax

# ~~~~~~~~~~~~~~~~~~~~~~
#%% Subplot Management
# ~~~~~~~~~~~~~~~~~~~~~~
def init_2rowodd(ncol,proj=None,figsize=(6,6),oddtop=False,debug=False):
    """
    Initialize a 2-row subplot where
    the bottom row has the smaller number of plots
    source: https://www.tutorialguruji.com/python/matplotlib-allign-uneven-number-of-subplots/

    Parameters
    ----------
    ncol : INT
        Number of columns (even). Last row will contain ncol-1 subplots

    proj : Cartopy Projection
        Projection to set the subplots as
        
    figsize : INT (Length x Height)
        Figure Size
        
    oddtop : BOOL
        Set to True to make odd row on top

    Returns
    -------
    axs : LIST of matplotlib axes
        Flattened list containing subplots

    """
    
    fig = plt.figure(figsize=figsize,constrained_layout=True)
    gs = gridspec.GridSpec(2,ncol*2)
    
    nodd = ncol*2-1
    
    axs = []
    for i in range(ncol*2-1):
        
        
        if oddtop: # Shorter row on top
            if i < ncol-1: 
                rowid   = 0     # Top row
                startid = i*2+1 # Start on 1
                stopid  = i*2+3 # Stop 2 subplots later
                msg = "for %i <= %i --> gs[0,%i:%i]" % (i,ncol-1,startid,stopid)
            else:
                rowid   = 1              # Bot Row
                startid = 2*(i-ncol)+2   # Start from 0 (+2 since i-ncol = -2)
                stopid  = 2*(i-ncol)+4   # End 2 plots later
                msg = "for %i > %i --> gs[1,%i:%i]" % (i,ncol,startid,stopid)
        else: # Shorter row on bottom
            if i < ncol:
                rowid = 0
                startid = 2 * i
                stopid  = 2 * i + 2
                msg = "for %i < %i --> gs[0,%i:%i]" % (i,ncol,startid,stopid)
            else:
                rowid = 1
                startid = 2 * i - nodd
                stopid  = 2 * i + 2 - nodd
                msg = "for %i >= %i --> gs[1,%i:%i]" % (i,ncol,startid,stopid)
        
        ax = plt.subplot(gs[rowid,startid:stopid],projection=proj)
        
        if debug:
            
            print(msg)
            
        axs.append(ax)
    return fig,axs

def label_sp(sp_id,case='upper',inside=True,ax=None,fig=None,x=0.0,y=1.0,
             fontsize=12,fontfamily='sans-serif',alpha=1,labelstyle=None,
             usenumber=False,fontcolor='k'):
    """
    Add alphabetical labels to subplots
    from: https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html
    
    Inputs:
        sp_id [int]                - Subplot Index for alphabet (0=A, 1=B, ...)
        case  ['upper' or 'lower'] - Case of subplot label
        inside [BOOL]              - True to plot inside, False to plot outside
        ax    [mpl.axes]           - axes to plot on. default=current axes
        fig   [mpl.fig]            - figure to scale
        x     [numeric]            - x position relative to upper left
        y     [numeric]            - y position relative to upper left
        fontsize [int]             - font size
        fontfamily [str]           - font family
        alpha [numeric]            - transparency of textbox for inside label
        labelstyle [str]           - labeling style, use %s to indicate string location "%s)"
        usenumber [bool]           - Set to true to use numeric labels (using sp_id)
    """
    
    if usenumber:
        label = str(sp_id)
    else:
        if case == 'upper':
            label = list(string.ascii_uppercase)[sp_id]
        elif case == 'lower':
            label = list(string.ascii_lowercase)[sp_id]
        else:
            print("case must be 'upper' or 'lower'!" )
    
    if labelstyle is None:
        labelstyle="%s)"
    label= labelstyle % (label)
    
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    
    if inside:
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(x, y, label, transform=ax.transAxes + trans,
                fontsize=fontsize, verticalalignment='top',
                bbox=dict(facecolor='1', edgecolor='none', pad=3.0,alpha=alpha),
                color=fontcolor)
    else:
        trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(x, y, label, transform=ax.transAxes + trans,
                fontsize=fontsize, va='bottom',
                color=fontcolor)
    return ax

# ~~~~~~~~~~~~~~~~~~~~~~
#%% Cartopy/Mapping
# ~~~~~~~~~~~~~~~~~~~~~~

def geosubplots(nrows=1,ncols=1,figsize=(12,4),proj=ccrs.PlateCarree(),constrained_layout=True):
    # Shortcut for making subplots with geoaxis
    fig,ax = plt.subplots(nrows,ncols,figsize=figsize,subplot_kw={'projection':proj},
                          constrained_layout=True)
    return fig,ax

def init_map(bbox,crs=ccrs.PlateCarree(),ax=None,return_gl=False):
    """
    Quickly initialize a map for plotting
    """
    if ax is None:
        ax = plt.gca()
    ax.set_extent(bbox,crs)
    
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.LAND,facecolor='k',zorder=-1)
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    if return_gl:
        return ax,gl
    return ax



def plot_box(bbox,ax=None,return_line=False,leglab="Bounding Box",
             color='k',linestyle='solid',linewidth=1,proj=ccrs.PlateCarree()):
    
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
    
    for i in [0,1]:
        if bbox[i] > 180:
            bbox[i] -= 360
            
    if ax is None:
        ax = plt.gca()
    
    # Plot North Boundary
    ax.plot([bbox[0],bbox[1]],[bbox[3],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    # Plot East Boundary
    ax.plot([bbox[1],bbox[1]],[bbox[3],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    # Plot South Boundary
    ax.plot([bbox[1],bbox[0]],[bbox[2],bbox[2]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    # Plot West Boundary
    ax.plot([bbox[0],bbox[0]],[bbox[2],bbox[3]],color=color,ls=linestyle,lw=linewidth,label='_nolegend_',transform=proj)
    
    if return_line == True:
        linesample =  ax.plot([bbox[0],bbox[0]],[bbox[2],bbox[3]],color=color,ls=linestyle,lw=linewidth,label=leglab,transform=proj)
        return ax,linesample
    return ax

def get_box_coords(bbox,dx=None,dy=None):
    # Get coordinates of box for plotting an ortho map/polygon
    # Given [Westbound EastBound Southbound Northbound]
    # Returns xcoords and ycoords of path drawn from:
    # Lower Left, counterclockwise around and back.
    
    if dx is None:
        dx = np.linspace(bbox[0],bbox[1],5)
        dx = dx[1] - dx[0]
    if dy is None:
        dy = np.linspace(bbox[2],bbox[3],5)
        dy = dy[1] - dy[0]
    
    # Lower Edge (Bot. Left --> Bot. Right)
    lower_x = np.arange(bbox[0],bbox[1]+dx,dx) # x-coord
    nx = len(lower_x) 
    lower_y = [bbox[2],]*nx # y-coord
    
    # Right Edge (Bot. Right ^^^ Top Right)
    right_y = np.arange(bbox[2],bbox[3]+dy,dy)
    ny = len(right_y)
    right_x = [bbox[1],]*ny
    
    # Upper Edge (Top Left <-- Top Right)
    upper_x = np.flip(lower_x)
    upper_y = [bbox[3],]*nx
    
    # Left Edge (Bot. Left vvv Top Left)
    left_y  = np.flip(right_y)
    left_x  = [bbox[0],]*ny
    
    x_coords = np.hstack([lower_x,right_x,upper_x,left_x])
    y_coords = np.hstack([lower_y,right_y,upper_y,left_y])
    
    return x_coords,y_coords

def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None,blabels=[1,0,0,1],ignore_error=False,
                   fill_color=None,line_color='k',grid_color='gray',c_zorder=1,
                   fix_lon=False,fix_lat=False,fontsize=12):
    """
    Add Coastlines, grid, and set extent for geoaxes
    
    Parameters
    ----------
    ax : matplotlib geoaxes
        Axes to plot on 
    bbox : [LonW,LonE,LatS,LatN], optional
        Bounding box for plotting. The default is [-180,180,-90,90].
    proj : cartopy.crs, optional
        Projection. The default is None.
    blabels : ARRAY of BOOL [Left, Right, Upper, Lower] or dict
        Lat/Lon Labels. Default is [1,0,0,1]
    ignore_error : BOOL
        Set to True to ignore error associated with gridlabeling
    fill_color : matplotlib color string
        Add continents with a given fill
    c_zorder : layering order of the continents
    
    Returns
    -------
    ax : matplotlib geoaxes
        Axes with setup
    """
    
    if type(blabels) == dict: # Convert dict to array
        blnew = [0,0,0,0]
        if blabels['left'] == 1:
            blnew[0] = 1
        if blabels['right'] == 1:
            blnew[1] = 1
        if blabels['upper'] == 1:
            blnew[2] = 1
        if blabels['lower'] == 1:
            blnew[3] = 1
        blabels=blnew
    
    if proj is None:
        proj = ccrs.PlateCarree()
        
    if fill_color is not None: # Shade the land
        ax.add_feature(cfeature.LAND,facecolor=fill_color,zorder=c_zorder)
    #ax.add_feature(cfeature.COASTLINE,color=line_color,lw=0.75,zorder=0)
    ax.coastlines(color=line_color,lw=0.75)
    ax.set_extent(bbox,proj)
    
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=2, color=grid_color, alpha=0.5, linestyle="dotted",
                  lw=0.75)
    
    # Remove the degree symbol
    if ignore_error:
        #print("Removing Degree Symbol")
        gl.xformatter = LongitudeFormatter(zero_direction_label=False,degree_symbol='')
        gl.yformatter = LatitudeFormatter(degree_symbol='')
        #gl.yformatter = LatitudeFormatter(degree_symbol='')
        gl.rotate_labels = False
    
    if fix_lon is not False:
        gl.xlocator = mticker.FixedLocator(fix_lon)
    if fix_lat is not False:
        gl.ylocator = mticker.FixedLocator(fix_lat)
    
    gl.left_labels = blabels[0]
    gl.right_labels = blabels[1]
    gl.top_labels   = blabels[2]
    gl.bottom_labels = blabels[3]
    
    # Set Fontsize
    gl.xlabel_style = {'size':fontsize}
    gl.ylabel_style = {'size':fontsize}
    return ax

def init_fig(nrow,ncol,proj=ccrs.PlateCarree(),figsize=(8,6),
             sharex=False,sharey=False,constrained_layout=True):
    fig,ax=plt.subplots(nrow,ncol,figsize=figsize,
                        sharex=sharex,sharey=sharey,
                        subplot_kw={'projection':proj},
                        constrained_layout=constrained_layout)
    return fig,ax


def init_blabels():
    """
    Initialize bounding box labels with keys
    'left','right','upper','lower' with all values set to False
    """
    return {'left':0,'right':0,'upper':0,'lower':0}

def init_orthomap(nrow,ncol,bboxplot,centlon=-40,centlat=35,precision=40,
                  dx=10,dy=5,
                  frame_lw=2,frame_col="k",
                  figsize=(8,4.5),constrained_layout=True):
    # Intiailize Ortograpphic map over North Atlantic.
    # Based on : https://stackoverflow.com/questions/74124975/cartopy-fancy-box
    # The default lat/lon projection
    noProj = ccrs.PlateCarree(central_longitude=0)
    
    # Set Orthographic Projection
    myProj = ccrs.Orthographic(central_longitude=centlon, central_latitude=centlat)
    myProj._threshold = myProj._threshold/precision  #for higher precision plot
    
    # Initialize Figure
    fig,axs = plt.subplots(nrow,ncol,figsize=figsize,subplot_kw={'projection': myProj},
                          constrained_layout=constrained_layout)
    
    # Get Line Coordinates
    xp,yp  = get_box_coords(bboxplot,dx=dx,dy=dy)
    
    # Draw the line
    if nrow ==1 and ncol ==1:
        #print("Nd Axis")
        axs = [axs,]
        ndaxis=False
    else:
        orishape = axs.shape
        axs      = axs.flatten()
        ndaxis   = True
    for ax in axs:
        [ax_hdl] = ax.plot(xp,yp,
            color=frame_col, linewidth=frame_lw,
            transform=noProj)
        
        # Make a polygon and crop
        tx_path                = ax_hdl._get_transformed_path()
        path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
        polygon1s              = mpath.Path( path_in_data_coords.vertices)
        ax.set_boundary(polygon1s) # masks-out unwanted part of the plot
        
    if ndaxis is False:
        axs = axs[0] # Return just the axis
    else:
        axs = axs.reshape(orishape)
    mapdict={
        'noProj'     : noProj,
        'myProj'     : myProj,
        'line_coords': (xp,yp),
        'polygon'    : polygon1s,
        }
    return fig,axs,mapdict

# ~~~~~~~~~~~~~~~~~~~~~~
#%% Time Series/1-D Plot
# ~~~~~~~~~~~~~~~~~~~~~~

def quickstatslabel(ts,fmt="%.2f"):
    """ Quickly generate label of mean ,stdev ,and maximum for a figure title/text
    """   
    statsstring = "( Mean:%.2e | Stdev:%.2e | Max:%.2e )" % (np.nanmean(ts),np.nanstd(ts),np.nanmax(np.abs(ts)))
    return statsstring

def quickstats(ts):
    """ Yields nanmean, nanstd, and and absolute max of a timeseries
    """
    tmean = np.nanmean(ts)
    tstd  = np.nanstd(ts)
    tmax  = np.nanmax(np.abs(ts))
    return tmean,tstd,tmax

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


def ensemble_plot(var,dim,ax=None,color='k',ysymmetric=1,ialpha=0.1,plotrange=1,returnlegend=True,returnline=False,plotindv=True):
    """
    
    plotrange [BOOL] - Set to true to plot max and minimum values
    returnlegend [BOOL] - set to true to return the legend
    returnline [BOOL] - set to True to return the ens average line object
    """
    if ax is None:
        ax = plt.gca()
    # Move ensemble dimension the back [time x dim]
    if dim == 0:
        var = var.T
    tper   = np.arange(0,var.shape[0]) # Preallocate time array
    nens   = var.shape[1]
    tmax   = np.around(np.nanmax(np.abs(var)))
    maxens = np.nanmax(var,1)
    minens = np.nanmin(var,1)
    # Plot for each ensemble member
    if plotindv==True:
        for e in range(nens):
            ax.plot(tper,var[:,e],color=color,alpha=ialpha)
        # Plot 1 more line for labeling and add to collection
        ln1 = ax.plot(tper,var[:,-1],color=color,alpha=ialpha,label='Indv. Member')
        lns = ln1
    # Plot ens average
    ln2 = ax.plot(tper,np.nanmean(var,1),color=color,linewidth=1.5,label="Ens. Avg.")
    if plotindv==True: # Add ensemble mean toline ollection
        lns += ln2
    else:
        lns = ln2
    # Plot maximum and minimum values
    if plotrange == 1:
        ln3 = ax.plot(tper,maxens,color=color,linestyle="dashed",linewidth=0.5,label="Max/Min")
        ax.plot(tper,minens,color=color,linestyle="dashed",linewidth=0.5)
        lns += ln3
    if ysymmetric == 1:
        ax.set_ylim([-1*tmax,tmax])
    # Set Legend
    if returnlegend==True:
        labs = [l.get_label() for l in lns]
        ax.legend(lns,labs,loc=0,ncol=2)
    elif returnline == True:
        if len(lns) > 1:
            if plotindv==True:
                return ax,lns[1] # Get 2nd line
            return ax,lns[0] # Get 1st line
        return ax,ln2
    return ax

def plot_mean_stdev(invar,axis,
                    ax=None,x_vals=None,
                    stdev=1,
                    return_lines=False,
                    alpha=0.1,
                    color="k"):
    """
    
    Plots meanline and N-standard deviations for a timeseries on the current (or given)
    axes

    Parameters
    ----------
    invar           (ARR)            : Timeseries to visualize 
    axis            (INT)            : Axis to take the mean/stdev along
    ax              (mpl.axes)       : Axes to plot on, default=current axes, optional
    x_vals          (ARR)            : Corresponding x-values, default is 0 to invar.shape[axis], optional
    stdev           (FLOAT)          : Number of stdevs to plot, optional
    return_lines    (BOOL)           : Set to True to return line objects, optional
    alpha           (FLOAT)          : Transparency of stdev shading. Default is 0.1, optional
    color           (STR)            : Color of the lines and region. Default is "k", optional

    Returns
    -------
    mu              (ARR)            : Mean of [invar] along [axis]
    sigma           (ARR)            : Standard deviation of [invar] along [axis]
    mean_line       (mpl.obj)        : (if return_line=True), Matplotlib object of mean line
    shaded_region   (mpl.obj)        : (if return_line=True), Matplotlib object of shadded region

    """
    
    # Get unspecified arguments
    if ax is None:
        ax        = plt.gca() 
    if x_vals is None:
        x_vals    = np.arange(0,invar.shape[axis])
    
    # Calculate mean/stdev
    mu            = np.nanmean(invar,axis)
    sigma         = np.nanstd(invar,axis) * stdev
    
    # Plot
    mean_line     = ax.plot(x_vals,mu,
                            color=color,
                            zorder= -9)
    shaded_region = ax.fill_between(x_vals,mu-sigma,mu+sigma,
                                    alpha=alpha,color=color,
                                    zorder=1)
    if return_lines:
        return mu,sigma,mean_line,shaded_region
    else:
        return mu,sigma
    

def init_monplot(row,col,figsize=(6,4),constrained_layout=True,skipaxis=False):
    fig,axs = plt.subplots(row,col,figsize=figsize,constrained_layout=True)
    tks = proc.get_monstr(nletters=3)
    if row+col > 2:
        for aa,ax in enumerate(axs.flatten()):
            if skipaxis is not False:
                if aa in skipaxis:
                    continue
            ax.set_xlim([0,11])
            ax.set_xticks(np.arange(0,12,1))
            ax.set_xticklabels(tks)
            ax = add_ticks(ax,minorx=False)
    else:
        axs.set_xlim([0,11])
        axs.set_xticks(np.arange(0,12,1))
        axs.set_xticklabels(tks)
        axs = add_ticks(axs,minorx=False)
    return fig,axs
        
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Power Spectra/Spectral Analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_axtime(ax,htax,denom='year'):
    
    # Units in Seconds
    dtday = 3600*24
    dtyr  = dtday*365
    
    fnamefull = ("Millennium","Century","Decade","Year","Month")
    if denom == 'month':
        
        # Set frequency (by 10^n months, in seconds)
        fs = [1/(dtyr*1000),1/(dtyr*100),1/(dtyr*10),1/(dtyr),1/(dtday*30)]
        xtk      = np.array(fs)#/dtin
        
        # Set frequency tick labels
        fsl = ["%.1e" % s for s in xtk]
        
        # Set period tick labels
        per = [ "%.1e \n (%s) " % (int(1/fs[i]/(dtday*30)),fnamefull[i]) for i in range(len(fnamefull))]
        
        # Set axis names
        axl_bot = "Frequency (cycles/sec)" # Axis Label
        axl_top = "Period (Months)"
        
        
    elif denom == 'year':
        
        # Set frequency (by 10^n years, in seconds)
        denoms = [1000,100,10,1,.1]
        fs = [1/(dtyr*1000),1/(dtyr*100),1/(dtyr*10),1/(dtyr),1/(dtyr*.1)]
        xtk      = np.array(fs)#/dtin
        
        # Set tick labels for frequency axis
        fsl = ["%.3f" % (fs[i]*dtyr) for i in range(len(fs))]
        
        # Set period tick labels
        per = [ "%.0e \n (%s) " % (denoms[i],fnamefull[i]) for i in range(len(fnamefull))]
        
        # Set axis labels
        axl_bot = "Frequency (cycles/year)" # Axis Label
        axl_top = "Period (Years)"
        
    for i,a in enumerate([ax,htax]):
        a.set_xticks(xtk,minor=False)
        if i == 0:
            a.set_xticklabels(fsl)
            a.set_xlabel("")
            a.set_xlabel(axl_bot)
        else:
            a.set_xticklabels(per)
            a.set_xlabel("")
            a.set_xlabel(axl_top)
        
        #x_major = LogLocator(base = dtyr, numticks = 1)
        #a.xaxis.set_major_locator(x_major)
        #minor_locator = AutoMinorLocator(10)
        #a.xaxis.set_minor_locator(minor_locator)
    #ax.grid(True,which='major',ls='dotted',lw=0.5)
    return ax,htax


def twin_freqaxis(ax,freq,tunit,dt,fontsize=12,mode='log-lin',xtick=None,include_title=True,usegrid=True):
    # Set top axis for linear-log plots, in terms of cycles/sec
    
    # Make top axis
    htax   = ax.twiny()
    
    # Set and get bottom axis units
    if xtick is None:
        xmin = 10**(np.floor(np.log10(np.min(freq))))
        ax.set_xlim([xmin,0.5/dt])
        htax.set_xlim([xmin,0.5/dt])
    else:
        xlm = [xtick[0],xtick[-1]]
        ax.set_xticks(xtick)
        ax.set_xlim(xlm)
        htax.set_xticks(xtick)
        htax.set_xlim(xlm)
        xtkl = ["%.1f" % (1/x) for x in xtick]
        #print(xtkl)
        htax.set_xticklabels(xtkl)
    
    if mode == 'log-lin': # Log (x) Linear (y)
        
        htax.set_xscale("log")
        htax.set_yscale("linear")
        
    elif mode == 'lin-log': # Lin(x) Log (y)
        htax.set_yscale("log")
        htax.set_xscale("linear")
        
    elif mode == 'lin-lin': # Linear (x,y)
        # Note this is probably not even needed...
        htax.set_xscale("linear")
        htax.set_yscale("linear")
        
    elif mode == 'log-log':
        htax.set_xscale("log")
        htax.set_yscale("log")
    
    if usegrid:
        ax.grid(True,ls='dotted')
    if include_title:
        htax.set_xlabel("Period (%s)"%tunit,fontsize=fontsize)
    
    # Make sure axis ticks are the same
    htax.set_xticks(ax.get_xticks())
    return htax

def add_yrlines(ax,dt=1,label=False):
    
    vlv = [1/(100*365*24*3600),1/(10*365*24*3600),1/(365*24*3600)]
    vlv = np.array(vlv) * dt
    if label==False:
        vll = ["","",""]
    else:
        vll = ["Century","Decade","Year"]
    for v,vv in enumerate(vlv):
        ax.axvline(vv,color='k',ls='dashed',label=vll[v],lw=0.75)
    return ax


def plot_freqxpower(specs,freqs,enames,ecolors,
                    plotdt=3600*24*365,ax=None,xtick=None,xlm=None,
                    plotconf=None,plottitle=None,alpha=None,return_ax2=False):
    """
    Frequency x Power plot.

    Parameters
    ----------
    specs : ARRAY [spec1,spec2]
        List containing each spectra to plot
    freqs : ARRAY [freq1,freq2]
        Corresponding frequency for each spectra
    enames : ARRAY of strings
        Labels for each spectra
    ecolors : ARRAY of strings
        Color for each line
    plotdt : INT, optional
        Plotting timestep. The default is annual (3600*24*365).
    ax : matplotlib.axis, optional
        Axis to plot on. The default is None.
    xtick :  ARRAY of floats, optional
        Frequencies to tick. The default is None.
    xlm : TYPE, optional
        X-limits of plot. The default is None.
    plotconf : ARRAY[CC1,CC2] where CC = [:,1] is the 95% confidence, optional
        Include the 95% confidence level. The default is None.
    plottitle : STR, optional
        Title of plot. The default is None.
    alpha : LIST [Numeric,...]
        List of alpha values. Default is 1 for all.

    Returns
    -------
    ax : matplotlib axes
        AAxes with the plot.
    """
    
    # Set default plotting parameters
    if ax is None:
        ax = plt.gca()
    if xtick is None:
        xtick  = [float(10)**(x) for x in np.arange(-4,2)]
    if xlm is None:
        xlm    = [5e-4,10]
    if plottitle is None:
        plottitle="Spectral Estimate"
    if alpha is None:
        alpha = np.ones(len(specs))
    
    # Plot spectra
    for n in range(len(specs)):
        ax.semilogx(freqs[n]*plotdt,specs[n]*freqs[n],color=ecolors[n],label=enames[n],
                    alpha=alpha[n])
        
        if plotconf is not None: # Plot 95% Significance level
            ax.semilogx(freqs[n]*plotdt,plotconf[n][:,1]*freqs[n],label="",color=ecolors[n],
                        ls="dashed")

    # Set Axis Labels
    ax.set_ylabel("Frequency x Power ($(\degree C)^{2}$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    
    # Twin x-axis for period
    htax = twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-lin',xtick=xtick,include_title=False)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl   = ["%i" % (1/x) for x in xtick2]
    for i in range(len(xtkl)): # Adjust ticks
        if (1/xtick2[i] < 1) and (1/xtick2[i] > 0):
            xtkl[i] = "%.2f" % (1/xtick2[i])
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)  
    ax.legend(fontsize=10)
    ax.set_title(plottitle)
    
    if return_ax2:
        return ax,htax
    return ax

# --------------
def plot_freqlin(specs,freqs,enames,ecolors,
                plotdt=3600*24*365,ax=None,xtick=None,xlm=None,
                plotconf=None,plottitle=None,alpha=None,return_ax2=False,marker=None,
                lw=1,plotids=None,legend=True,linearx=0,usegrid=True):
    """
    Linear-Linear Plot

    Parameters
    ----------
    specs : ARRAY [spec1,spec2]
        List containing each spectra to plot
    freqs : ARRAY [freq1,freq2]
        Corresponding frequency for each spectra
    enames : ARRAY of strings
        Labels for each spectra
    ecolors : ARRAY of strings
        Color for each line
    plotdt : INT, optional
        Plotting timestep. The default is annual (3600*24*365).
    ax : matplotlib.axis, optional
        Axis to plot on. The default is None.
    xtick :  ARRAY of floats, optional
        Frequencies to tick. The default is None.
    xlm : TYPE, optional
        X-limits of plot. The default plots the multidecadal band
    plotconf : ARRAY[CC1,CC2] where CC = [:,1] is the 95% confidence, optional
        Include the 95% confidence level. The default is None.
    plottitle : STR, optional
        Title of plot. The default is None.
    alpha : LIST [Numeric,...]
        List of alpha values. Default is 1 for all.
    lw    : Numeric
        Linewidth
    plotids : list of int
        Indices of which specs/freqs to plot
    legend : BOOL
        Set to true to include legend (default=True)
    linearx : [0,1,2]
        0 - Set both frequency and period axis to xtks
        1 - Keep frequency axis linear, period set to xtks
        2 - Keep both axes linear

    Returns
    -------
    ax : matplotlib axes
        AAxes with the plot.
    """
    
    # Set default plotting parameters
    if ax is None:
        ax = plt.gca()
    if xtick is None:
        xtick    = np.array([0,0.02,0.04,0.1,0.2])
    if xlm is None:
        xlm      = [0,0.2]
    if plottitle is None:
        plottitle="Spectral Estimate"
    if alpha is None:
        alpha = np.ones(len(specs))
    
    # Plot spectra
    if plotids is None:
        plotids = range(len(specs))
    for n in plotids:
        ax.plot(freqs[n]*plotdt,specs[n]/plotdt,color=ecolors[n],label=enames[n],
                    alpha=alpha[n],marker=marker,lw=lw)
        
        if plotconf is not None: # Plot 95% Significance level
            ax.plot(freqs[n]*plotdt,plotconf[n][:,1]/plotdt,label="",color=ecolors[n],ls="dashed")

    # Set Axis Labels, plot Titles
    ax.set_ylabel("Power ($(\degree C)^{2} / cpy$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    ax.set_title(plottitle)
    
    # Handle Second Axis
    ax.set_xlim(xlm)
    
    if linearx == 0:
        # Twin x-axis for period
        htax = twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',
                             xtick=xtick,include_title=False,usegrid=usegrid)
        
        # Set upper x-axis ticks
        xtick2 = htax.get_xticks()
        xtkl   = ["%i" % (1/x) for x in xtick2]
        for i in range(len(xtkl)): # Adjust ticks
            if (1/xtick2[i] < 1) and (1/xtick2[i] > 0):
                xtkl[i] = "%.2f" % (1/xtick2[i])
        htax.set_xticklabels(xtkl)
        
        # Set second axis limits to match
        htax.set_xlim(xlm)
        if legend:
            ax.legend(fontsize=10,ncol=2)
    else:
        
        htax = ax.twiny()  # Twin axis
        htax.set_xlim(xlm) # Match Axis Limits
        
        if linearx == 1:# Keep frequency axis linear
        
            # Set 2nd axis to xticks
            htax.set_xticks(xtick)
            # Calculate period from frequency, then label
            xtkl   = ["%i" % (1/x) for x in xtick]
            for i in range(len(xtkl)): # Adjust ticks
                if (1/xtick[i] < 1) and (1/xtick[i] > 0):
                    xtkl[i] = "%.2f" % (1/xtick[i])
            htax.grid(True,ls='dotted',lw=1)
        elif linearx == 2:
            
            # Set 2nd axis to values on first
            xtick2 = ax.get_xticks()
            htax.set_xticks(xtick2)
            xtkl   = ["%s" % (str(1/x)) for x in xtick2]
            for i in range(len(xtkl)): # Adjust ticks
                if (1/xtick2[i] < 1) and (1/xtick2[i] > 0):
                    xtkl[i] = "%.2f" % (1/xtick2[i])
        htax.set_xticklabels(xtkl) # Set second axis tick labels
        if usegrid:
            ax.grid(True,ls='dotted',color="k")
            
    
    if return_ax2:
        return ax,htax
    return ax
# --------------
def plot_freqlog(specs,freqs,enames,ecolors,
                plotdt=3600*24*365,ax=None,xtick=None,xlm=None,
                plotconf=None,plottitle=None,alpha=None,return_ax2=False,lw=1,
                plotids=None,legend=True,semilogx=False,semilogy=False):
    """
    Log-Log Plot

    Parameters
    ----------
    specs : ARRAY [spec1,spec2]
        List containing each spectra to plot
    freqs : ARRAY [freq1,freq2]
        Corresponding frequency for each spectra
    enames : ARRAY of strings
        Labels for each spectra
    ecolors : ARRAY of strings
        Color for each line
    plotdt : INT, optional
        Plotting timestep. The default is annual (3600*24*365).
    ax : matplotlib.axis, optional
        Axis to plot on. The default is None.
    xtick :  ARRAY of floats, optional
        Frequencies to tick. The default is None.
    xlm : TYPE, optional
        X-limits of plot. The default plots the multidecadal band
    plotconf : ARRAY[CC1,CC2] where CC = [:,1] is the 95% confidence, optional
        Include the 95% confidence level. The default is None.
    plottitle : STR, optional
        Title of plot. The default is None.
    alpha : LIST [Numeric,...]
        List of alpha values. Default is 1 for all.

    Returns
    -------
    ax : matplotlib axes
        AAxes with the plot.
    """
    
    # Set default plotting parameters
    if ax is None:
        ax = plt.gca()
    if xtick is None:
        xtick  = [float(10)**(x) for x in np.arange(-4,2)]
    if xlm is None:
        xlm    = [5e-4,10]
    if plottitle is None:
        plottitle="Spectral Estimate"
    if alpha is None:
        alpha = np.ones(len(specs))
    
    # Plot spectra
    if plotids is None:
        plotids = range(len(specs))
    for n in plotids:
        
        if semilogx:
            mode='log-lin'
            ax.semilogx(freqs[n]*plotdt,specs[n]/plotdt,color=ecolors[n],label=enames[n],
                        alpha=alpha[n],lw=lw)
            
        elif semilogy:
            mode='lin-log'
            ax.semilogy(freqs[n]*plotdt,specs[n]/plotdt,color=ecolors[n],label=enames[n],
                        alpha=alpha[n],lw=lw)
        
        else:
            mode='log-log'
            ax.loglog(freqs[n]*plotdt,specs[n]/plotdt,color=ecolors[n],label=enames[n],
                        alpha=alpha[n],lw=lw)
        
        if plotconf is not None: # Plot 95% Significance level
            ax.loglog(freqs[n]*plotdt,plotconf[n][:,1]/plotdt,label="",color=ecolors[n],ls="dashed")

    # Set Axis Labels
    ax.set_ylabel("Power ($(\degree C)^{2} / cpy$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    
    # Twin x-axis for period
    htax = twin_freqaxis(ax,freqs[1],"Years",plotdt,mode=mode,xtick=xtick,include_title=False)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl   = ["%i" % (1/x) for x in xtick2]
    for i in range(len(xtkl)): # Adjust ticks
        if (1/xtick2[i] < 1) and (1/xtick2[i] > 0):
            xtkl[i] = "%.2f" % (1/xtick2[i])
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)  
    if legend:
        ax.legend(fontsize=10,ncol=2)
    ax.set_title(plottitle)
    if return_ax2:
        return ax,htax
    return ax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Spatial/2-D Plots/Contours
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_contoursign(var,lon,lat,cint,ax=None,bbox=None,clab=True,clab_fmt="%.1f",lw=1,add_cyc=False):
    """
    Plot contours, with solid as positive and dashed as negative values
    Inputs:
        1) varin [Array: lat x lon] - Input Variable
        2) lon [Array: lon] - Longitude values (tested with 360, should work with 180)
        3) lat [Array: lat] - Latitude values
        4) cint [Array: levels] - contour levels (negative and positive)
        Optional Arguments
            5) ax [geoaxes] - Axes to plot on
            6) bbox [Array: lonW,lonE,latS,latN] - Region to plot
            7) clab [Bool] - Option to include contour labels
            8) clab_fmt [Str] - String formatting option
            9) lw [Float] - Contour Linewidths 
    
    """
    # Get current axis if none is assigned
    if ax is None:
        ax = plt.gca()
    

    #Set Plotting boundaries/extent
    if bbox != None:
        ax.set_extent(bbox)
    
    # # Add cyclic point to remove blank meridian
    if add_cyc is True:
        var,lon = add_cyclic_point(var,lon) 

    # Negative contours
    cln = ax.contour(lon,lat,var,
                cint[cint<0],
                linestyles='dashed',
                colors='k',
                linewidths=lw)
        
    # Positive Contours
    clp = ax.contour(lon,lat,var,
                cint[cint>=0],
                colors='k',
                linewidths=lw,
                transform=ccrs.PlateCarree())
    if clab is True:
        # Add Label
        plt.clabel(cln,fmt=clab_fmt,fontsize=8)
        plt.clabel(clp,fmt=clab_fmt,fontsize=8)
    
    return ax

def plot_mask(lon,lat,mask,reverse=False,color="k",marker="o",markersize=1.5,
              ax=None,proj=None,geoaxes=False):
    
    """
    Plot stippling based on a mask
    
    1) lon     [ARRAY] : Longitude values
    2) lat     [ARRAY] : Latitude values
    3) mask    [ARRAY] : (Lon,Lat) Mask (True = Where to plot Stipple)
    4) reverse [BOOL]  : Set to True to reverse the mask values
    5) color [STR] : matplotlib color
    6) marker [STR] : matplotlib markerstyle
    7) markersize [STR] : matplotlib markersize

    Solution from: https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_corner_mask.html
    
    """
    if proj is None:
        if geoaxes:
            proj = ccrs.PlateCarree()
        else:
            proj = None
    # Get current axis
    if ax is None:
        ax = plt.gca()
        
    # Invert Mask
    if reverse:
        # Inversion doesnt work with NaNs...
        nlon,nlat = mask.shape
        newcopy   = np.zeros((nlon,nlat)) * np.nan
        newcopy[mask == True]  = False
        newcopy[mask == False] = True
        mask      = newcopy.copy()
    
    # Make meshgrid and plot masked array
    yy,xx = np.meshgrid(lat,lon)
    if geoaxes:
        smap = ax.plot(np.ma.array(xx,mask=mask),yy,
                       c=color,marker=marker,markersize=markersize,ls="",transform=proj)
    else:
        smap = ax.plot(np.ma.array(xx,mask=mask),yy,
                       c=color,marker=marker,markersize=markersize,ls="")
    return smap 

def return_clevels(cmax,cstep,lstep=None):
    # cmax : Contour limit
    # cstep : Contour interval
    # lstep : Label Inverval
    
    clevels   = np.arange(-cmax,cmax+cstep,cstep)
    if lstep is None:
        return clevels
    clabels   = np.arange(-cmax,cmax+lstep,lstep)
    return clevels, clabels

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Specialized/Misc.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def viz_kprev(h,kprev,locstring="",ax=None,lw=1,msize=25,mstyle="x",
              cmap=None,fill_style='full',txtalpha=1,usetitle=True,
              fsz_lbl=12,fsz_axis=14,plotarrow=False,shade_layers=True):
    """
    Quick visualization of mixed layer cycle (h)
    and the corresponding detrainment index found using the
    kprev function in scm (or prep_mld scripts)
    
    Inputs:
        1) h - MLD cycle (array of size 12)
        2) kprev - Detraining months (array of size 12)
        3) string indicate location (Lon/Lat)
        4) lw = linewidths
        5) msize = markersize
        6) mstyle = markerstyle
        7) cmap = colors of the lines/markers
        8) fill_style = marker fill style
        9) txtalpha = Alpha of text background/highlight
    
    """
    
    if ax is None:
        ax = plt.gca()
        
    # Create Connector lines ([entrainmon,detrainmon],[MLD,MLD])
    connex = [([im+1,kprev[im]],[h[im],h[im]]) for im in range(12) if kprev[im] != 0]
    # Set colormap
    if cmap is None:
        cmap = cc.glasbey[:len(connex)]
    else:
        cmap = cmap[:len(connex)]
    for m in range(len(connex)):
        if connex[m][0][0] < connex[m][0][1]: # The first point comes before last point
            connex[m][0][0] += 12 # Shift 1 year ahead
    
    # Indicate entraining months
    foundmon = kprev[kprev!=0]
    foundmld = h[kprev!=0]
    
    dtmon = kprev[kprev == 0]
    dtmld = h[kprev==0]
    
    # Append month to the end
    plotmon = np.arange(1,14,1)
    plotmld = np.concatenate((h,[h[0]]))
    
    # Start Plot
    #fig,ax = plt.subplots(1,1,figsize=(6,4))
    #plt.style.use('seaborn-bright')
    
    # Plot the MLD cycle
    ax.plot(plotmon,plotmld,color='k',label='MLD Cycle',lw=lw,zorder=1)
    
    # Plot the connectors
    if plotarrow:
        lns  = []
        clrs = []
        for m in range(len(connex)):
            if m == h.argmax():
                hl = 0 
                hw = 0 
            else:
                hl = 0.15
                hw = 3.5
                
            arx  = connex[m][0][1]
            ary  = connex[m][1][1]
            ardx = connex[m][0][0] - connex[m][0][1]
            ardy = connex[m][1][0] - connex[m][1][1]
            ln   = ax.arrow(arx,ary,ardx,ardy,color=cmap[m],zorder=1,head_starts_at_zero=False,
                            head_length=hl,head_width=hw)
            lns.append(ln)
            clrs.append(ln.get_facecolor())
            
    else:
        lns  = [ax.plot(connex[m][0],connex[m][1],
                        lw=lw,color=cmap[m],zorder=-1) for m in range(len(connex))]
        clrs = [ln[0].get_color() for ln in lns] 
    
    # Plot markers
    pts  = [ax.plot(foundmon[m],foundmld[m],
                    markersize=msize,marker=mstyle,color=clrs[m],fillstyle=fill_style,
                    linestyle='None',zorder=5) for m in range(len(connex))]
    
    
    # Plot Annotations
    txts = [ax.annotate("%.2f" %(connex[m][0][1]),(connex[m][0][1],connex[m][1][1]),
                 zorder=9,fontsize=fsz_lbl) for m in range(len(connex))]
    for txt in txts:
        txt.set_path_effects([PathEffects.withStroke(linewidth=2.5,foreground="w",alpha=txtalpha)])
        
    # Plot Entrain Arrows
    ax.scatter(np.arange(1,13,1)[kprev!=0],h[kprev!=0],
               marker=r"$\circlearrowleft$",s=250,color=cmap,zorder=3,#edgecolor="w",
               linewidth=0.99,)
        
    # Labeling
    ax.set(xlabel='Month',
           ylabel='Mixed Layer Depth (m)',
           xlim=(1,12),
           )
    
    # Add Shading
    if shade_layers:
        ax.fill_between(plotmon,0,plotmld,alpha=0.90,color="cornflowerblue",zorder=-5)
        ax.fill_between(plotmon,plotmld,plotmld.max()+50,alpha=0.70,color="navy",zorder=-5)
        ax.set_ylim([0,plotmld.max()+10])
    
    if usetitle:
        ax.set_title("Mixed Layer Depth Seasonal Cycle at " + locstring)

    ax.set_xticks(range(1,14,1))
    ax.invert_yaxis()
    
    return ax


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


def plot_AMV_spatial(var,lon,lat,bbox,cmap,cint=[0,],clab=[0,],ax=None,pcolor=0,labels=True,fmt="%.1f",clabelBG=False,fontsize=10,returncbar=False,
                     omit_cbar=False):
    
    fig = plt.gcf()
    
    if ax is None:
        ax = plt.gca()
        ax = plt.axes(projection=ccrs.PlateCarree())
        
    # Add cyclic point to avoid the gap
    var,lon1 = add_cyclic_point(var,coord=lon)
    
    # Set  extent
    ax.set_extent(bbox)
    
    # Add filled coastline
    ax.add_feature(cfeature.LAND,color=[0.4,0.4,0.4])
    
    if len(cint) == 1:
        # Automaticall set contours to max values
        cmax = np.nanmax(np.abs(var))
        cmax = np.round(cmax,decimals=2)
        cint = np.linspace(cmax*-1,cmax,9)
    
    if pcolor == 0:

        # Draw contours
        cs = ax.contourf(lon1,lat,var,cint,cmap=cmap,extend='both')
        
        cs.cmap.set_over('red')
        cs.cmap.set_under('blue')
    
    
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
                          
        if labels is True:
            clabelsn= ax.clabel(cln,colors=None,fmt=fmt,fontsize=fontsize)
            clabelsp= ax.clabel(clp,colors=None,fmt=fmt,fontsize=fontsize)
            
            # if clabelBG is True:
            #     [txt.set_backgroundcolor('white') for txt in clabelsn]
            #     [txt.set_backgroundcolor('white') for txt in clabelsp]
    else:
        
        cs = ax.pcolormesh(lon1,lat,var,vmin = cint[0],vmax=cint[-1],cmap=cmap)
        
                                
                
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    gl.xlabel_style={'size':fontsize}
    gl.ylabel_style={'size':fontsize}
    
    # Colorvar
    if omit_cbar is True:
        return ax,cs
    
    if len(clab) == 1:
        cbar= fig.colorbar(cs,ax=ax,fraction=0.046, pad=0.04,format=fmt)
        cbar.ax.tick_params(labelsize=fontsize)
    else:
        cbar = fig.colorbar(cs,ax=ax,ticks=clab,fraction=0.046, pad=0.04,format=fmt,shrink=0.95)
        cbar.ax.tick_params(labelsize=fontsize)
    
    #cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in cint], fontsize=10, weight='bold')
    
    if returncbar:
        return ax,cbar
    return ax


def summarize_params(lat,lon,params,synth=False):
    """
    Creates a quick 3-panel plot of Damping, MLD, and Forcing
    at a single point

    Parameters
    ----------
    lat : ARRAY
        Latitudes
    lon : ARRAY
        Longitudes
    params : Tuple
        Contains the output of scm.get_data, specifically:
            [lon_index,lat_index],damping,mld,dentrain_month,forcing           
    synth: Tuple
        Contains synthetic data for
            [damping,mld,forcing]

    Returns
    -------
    fig,ax (matplotlib objects)
    """
    alpha = 1
    if synth is not False:
        alpha = 0.25
        
    # Initialized Figure
    xtks = np.arange(1,13,1)
    locstring = "LON: %.1f, LAT: %.1f" % (lon[params[0][0]],lat[params[0][1]]) 
    fig,axs = plt.subplots(3,1,figsize=(4,6),sharex=True)
    
    # Plot Damping
    ax = axs[0]
    ax.plot(xtks,params[1],color='r',alpha=alpha,label="Seasonal Mean")
    ax.set_ylabel("Damping $(W/m^{2})$")
    ax.set_xticks(xtks)
    ax.grid(True,linestyle='dotted')
    if synth is not False:
        ax.plot(xtks,synth[0],color='r',label="Model Input")
        ax.legend(fontsize=8)
    
    # Plot MLD
    ax = axs[1] 
    if synth is not False: # Don't Plot Kprev for constant MLD
        ax.plot(xtks,params[2],alpha=alpha,color='b',label="Seasonal Mean")
        ax.plot(xtks,synth[1],color='b',label="Model Input")
        ax.legend(fontsize=8)
    else:
        ax=viz_kprev(params[2],params[3],ax=ax)
    ax.set_xticks(xtks)
    ax.set_title("")
    ax.set_ylabel("Mixed-Layer Depth (m)")
    ax.grid(True,linestyle='dotted',color="w",zorder=5)
    

    
    # Plot Forcing
    ax = axs[2]
    
    ax.plot(xtks,params[4],color='k',label="Seasonal Mean",alpha=alpha)
    ax.set_ylabel("Forcing $(W/m^{2})$")
    ax.set_xticks(xtks)
    ax.grid(True,linestyle='dotted')
    fmax = np.abs(params[4]).max()
    if synth is not False:
        ax.plot(xtks,synth[2],color='k',label="Model Input")
        ax.legend(fontsize=8)
    ax.set_ylim([-fmax,fmax])
    plt.suptitle(locstring)
    plt.tight_layout()
    return fig,ax

def init_acplot(kmonth,xticks,lags,ax=None,title=None,loopvar=None,
                usegrid=True,tickfreq=None,fsz_axis=14,fsz_ticks=12,fsz_title=18,vlines=None):
    """
    Function to initialize autocorrelation plot with months on top,
    lat on the bottom
    
    Parameters
    ----------
    kmonth : INT
        Index of Month corresponding to lag=0.
    xticks : ARRAY
        Lags that will be shown
    lags : ARRAY
        Lags to visulize
    ax : matplotlib object, optional
        Axis to plot on
    title : STR, optional
        Title of plot. The default is "SST Autocorrelation, Lag 0 = Month.
    loopvar: ARRAY [12,], optional
        Monthly variable to tile and plot in the background
    vlines: indices of months to put vertical values at
    
    Returns
    -------
    ax,ax2, and ax3 if loopvar is not None : matplotlib object
        Axis with plot

    """
    if ax is None:
        
        ax  = plt.gca()
    
    # Tile Months for plotting
    mons3     = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
    mons3tile = np.tile(np.array(mons3),int(np.floor(len(lags)/12))) 
    mons3tile = np.concatenate([np.roll(mons3tile,-kmonth),[mons3[kmonth]]])
    
    # Set up second axis
    ax2 = ax.twiny()
    ax2.set_xticks(xticks)#,size=fsz_ticks)
    ax2.set_xticklabels(mons3tile[xticks], rotation = 45,fontsize=fsz_ticks)
    ax2.set_axisbelow(True)
    ax2.grid(zorder=0,alpha=0)
    ax2.set_xlim(xticks[[0,-1]])
    
    # Plot second variable if option is set
    if loopvar is not None:
        ax3 = ax.twinx()
        loopvar = proc.tilebylag(kmonth,loopvar,lags)
        ax3.plot(lags,loopvar,color='gray',linestyle='dashed')
        ax3.tick_params(axis='y',labelcolor='gray',fontsize=fsz_ticks)
        ax3.grid(False)
    
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=fsz_ticks)
    ax.set_xlim([xticks[0],xticks[-1]])
    if title is None:
        ax.set_title("SST Autocorrelation, Lag 0 = %s" % (mons3[kmonth]),fontsize=fsz_title)
    else:
        ax.set_title(title)
    ax.set_xlabel("Lags (Months)",fontsize=fsz_axis)
    ax.set_ylabel("Correlation",fontsize=fsz_axis)
    if usegrid:
        ax.grid(True,linestyle='dotted')
    #plt.tight_layout()
    
    # Adjust ticks if option is set
    if tickfreq is not None:
        lbl_new_mon = []
        lbl_new     = []
        for i in range(len(xticks)):
            
            ilag = xticks[i]
            
            if i%tickfreq == 0:
                lbl_new_mon.append(mons3tile[ilag])
                lbl_new.append(lags[ilag])
            else:
                lbl_new_mon.append("")
                lbl_new.append("")
        ax.set_xticklabels(lbl_new,fontsize=fsz_ticks)
        ax2.set_xticklabels(lbl_new_mon,fontsize=fsz_ticks)
    
    # Add some vertical lines
    if vlines is not None:
        vline_mons = [mons3[mm] for mm in vlines]
        for l,lag in enumerate(lags):
            lbl = mons3tile[l]
            if lbl in vline_mons:
                ax.axvline([l],lw=0.75,c="k",label="")
    
    if loopvar is not None:
        return ax,ax2,ax3
    return ax,ax2

def prep_monlag_labels(kmonth,lagtick,label_interval,useblank=True):
    """
    Add month labels below the lag for autocorrelation plots

    Parameters
    ----------
    kmonth : Int
        DESCRIPTION.
    lagtick : TYPE
        DESCRIPTION.
    label_interval : TYPE
        DESCRIPTION.
    useblank : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    mon_labels : TYPE
        DESCRIPTION.

    """
    mon_labels = []
    kmonth_seen = []
    mons3       = [return_mon_label(m,nletters=3) for m in np.arange(1,13)]
    
    for t,tk in enumerate(lagtick):
        if tk%label_interval == 0:
            monlbl = [(kmonth+tk)%12]
            if monlbl in kmonth_seen:
                lbl = tk
            else:
                lbl = "%i\n %s" % (tk,mons3[(kmonth+tk)%12])
                #kmonth_seen.append(monlbl) # Uncomment this to only plot first feb/aug
            #print(lbl)
        else:
            if useblank:
                lbl = ""
            else:
                lbl = tk
        mon_labels.append(lbl)
    return mon_labels

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Quick Visualization (qv) series
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def qv_seasonal(lon,lat,var,
                add_coast=True,cmap="inferno",
                bbox=None,anom=False,vmax=None,
                contour=False,cints=None):
    """
    Quickly Plot the seasonal cycle of a 2d variable
    var = [lon x lat x month]
    """
    
    fig,axs = plt.subplots(4,3,figsize=(12,12),constrained_layout=True,
                           subplot_kw={'projection':ccrs.PlateCarree()}) # (each row is a season)
    monloop = np.roll(np.arange(0,12),1) # Start with Dec. 
    
    if bbox is None:
        bbox =[lon[0],lon[-1],lat[0],lat[-1]]
    
    for i,im in tqdm(enumerate(monloop)):
        
        ax      = axs.flatten()[i]
        
        # Set labels
        blabel = [0,0,0,0]
        if i%3 == 0:
            blabel[0] = 1
        if i>8:
            blabel[3] = 1
            
        if add_coast:
            ax      = add_coast_grid(ax,bbox=bbox,blabels=blabel,fill_color="gray")
        
        plotvar = var[:,:,im].T
        
        if anom:
            if vmax is None: # Find maximum value in dataset
                vmax = np.nanmax(np.abs(plotvar.flatten()))
                
            if contour:
                pcm = ax.contourf(lon,lat,plotvar,cmap=cmocean.cm.balance,levels=np.linspace(-vmax,vmax,10))
            else:
                pcm = ax.pcolormesh(lon,lat,plotvar,cmap=cmocean.cm.balance,vmin=-vmax,vmax=vmax)
        else:
            if contour:
                if cints is None:
                    pcm = ax.contourf(lon,lat,plotvar,cmap=cmap)
                else:
                    pcm = ax.contourf(lon,lat,plotvar,cmap=cmap,levels=cints)
            else:
                pcm = ax.pcolormesh(lon,lat,plotvar,cmap=cmap)
        if cints is None:
            fig.colorbar(pcm,ax=ax)
        
        #ax.set_title("Month %i"%(im+1))
        ax = label_sp("Mon%02i" % (im+1),ax=ax,
                      usenumber=True,labelstyle="%s",alpha=0.80,fontsize=14)
        
    if cints is not None:
        fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)
        
    return ax

def hcbar(mpl_obj,ax=None,fig=None,fraction=0.035,pad=.01):
    """
    Make quick horizontal colorbar. Arguments are same as ax.colorbar()
    
    """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    cb = plt.colorbar(mpl_obj,ax=ax,fraction=fraction,
                      pad=pad,orientation='horizontal',)
    return cb
    
#%% Spectral Analysis


#%% Quick Spectra Analysis Plots









# def plotmap(ax=None,bbox=None,proj=None,clon=0,figsize=(12,8),land_color=None,blabels=[1,0,0,1]):
#     """
#     Usage: fig,ax = plotmap(ax=None,bbox=None,proj=None,clon=0,figsize=(12,8),land_color=None,blabels=[1,0,0,1])
#     if ax is provided: ax = plotmap(ax=ax,bbox=None,proj=None,clon=0,figsize=(12,8),land_color=None,blabels=[1,0,0,1])
    
#     Initialize a figure or axes with coastlines and gridlines.

#     Parameters (All Arguments are Optional!)
#     ----------
#     ax : Cartopy GeoAxes
#         Axes to plot on. The default is to create both figure and axes within the function.
#     bbox : LIST [west_bound,east_bound,south_bound,north_bound]
#         Geographic bounding box/extent of the plot. First two elements are longitude bounds, last two
#         are latitude bounds. The default is Global ([-180,180,-90,90]).
#     proj : crs.projection
#         The spatial projection. The default is ccrs.PlateCarree().
#     clon : NUMERIC
#         Central Longitude for the projection. The default is 0.
#     figsize : LIST(Width,Height)
#         Figure width and height in inches. The default is (12,8).
#     land_color : STR,
#         Color of the continents. The default is None.
#     blabels : LIST of BOOL[Left, Right, Upper, Lower]
#         Set to 1 to label the axis (for PlateCarree()). The default is [1,0,0,1].
    
#     Returns
#     -------
#     if ax is None:
#         returns fig,ax
#     if ax is provided
#         returns ax
        
#     Dependencies
#         import cartopy.crs as ccrs
#         import cartopy.feature as cfeature
#         from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    
#     """
#     # Set up projection
#     if proj is None:
#         proj = ccrs.PlateCarree(central_longitude=clon)

#     # Initialize Figure
#     init_fig=False
#     if ax is None: # If no axis is provided, intialize figure
#         init_fig=True
#         fig = plt.figure(figsize=figsize)
#         ax  = plt.axes(projection=proj)

#     # Set plot extent
#     if bbox is not None:
# #         ax.set_extent([-180,180,-90,90],crs=ccrs.PlateCarree()) # Set the specified extent with bbox
# #     else:
#         ax.set_extent(bbox,crs=ccrs.PlateCarree()) # Set the specified extent with bbox

#     # Add coastlines, continents
#     ax.coastlines()
#     if land_color:
#         ax.add_feature(cfeature.LAND,facecolor=land_color)

#     # Add gridlines
#     gl = ax.gridlines(draw_labels=False,
#                   linewidth=2, color='gray', alpha=1, linestyle="dotted",lw=0.75)

#     # Remove the degree symbol
#     gl.xformatter = LongitudeFormatter(direction_label=False,degree_symbol='')
#     gl.yformatter = LatitudeFormatter(direction_label=False,degree_symbol='')
    
#     # Turn off labels according to blabels
#     gl.left_labels   = blabels[0]
#     gl.right_labels  = blabels[1]
#     gl.top_labels    = blabels[2]
#     gl.bottom_labels = blabels[3]    
    
#     if init_fig:
#         return fig,ax
#     else:
#         return ax







    
    
    
# %% Exploratory plots (qv module?)



#%%










    
