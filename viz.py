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
from matplotlib.ticker import LogLocator



#%% Functions

def return_mon_label(m,nletters='all'):
    """
    Return month labels for a given month m
    
    inputs
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


def init_map(bbox,crs=ccrs.PlateCarree(),ax=None):
    """
    Quickly initialize a map for plotting
    """
    # Create Figure/axes
    #fig = plt.gcf() 
    
    #ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.gca()
    #ax = plt.axes(projection=ccrs.PlateCarree())
        
    
    ax.set_extent(bbox,crs)
    
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE)
    #ax.add_feature(cfeature.LAND,facecolor='k',zorder=-1)
    
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.top_labels = gl.right_labels = False
    

    
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    
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
    
    tper = np.arange(0,var.shape[0]) # Preallocate time array
    nens = var.shape[1]
    
    tmax = np.around(np.nanmax(np.abs(var)))
    
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

def viz_kprev(h,kprev,locstring="",ax=None):
    
    """
    Quick visualization of mixed layer cycle (h)
    and the corresponding detrainment index found using the
    kprev function in scm (or prep_mld scripts)
    
    Inputs:
        1) h - MLD cycle (array of size 12)
        2) kprev - Detraining months (array of size 12)
        3) string indicate location (Lon/Lat)
    
    """
    if ax is None:
        ax = plt.gca()
        
    # Create Connector lines ([entrainmon,detrainmon],[MLD,MLD])
    connex = [((im+1,kprev[im]),(h[im],h[im])) for im in range(12) if kprev[im] != 0]
    
    # Indicate entraining months
    foundmon = kprev[kprev!=0]
    foundmld = h[kprev!=0]
    
    # Append month to the end
    plotmon = np.arange(1,14,1)
    plotmld = np.concatenate((h,[h[0]]))
    
    # Start Plot
    #fig,ax = plt.subplots(1,1,figsize=(6,4))
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
    ax.grid(True,linestyle='dotted')
    
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

def init_acplot(kmonth,xticks,lags,ax=None,title=None,loopvar=None):
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
    
    Returns
    -------
    ax,ax2, and ax3 if loopvar is not None : matplotlib object
        Axis with plot

    """
    if ax is None:
        ax = plt.gca()
    # Tile Months for plotting
    mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
    mons3tile = np.tile(np.array(mons3),int(np.floor(len(lags)/12))) 
    mons3tile = np.concatenate([np.roll(mons3tile,-kmonth),[mons3[kmonth]]])
    
    # Set up second axis
    ax2 = ax.twiny()
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(mons3tile[xticks], rotation = 45)
    ax2.set_axisbelow(True)
    ax2.grid(zorder=0,alpha=0)
    ax2.set_xlim(xticks[[0,-1]])
    
    # Plot second variable if option is set
    if loopvar is not None:
        ax3 = ax.twinx()
        loopvar = proc.tilebylag(kmonth,loopvar,lags)
        ax3.plot(lags,loopvar,color='gray',linestyle='dashed')
        ax3.tick_params(axis='y',labelcolor='gray')
        ax3.grid(False)
    
    ax.set_xticks(xticks)
    ax.set_xlim([xticks[0],xticks[-1]])
    if title is None:
        ax.set_title("SST Autocorrelation, Lag 0 = %s" % (mons3[kmonth]))
    else:
        ax.set_title(title)
    ax.set_xlabel("Lags (Months)")
    ax.set_ylabel("Correlation")
    ax.grid(True,linestyle='dotted')
    plt.tight_layout()
    
    if loopvar is not None:
        return ax,ax2,ax3
    return ax,ax2

def add_coast_grid(ax,bbox=[-180,180,-90,90],proj=None,blabels=[1,0,0,1]):
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
    blabels : ARRAY of BOOL [Left, Right, Upper, Lower]
        Lat/Lon Labels. Default is [1,0,0,1]

    Returns
    -------
    ax : matplotlib geoaxes
        Axes with setup
    """
    if proj is None:
        proj = ccrs.PlateCarree()
    ax.add_feature(cfeature.COASTLINE,color='black',lw=0.75)
    ax.set_extent(bbox)
    gl = ax.gridlines(crs=proj, draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle="dotted",lw=0.75)
    gl.left_labels = blabels[0]
    gl.right_labels = blabels[1]
    gl.top_labels   = blabels[2]
    gl.bottom_labels = blabels[3]
    return ax


#%% Spectral Analysis
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


def twin_freqaxis(ax,freq,tunit,dt,fontsize=12,mode='log-lin',xtick=None):
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
        
    elif mode == 'lin-lin': # Linear (x,y)
        # Note this is probably not even needed...
        htax.set_xscale("linear")
        htax.set_yscale("linear")
        
    elif mode == 'log-log':
        htax.set_xscale("log")
        htax.set_yscale("log")
        
    ax.grid(True,ls='dotted')
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

