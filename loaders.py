#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Loaders

Scripts to load particular model output. Mostly works with data paths
on stormtrack.

Functions:
    
>>> CESM1 Specific <<<<
    get_scenario_str : Get CESM1 scenario names
    load_rcp85       : Load CESM1 rcp85 DataArray for 1 ens. member, combining separate files
    load_htr         : Load CESM1 historical DataArray for 1 ens. member, crop to 1920-onwards
    
>>> Other Models <<<
    get_lens_nc      : Get netcdf list for different model LENS (historical and rpc85)

Created on Fri Jan 20 14:24:31 2023
@author: gliu
"""

import glob
import xarray as xr
from tqdm import tqdm
import numpy as np

def get_scenario_str(scenario):
    """
    Get CESM 1 Scenario string where:
        HTR      : historical   (1850-2005), "B[20TR]C5CNBDRD"
        RCP85    : rcp 8.5      (2006-2100), "B[RCP85]C5CNBDRD"
        PIC      : pre-industrial control  , "B[1850]C5CN"
    """
    if scenario == "HTR":
        out_str = "20TR"
    elif scenario == "RCP85":
        out_str = "RCP85"
    elif scenario == "PIC":
        out_str = "1850"
    return out_str

# RCP85 Loader
def load_rcp85(vname,N,datpath=None,atm=True):
    """
    Load a given variable for an ensemble member for rcp85.
    Concatenates the two files for N<34.
    
    Parameters
    ----------
        vname     : STR
            CESM Variable Name.
        N         : INT
            Ensemble member number for loading
        datpath   : STR
            Location to search for the file
        atm       : BOOL
            True to load atmospheric data (default)
    Returns
    -------
        ds[vname] : xr.DataArray
            DataArray containing the variable, concatenated for the whole RCP85 period.
    
    Taken from preproc_CESM1_LENS.py on 2023.01.24
    
    """
    if datpath is None:
        datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/"
        
    # Append variable name to path
    vdatpath = "%s%s/" % (datpath,vname)
    
    if atm:
        model_string = "cam.h0"
    else:
        model_string = "pop.h"
        
    # Files are split into 2
    if N<34:
        fn1 = "b.e11.BRCP85C5CNBDRD.f09_g16.%03i.%s.%s.200601-208012.nc" % (N,model_string,vname)
        fn2 = "b.e11.BRCP85C5CNBDRD.f09_g16.%03i.%s.%s.208101-210012.nc" % (N,model_string,vname)
        ds = []
        for fn in [fn1,fn2]:
            dsf = xr.open_dataset(vdatpath + fn)
            ds.append(dsf)
        ds = xr.concat(ds,dim='time')
    else:
        fn1 = "%sb.e11.BRCP85C5CNBDRD.f09_g16.%03i.cam.h0.%s.200601-210012.nc" % (vdatpath,N,vname)
        ds = xr.open_dataset(fn1)
    return ds[vname]

def load_htr(vname,N,datpath=None,atm=True):
    """
    Load a given variable for an ensemble member for the historical period.
    Accounts for different length of ensemble member 1 by cropping to 1920 onwards...
    
    Parameters
    ----------
        vname     : STR
            CESM Variable Name.
        N         : INT
            Ensemble member number for loading
        datpath   : STR
            Location to search for the file
        atm       : BOOL
            True to load atmospheric data ()
    Returns
    -------
        ds[vname] : xr.DataArray
            DataArray containing the variable.
    Taken from preproc_CESM1_LENS.py on 2023.01.24
    
    """
    if datpath is None:
        datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/"
    
    # Append variable name to path
    vdatpath = "%s%s/" % (datpath,vname)
    
    if atm:
        model_string = "cam.h0"
    else:
        model_string = "pop.h"
    
    # Ensemble 1 has a different time
    if N == 1:
        fn = "%sb.e11.B20TRC5CNBDRD.f09_g16.%03i.%s.%s.185001-200512.nc" % (vdatpath,N,model_string,vname)
    else:
        fn = "%sb.e11.B20TRC5CNBDRD.f09_g16.%03i.%s.%s.192001-200512.nc" % (vdatpath,N,model_string,vname)
    ds = xr.open_dataset(fn)
    if N == 1:
        ds = ds.sel(time=slice("1920-02-01","2006-01-01"))
    return ds[vname]
    
def load_atmvar(vname,mnum,mconfig,datpath,preproc=None,return_ds=False): 
    """
    Load all [mnum] ensemble members for variable vname, applying preprocessing function [preproc].
    
    Parameters
    ----------
        vname     : STR
            CESM Variable Name.
        mnum      : LIST
            Ensemble member numbers for loading
        mconfig   : STR
            Scenario (rcp85 or htr)
        datpath   : STR
            Location to search for the file
        preproc   : function
            Preprocessing function to apply to xr.dataset
        return_ds : BOOL
            Set to True to return dataset
    Returns
    -------
        ds[vname] : xr.DataArray
            DataArray containing the variable.
            
    Taken from merge_cesm1_atm.py
    
    """
    nens = len(mnum)
    var_allens = []
    for e in tqdm(range(nens)):
        N = mnum[e]
        if mconfig =='rcp85':
            ds = load_rcp85(vname,N,datpath=datpath)
        elif mconfig == 'htr':
            ds = load_htr(vname,N,datpath=datpath)
            
        # Apply Preprocessing
        if preproc is not None:
            ds = preproc(ds)
        
        if return_ds: # Just return dataset
            var_allens.append(ds)
        else: # Load out values
            invar = ds.values # [Time x Lat x Lon]
            if e == 0:
                ntime,nlat,nlon = invar.shape
                var_allens = np.zeros((nens,ntime,nlat,nlon))
                times = ds.time.values
                lon   = ds.lon.values
                lat   = ds.lat.values
            var_allens[e,...] = invar.copy()
    if return_ds: # Just return dataset (has all variables within)
        return var_allens
    else:
        return var_allens,times,lat,lon

#%%

def get_lens_nc(modelname,vname,e,compname="Amon"):
    """
    Get the searchstring for a given large ensemble dataset.
    Taken from preproc_damping_lens.npy on 2023.01.23

    Parameters
    ----------
    modelname : STR
        Name of the model. Supports: ("gfdl_esm2m_lens","csiro_mk36_lens","canesm2_lens") 
    vname : STR
        Name of the variable.
    e : INT
        Ensemble Index (e=0 is ensemble member 1).
    compname : STR, optional
        Component Name. The default is "Amon".

    Returns
    -------
    ncname : STR
        Name of the netcdf file.

    """
    if modelname == "gfdl_esm2m_lens":
        ncname = "%s_%s_GFDL-ESM2M_historical_rcp85_r%ii1p1_195001-210012.nc" % (vname,compname,e+1)
    elif modelname == "csiro_mk36_lens":
        ncname = "%s_%s_CSIRO-Mk3-6-0_historical_rcp85_r%ii1p1_185001-210012.nc" % (vname,compname,e+1)
    elif modelname == "canesm2_lens":
        ncname = "%s_%s_CanESM2_historical_rcp85_r%ii1p1_195001-210012.nc" % (vname,compname,e+1)
    elif modelname == "mpi_lens":
        if vname == "sic": # sic files are split into htr, rcp85
            ncname1 = "%s_%s_MPI-ESM_historical_r%03ii1850p3_185001-200512.nc" % (vname,compname,e+1)
            ncname2 = "%s_%s_MPI-ESM_rcp85_r%03ii2005p3_200601-209912.nc" % (vname,compname,e+1)
            ncname  = [ncname1,ncname2]
        else:
            ncname = "%s_%s_MPI-ESM_historical_rcp85_r%ii1p1_185001-209912.nc" % (vname,compname,e+1)
    return ncname

# def get_cesm1_ocn_nclist(varname,scenario="HTR",path=None):
    
#     """
#     varname  : [STR}Name of variable in CESM
#     scenario :
    
#     """
    
#     # Set data path
#     if path is None:
#         if varname == "TEMP":
#             datpath = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/TEMP/"
#         else:
#             datpath = ""
    
#     if scenario == "HTR":
#         # b.e11.B20TRC5CNBDRD.f09_g16.001.pop.h.TEMP.185001-200512.nc
#         search_string = "b.e11.*.f09_g16.*.pop.h.%s.*.nc" % (varname)

# def load_pop_3d()