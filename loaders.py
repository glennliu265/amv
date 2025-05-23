#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Loaders

Scripts to load particular model output. Mostly works with data paths
on stormtrack or Astraeus.

Functions:
    
>>> CESM1 Specific <<<<
    get_scenario_str    : Get CESM1 scenario names
    load_rcp85          : Load CESM1 rcp85 DataArray for 1 ens. member, combining separate files
    load_htr            : Load CESM1 historical DataArray for 1 ens. member, crop to 1920-onwards
    
    load_bsf            : Load Mean Barotropic Streamfunction
    load_monmean        : Load monthly mean pattern of variables
    load_current        : Load Mean Gulf Stream position
    
>>> Other Models <<<
    get_lens_nc         : Get netcdf list for different model LENS (historical and rpc85)
    
>>> Stochastic Model <<< from /reemergence/ stochastic model inputs/outputs
    load_smoutpur       : Load output from the stochastic model
    load_rei            : Load Re-emergence Index (REI), and max/min seasonal correlations
    load_mask           : Load Land-Ice Mask 

Created on Fri Jan 20 14:24:31 2023
@author: gliu

"""

import glob
import xarray as xr
from tqdm import tqdm
import numpy as np

def get_mnum():
    return np.concatenate([np.arange(1,36),np.arange(101,108)])

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
def load_rcp85(vname,N,datpath=None,atm=True,return_da=True):
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
        return_da : BOOL
            True to return DataArray (False for Dataset)
    Returns
    -------
        ds[vname] : xr.DataArray
            DataArray containing the variable, concatenated for the whole RCP85 period.
    
    Taken from preproc_CESM1_LENS.py on 2023.01.24
    
    """
    if atm:
        model_string = "cam.h0"
        comp         = "atm"
    else:
        model_string = "pop.h"
        comp         = "ocn"
    if datpath is None:
        datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/%s/proc/tseries/monthly/" % comp
        
    # Append variable name to path
    vdatpath = "%s%s/" % (datpath,vname)
    

        
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
    if return_da:
        return ds[vname]
    else:
        return ds

def load_htr(vname,N,datpath=None,atm=True,return_da=True):
    
    """
    Load a given variable for an ensemble member for the historical period.
    **Accounts for different length of ensemble member 1 by cropping to 1920 onwards...
    
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
        return_da : BOOL
            True to return DataArray (False for Dataset)
    Returns
    -------
        ds[vname] : xr.DataArray
            DataArray containing the variable.
    Taken from preproc_CESM1_LENS.py on 2023.01.24
    
    """
    if atm:
        model_string = "cam.h0"
        comp         = "atm"
    else:
        model_string = "pop.h"
        comp         = "ocn"
        
    if datpath is None: # Stormtrack path (need to generalize this)
        datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/%s/proc/tseries/monthly/" % comp
        
        # Change Datpath for Salt
        if vname == "SALT":
            datpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/"
        
        # Change Datpath for HMXL (Ens Member 41 and 42)
        if (N > 105) and vname == "HMXL":
            datpath = "/stormtrack/data4/glliu/01_Data/CESM1_LE/" # Ens 41 and 42 for HMXL downloaded separately...
    
    # Append variable name to path
    vdatpath = "%s%s/" % (datpath,vname)
    
    # Ensemble 1 has a different time
    if N == 1:
        fn = "%sb.e11.B20TRC5CNBDRD.f09_g16.%03i.%s.%s.185001-200512.nc" % (vdatpath,N,model_string,vname)
    else:
        fn = "%sb.e11.B20TRC5CNBDRD.f09_g16.%03i.%s.%s.192001-200512.nc" % (vdatpath,N,model_string,vname)
    ds = xr.open_dataset(fn)
    if N == 1:
        ds = ds.sel(time=slice("1920-02-01","2006-01-01"))
    
    if return_da:
        return ds[vname]
    else:
        return ds

def load_pic(vname,datpath=None,atm=True):
    
    if atm:
        model_string = "cam.h0"
        comp         = "atm"
    else:
        model_string = "pop.h"
        comp         = "ocn"
    
    if datpath is None:
        datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/%s/proc/tseries/monthly/" % comp
    
    # b.e11.B1850C5CN.f09_g16.005.cam.h0.LHFLX.040001-049912.nc
    nclist = glob.glob("%s/%s/b.e11.B1850C5CN.f09_g16.005.%s.%s.*.nc" % (datpath,vname,model_string,vname))
    nfiles = len(nclist)
    nclist.sort()
    print("Found %i files!" % (nfiles))
    if nfiles != 18:
        print("WARNING: did not find all files")
    dslist = []
    for f in range(nfiles):
        ds = xr.open_dataset(nclist[f])
        dslist.append(ds)
    return dslist
    

    
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
        if mconfig in ['rcp85',"RCP85"]:
            ds = load_rcp85(vname,N,datpath=datpath)
        elif mconfig in ['htr',"HTR"]:
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

def load_bsf(datpath=None,stormtrack=0,ensavg=True,ssh=False):
    
    # Load mean BSF (or SSH) [ens x mon x lat x lon180]
    # As processed by calc_mean_bsf.py
    
    vname = "BSF"
    if ssh:
        vname = "SSH" # Load SSH instead
    
    if datpath is None:
        if stormtrack == 0: # Assumed to be Astraeus
            datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/CESM1/NATL_proc/"
        else: # stormtrack path
            datpath = "/home/glliu/01_Data/"
    if ensavg:
        nc = "%sCESM1_HTR_%s_bilinear_regridded_EnsAvg.nc" % (datpath,vname)
    else:
        nc = "%sCESM1_HTR_%s_bilinear_regridded_AllEns.nc" % (datpath,vname)
        
    ds = xr.open_dataset(nc).load()
    return ds

def load_current(datpath=None,stormtrack=0,z=0,regrid=False,mldavg=False):
    # Lead monthly mean UVEL and VVEL, regridded to CAM5
    # z is the index of the depth. Loads surface values by default
    # Note mldavg not implemented yet... need to take seasonal average and add
    
    if datpath is None:
        datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/proc/NATL/"
    else:
        print("WARNING this is currently not supported on alternate datpaths")
        return None
        #datpath = "/home/glliu/01_Data/"
    if regrid:
        ds_uvel = xr.open_dataset(datpath + "CESM1_HTR_UVEL_NATL_scycle_regrid_bilinear.nc").isel(z_t=z).load()
        ds_vvel = xr.open_dataset(datpath + "CESM1_HTR_VVEL_NATL_scycle_regrid_bilinear.nc").isel(z_t=z).load()
    else:
        ds_uvel = xr.open_dataset(datpath + "CESM1_HTR_UVEL_NATL_scycle.nc").isel(z_t=z).load()
        ds_vvel = xr.open_dataset(datpath + "CESM1_HTR_VVEL_NATL_scycle.nc").isel(z_t=z).load()
    return ds_uvel,ds_vvel

def load_monmean(vname,datpath=None):
    
    # Loads monthly mean output [ens x mon x lat x lon] from the [calc_monmean_CESM1.py] script
    if datpath is None:
        datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/proc/NATL/"
    else:
        print("WARNING this is currently not supported on alternate datpaths")
        return None
    
    ncname = "CESM1_HTR_%s_NATL_scycle.nc" % (vname)
    ds     = xr.open_dataset(datpath+ncname).load()
    
    return ds

def load_gs(datpath=None,load_u2=False):
    # Loads lat/lon for gulf stream computed using [calc_gulfstream_position.py]
    # Using the maximum sea-level anomaly standard deviation
    # based on a function written by Lilli Enders
    if datpath is None:
        datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LE/proc/NATL/" # Uses Astraeus datpath
    if load_u2:# Load output from compute_gs_uvel.py
        ncname = "GSI_Location_CESM1_HTR_MaxU2Mag.nc"
    else:
        ncname = "GSI_Location_CESM1_HTR_MaxStdev.nc"
    ds = xr.open_dataset(datpath+ncname)
    return ds

#%%

def load_smoutput(expname,output_path,debug=True,return_nclist=False,runids=None,load=True):
    # Load output from [run_SSS_basinwide.py]
    # Copied from [pointwise_crosscorrelation]
    
    # Load NC Files
    expdir       = output_path + expname + "/Output/"
    nclist       = glob.glob(expdir +"*.nc")
    nclist.sort()
    
    if runids is not None:
        nclist = np.array(nclist)[runids]
    
    if debug:
        print(nclist)
    
    nclist = [nc for nc in nclist if "ravgparams" not in nc]
    if return_nclist: # Just return the nclist
        return nclist
    
    # Load DS, deseason and detrend to be sure
    if len(nclist) == 1:
        print("Only found 1 file")
        ds_all = xr.open_dataset(nclist[0])#.load()
    else:
        ds_all   = xr.open_mfdataset(nclist,concat_dim="run",combine='nested')#.load()
    
    if load:
        return ds_all.load()
    return ds_all

def load_rei(expname,output_path,maxmin=False):
    if maxmin:
        ncname = output_path + expname + "/Metrics/MaxMin_Pointwise.nc"
    else:
        ncname = output_path + expname + "/Metrics/REI_Pointwise.nc"
    return xr.open_dataset(ncname)

def load_mask(expname,maskpath=None):
    """
    Load Land Ice Mask.

    Parameters
    ----------
    expname : str
        Name of the dataset/model experiment.
        Currently available: ERA5,CESM1 HTR, cesm2 pic
    maskpath : str, optional
        Path to mask location. The default is the path on Astraeus (to rememergence model inputs).

    Returns
    -------
    mask : xr.DataArray
        Land Ice Mask.

    """
    # Set Path to masks (based on Astraeus)
    if maskpath is None:
        maskpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/model_input/masks/"
    
    if "ERA5" in expname:
        print("Loading ERA5 Land Ice Mask, 1979-2024")
        ncname = "ERA5_1979_2024_limask_0.05p.nc"
    elif "CESM1" in expname and "HTR" in expname:
        print("Loading CESM1-LE Historical Land Ice Mask, 1920-2005")
        ncname = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"
    elif "cesm2" in expname and "pic" in expname:
        print("Loading CESM2-PiControl Land Ice Mask, 200-2000")
        ncname = "cesm2_pic_limask_0.3p_0.05p_0200to2000.nc"
    
    return xr.open_dataset(maskpath + ncname).load()

    

# Get mean SST.SSS gradient

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