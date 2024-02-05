#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Wed Dec 14 10:19:48 2022

@author: alexandre

Listed most useful functions to compute DYCOMED database

"""


import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
import datetime as dt
#import imageio
from astropy.convolution import convolve, Gaussian2DKernel
import scipy.ndimage as ndimage
from scipy.stats import kde
import scipy.interpolate as interp
import matplotlib.path as mpp
from tqdm import tqdm
import cartopy.crs as ccrs
import geopandas as gpd
import cartopy.feature as cf
# %%

def distance(x0,x1,y0,y1):
    """
    distance
    Compute shortest distance on a sphere between (x0,y0) and (x1,y1), y as latitude
    """
    return np.sqrt((np.cos(np.deg2rad(y0))*(x0-x1))**2 + (y0-y1)**2)*40219/360


def unique_prof(time, x, y):
    """
    unique_prof
    return an integer as unique identifier of a given profile as function of its 2D position + time
    Intended to count doubles, but not 100% efficient given errors in data
    """
    Unic=(time.astype(int)*1e9 + (x*100).astype(int)*100000 + (y*100).astype(int)).astype(int)
    return Unic

def juld2str(juld, format='%Y%m%d'):
    return dt.datetime.strftime(dt.date(2000,1,1)+dt.timedelta(days=int(juld)),format=format)

def juldHour2str(juld, format='%Y%m%dH%H%M'):
    return dt.datetime.strftime(dt.datetime(2000,1,1,0,0)+dt.timedelta(days=int(juld),seconds=juld%1*86400), format=format)

def str2juld(s):
    return (dt.date(int(s[0:4]),int(s[4:6]),int(s[6:8]))-dt.date(2000,1,1)).days

# %%
def fill_gap(A):
    """
    fill_gap

    Very useful function taking as input a 1D array with potential NaN
    All gaps inbetween are filled with the MEAN of non-NaN bounds
    Upper and lower edges are assumed to be constant to first/last values of input vector

    """
    if np.all(np.isnan(A)):
        return np.array([np.nan]*len(A))
    else:
        B=np.zeros(len(A))
        i=0
        while i <len(A):
            if ~np.isnan(A[i]):
                B[i]=A[i]
                i+=1
            else:
                if i==0:   ### If gap in the upper part, assumes it's contant
                    step=np.where(~np.isnan(A[i:]))[0][0]
                    B[i:i+step]=A[i+step]
                    i+=step
                else:     ### If gap in the middle, fill it with mean value between non-nan bounds
                    if len(np.where(~np.isnan(A[i:]))[0])>0:
                        step=np.where(~np.isnan(A[i:]))[0][0]
                        B[i:i+step]=(A[i-1]+A[i+step])/2
                        i+=step
                    else:  ### If gap in the upper part, assumes it's contant
                        B[i:]=A[i-1]
                        i+=len(A)
        return B


# %% Defining coloc function

def link_eddies_end(x_cast, y_cast, time_cast, x_cen, y_cen, x_max, y_max, x_end, y_end, time_dyned, search_radius=120):
    """
    link_eddies_end  
    
    Attributes colocalization between profiles at given position and time, and the DYNED observations.
    This function looks for potential eddy centers ON THE SAME DAY, in a perimeter closer than search_radius
    By default search_radius =120 km, but to be adjusted with the ocean region
    Colocalization are flagged as 1 if profiles is between Rend and Rmax contours, flagged as 2 if indeed inside Ramx contour
    
    Change from previous function 'link_eddies' : two levels of eddy tags in 'flageddy', within last closed SSH contour and max speed radius differenciated
     
    Warning : - flageddy does not take into account colocalized eddy polarity, then it is different from "eddy tag" in DYCOMED (positive tags => AE / negative => CE)
              - Latest version of DYCOMED also encompasses detection at +/- several days and takes a 3rd flag "ambiguous"
    
    INPUT
    x/y/time_cast      array size P            2D postion and time of cast profile
    x/y_cen            array size Nobs         2D position of eddy centers from DYNED
    time_dyned         array size Nobs         observation time array in DYNED
    x/y_max            array size (Nobs x 50)  coordinates of Rmax contours observations
    x/y_end            array size (Nobs x 50)  coordinates of Rend contours observations
    
    OPTIONAL
    search_radius=120   int        maximal distance in km to search for colocalized eddy center
    
    OUTPUT
    ctd2obs     int array size P     index of colocalized observation with profiles, if any. If not colocalized, =-1
    disteddy    float array size P   distance to colocalized eddy with profiles, if any. If not colocalized, =-1
    flageddy    int array size P     Type of colocalization. If not colocalized, =0, if between Rmax and Rend =1, if inside Rmax=2
    """

    day_cast=time_cast - time_cast%1
    day_cast[(time_cast%1)>0.5]+=1
    day_cast=day_cast.astype(int)
    ctd2obs=-1*np.ones(len(day_cast))
    disteddy=-1*np.ones(len(day_cast))
    flageddy=np.zeros(len(day_cast))
    for i in tqdm(range(len(day_cast))):
        I=np.where(time_dyned==day_cast[i])[0]
        D=distance(x_cast[i], x_cen[I], y_cast[i], y_cen[I])
        x_contour=x_max[I][D<search_radius] ; y_contour=y_max[I][D<search_radius]
        x_cont_end=x_end[I][D<search_radius+30] ; y_cont_end=y_end[I][D<search_radius+30]
        for j in range(np.shape(x_cont_end)[0]):
            path_eddy=mpp.Path(np.array([x_cont_end[j],y_cont_end[j]]).T, closed=True)
            if path_eddy.contains_point([x_cast[i],y_cast[i]]):
                ctd2obs[i]=I[D<search_radius+30][j]
                disteddy[i]=D[D<search_radius+30][j]
                flageddy[i]=1
        for j in range(np.shape(x_contour)[0]):
            path_eddy=mpp.Path(np.array([x_contour[j],y_contour[j]]).T, closed=True)
            if path_eddy.contains_point([x_cast[i],y_cast[i]]):
                flageddy[i]=2
    return ctd2obs.astype(int), disteddy, flageddy

def newlink(x_common, y_common, day_common, x_cen, y_cen, x_max, y_max, x_end, y_end, time_dyned, delay=2, search_radius=150):
    """
    newlink  
    
    Attributes colocalization between profiles at given position and time, and the DYNED observations.
    This function looks for potential eddy centers ON THE SAME DAY, in a perimeter closer than search_radius
    By default search_radius =120 km, but to be adjusted with the ocean region
    Colocalization are flagged as 1 if profiles is between Rend and Rmax contours, flagged as 2 if indeed inside Ramx contour
    
    Change from previous function 'link_eddies' : two levels of eddy tags in 'flageddy', within last closed SSH contour and max speed radius differenciated
     
    Warning : - flageddy does not take into account colocalized eddy polarity, then it is different from "eddy tag" in DYCOMED (positive tags => AE / negative => CE)
              - Latest version of DYCOMED also encompasses detection at +/- several days and takes a 3rd flag "ambiguous"
    
    INPUT
    x/y/time_cast      array size P            2D postion and time of cast profile
    x/y_cen            array size Nobs         2D position of eddy centers from DYNED
    time_dyned         array size Nobs         observation time array in DYNED
    x/y_max            array size (Nobs x 50)  coordinates of Rmax contours observations
    x/y_end            array size (Nobs x 50)  coordinates of Rend contours observations
    
    OPTIONAL
    search_radius=120   int        maximal distance in km to search for colocalized eddy center
    delay =2            int        time at +/- n days at which profiles are colocated
    
    OUTPUT
    ctd2obs     int array size P     index of colocalized observation with profiles, if any. If not colocalized, =-1
    disteddy    float array size P   distance to colocalized eddy with profiles, if any. If not colocalized, =-1
    flageddy    int array size P     Type of colocalization. If not colocalized, =0, if between Rmax and Rend =1, if inside Rmax=2
    """
    new2obs=-1*np.ones((len(x_common),2*delay+1))
    newdist=-1*np.ones((len(x_common),2*delay+1))
    newflag=np.zeros((len(x_common),2*delay+1))
    
    for i in tqdm(range(len(x_common))):
        for k in np.arange(-delay,delay+1):
            I=np.where(time_dyned==day_common[i]+k)[0]
            D=distance(x_common[i], x_cen[I], y_common[i], y_cen[I])
            
            x_contour=x_max[I][D<search_radius] ; y_contour=y_max[I][D<search_radius]
            x_cont_end=x_end[I][D<search_radius+30] ; y_cont_end=y_end[I][D<search_radius+30]
            
            for j in range(np.shape(x_cont_end)[0]):
                path_eddy=mpp.Path(np.array([x_cont_end[j],y_cont_end[j]]).T, closed=True)
                if path_eddy.contains_point([x_common[i],y_common[i]]):
                    new2obs[i,k+delay]=I[D<search_radius+30][j]
                    newdist[i,k+delay]=D[D<search_radius+30][j]
                    newflag[i,k+delay]=1
            for j in range(np.shape(x_contour)[0]):
                path_eddy=mpp.Path(np.array([x_contour[j],y_contour[j]]).T, closed=True)
                if path_eddy.contains_point([x_common[i],y_common[i]]):
                    newflag[i,k+delay]=2
                    
    newdist[newdist==-1]=np.nan
    new2obs=new2obs.astype(int)
    newflag=newflag.astype(int)
    return new2obs,newdist, newflag

# %% MLD functions

def MLD_threshold(D, depth, delta=0.03, surf_accept=10, only_dens=False):
    """
    function MLD_threshold
    
    Compute mixed layer depth on a threshold method. A minimal number of non-Nan values in the upper column is require by 'surf_accept'
    delta is the threshold, common values are 0.03 for potential density profiles (in kg/m3) or 0.1 for temperature profiles (in degC)
    (see Houpert et al 2014, De Boyer-Montegut et al 2004)
    Return a nan if not a single non-nan value in the upper part.
    surface value is assumed to be first index in D, ie reshape D to compute difference from 10meter value
    
    INPUT
    D     size N     Measurement profile
    depth size N     depth profile
    delta   float    threshold value
    surf_accept  int  maximal first non-nan value
    only_dens  bool   Key for only density profile, useful in case of static instability.
    
    OUTPUT 
    mld float
    
    """
    if ~np.all(np.isnan(D)):
        I10=np.argmin(np.isnan(D))
        if I10<surf_accept:
            if only_dens:
                rho_anom=D-D[I10]
            else:
                rho_anom=np.abs(D-D[I10])
            ttt=np.where(rho_anom>delta)[0]
            if len(ttt)>0:
                mld=depth[ttt[0]-1]
            else:
                mld=np.nan
        else:
            mld=np.nan
    else:
        mld=np.nan
    return mld

def MLD_threshold_V2(profil, depth, delta, grad_delta, type_profil,surf_accept):
    
    """
    
    Parameters
    ----------
    profil : 1D array of floats
        Temperature or Density profile
    depth : 1D array of floats
        Depth profile
    delta : float
        Temperature or density threshold 
    grad_delta : float
        Temperature or density gradient threshold 
    type_profil : str
        'temp' or 'dens'
    surf_accept : int
        Maximum index of the first non NaN value in profil, needed to define
        MLD
        
    Returns
    -------
    mld : float
        Mixed Layer Depth estimate
        
    """
    if ~np.all(np.isnan(profil)):
        istart=np.argmin(np.isnan(profil))
        iend = len(profil) - np.argmin(np.isnan(np.flip(profil)))-1
        if istart < surf_accept:
            depth=depth[istart:iend]
            profil=profil[istart:iend]
            surf_value = profil[0]
            ttt = np.where(abs(surf_value-profil)>delta)[0]
            if len(ttt)>0:
                
                #If the threshold is exceeded before reaching 25m depth,
                #keep this value as the MLD but if deeper, you want to see if 
                #you didn't miss a smaller jump before by computing the gradient
                
                if depth[ttt[0]]<25:
                    mld=depth[ttt[0]-1] 
                else:
                    first_guess=depth[ttt[0]-1]
                    id_start=np.where(depth==20)[0][0]
                    
                    #Apply a three-point moving average before computing gradient
                    #in order to avoid small vertical-scale spikes
                    
                    profil_lisse = moving_average(profil[id_start:ttt[0]+1], 3)
                    depth_lisse = depth[id_start+1:ttt[0]]
                    if len(profil_lisse)>1:
                        grad_profil = np.gradient(profil_lisse,depth_lisse)
                        
                        #If the profile is a temperature profile, i.e with decreasing
                        #values with depth, take the opposite sign for the gradient
                        
                        if type_profil == 'temp':
                            grad_profil = -grad_profil;
                        ttt_grad = np.where(grad_profil>grad_delta)[0]
                        if len(ttt_grad)>0:
                            mld=depth_lisse[ttt_grad[0]]
                        else:
                            mld=first_guess
                    else:
                        mld=first_guess
            else:
                mld=np.nan
        else:
            mld=np.nan
    else:
        mld=np.nan
    return mld 

    #Moving average (needed in the New MLD estimate function)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
# %% Profile interpolating functions

"""
interpol_profiles
Interpolate an array of profiles over a reference depth vector depth_array

New December 2021 : Check that Temp and depth_array vectors have nans at the same levels
INPUT

OUTPUT


"""

def interpol_profiles_column(depth_array, Temp, depth_ref, kind='linear', max_gap=20, min_lev_nb=10):
    if np.sum(~np.isnan(depth_array))<min_lev_nb:
        print('Problem : NaN column')
        return np.nan
    Result=np.ones(len(depth_ref))*-1

    ### Selecting depth levels covered by the profile
    index_z=[]
    for k in range(len(depth_ref)):
        h=np.nanargmin(np.abs(depth_array-depth_ref[k]))
        if (np.abs(depth_array[h]-depth_ref[k])<max_gap) & (depth_ref[k]>np.nanmin(depth_array)) & (depth_ref[k]<np.nanmax(depth_array)) :
            index_z+=[k]
    index_z=np.unique(index_z)
    local_z=depth_ref[index_z].data


    Result[index_z]=interp.interp1d(depth_array,Temp, kind=kind)(local_z)
    Result[Result==-1]=np.nan
    return Result

def interpol_profiles_column_filter(depth_prim, Temp, depth_ref, kind='linear', max_gap200=40, max_gap500=100, max_gap2000=200, min_lev_nb=10, check=True):

    Result=np.ones(len(depth_ref))*-1

    ### Checking that Nan are at same level in both vectors

    depth_array=np.copy(depth_prim)
    if check:
        depth_array[np.isnan(Temp)]=np.nan

    ### Check if enough data
    if (np.sum(~np.isnan(depth_array))<min_lev_nb) + (np.sum(~np.isnan(Temp))<min_lev_nb) + (np.nanmax(depth_array)-np.nanmin(depth_array)<50):
        print('Problem : NaN column')
        return np.nan
    
    id200=np.nanargmin(np.abs(depth_ref-200))
    id500=np.nanargmin(np.abs(depth_ref-500))
    ### Selecting depth levels covered by the profile, with varying threshold for interpolation
    index_z=[]
    for k in range(id200):
        h=np.nanargmin(np.abs(depth_array-depth_ref[k]))
        if (np.abs(depth_array[h]-depth_ref[k])<max_gap200) & (depth_ref[k]>np.nanmin(depth_array)) & (depth_ref[k]<np.nanmax(depth_array)) :
            index_z+=[k]
    for k in range(id200,id500):
        h=np.nanargmin(np.abs(depth_array-depth_ref[k]))
        if (np.abs(depth_array[h]-depth_ref[k])<max_gap500) & (depth_ref[k]>np.nanmin(depth_array)) & (depth_ref[k]<np.nanmax(depth_array)) :
            index_z+=[k]
    for k in range(id500,len(depth_ref)):
        h=np.nanargmin(np.abs(depth_array-depth_ref[k]))
        if (np.abs(depth_array[h]-depth_ref[k])<max_gap2000) & (depth_ref[k]>np.nanmin(depth_array)) & (depth_ref[k]<np.nanmax(depth_array)) :
            index_z+=[k]
    index_z=np.unique(index_z)
    
    local_z=depth_ref[index_z].data
    if len(local_z)>0:
        Result[index_z]=interp.interp1d(depth_array,Temp, kind=kind)(local_z)
        Result[Result==-1]=np.nan
    else:
        Result[:]=np.nan
    return Result


# %% [markdown]
# ### Mapping tools

# %%
def add_contour(ax, shp_filename, color='k',lw=1):
    # Read the shapefile using geopandas
    gdf = gpd.read_file(shp_filename)

    # Create a Cartopy feature from the GeoDataFrame
    shapely_feature = cf.ShapelyFeature(gdf['geometry'], ccrs.PlateCarree(), facecolor='none', linewidth=lw,edgecolor=color)

    # Add the feature to the plot
    ax.add_feature(shapely_feature)

    #shapefile_path =
    
def plot_coast(ax,color='k',lw=1):
    add_contour(ax,'/home6/datawork/abarboni/OceanData/ne_50m_land/ne_50m_land.shp',color=color,lw=lw)
def plot_eez(ax,color='k',lw=1):
    add_contour(ax, '/home6/datawork/abarboni/OceanData/eez_boundaries_v12.shp',color=color,lw=lw)
    add_contour(ax,'/home6/datawork/abarboni/OceanData/ne_10m_marine/ne_10m_admin_0_boundary_lines_maritime_indicator.shp',color=color,lw=lw)
 

def plot_borders(ax, color='k',lw=1):
    add_contour(ax,'/home6/datawork/abarboni/OceanData/ne_10m_land/ne_10m_admin_0_boundary_lines_land.shp',color=color,lw=lw)
    


# %%
def load_rossby_radius(CoorLon,CoorLat, path_rd='/home6/datawork/abarboni/OceanData/Rd_WOA.nc', convol=True, sigma=3):
    """
    Loading Rd atlas betwwen CoorLon; CoorLat at path_rd
    optional astropy convolve smoothing
    """ 
    rdmat=nc4.Dataset(path_rd)
    lat_rd=rdmat['lat'][:] ; lon_rd=rdmat['lon'][:]
    idlat = np.where((lat_rd >= CoorLat[0]) & (lat_rd <= CoorLat[1]))[0]
    idlon = np.where((lon_rd >= CoorLon[0]) & (lon_rd <= CoorLon[1]))[0]

    Rd=rdmat['Rd'][idlon,:][:,idlat]
    lat_rd=rdmat['lat'][idlat] ; lon_rd=rdmat['lon'][idlon]
    rdmat.close()
    if convol:
        kernel = Gaussian2DKernel(x_stddev=sigma)
        Rd = convolve(Rd, kernel, boundary='extend')
    return lon_rd,lat_rd,Rd


# %% [markdown]
# ### Merging Neodyn and DYNED Atlas

# %%
def merge_dyned_neodyn(path_dyned,path_neo):
    Nobs_d=1018441 ; Nobs_n=114068 #Obs number in Dyned and Neodyn
    Ned_d=13694 ; Ned_n=2334 #eddy number in Dyned and Neodyn

    x_cen=np.zeros(Nobs_d+Nobs_n) ; y_cen=np.zeros(Nobs_d+Nobs_n)
    x_max=np.zeros((Nobs_d+Nobs_n,50)) ; y_max=np.zeros((Nobs_d+Nobs_n,50))
    track=np.zeros(Nobs_d+Nobs_n)
    r_max=np.zeros(Nobs_d+Nobs_n)
    v_max=np.zeros(Nobs_d+Nobs_n)
    time=np.zeros(Nobs_d+Nobs_n)
    life_cycle=np.zeros(Nobs_d+Nobs_n)
    ro=np.zeros(Nobs_d+Nobs_n)
    dSSH=np.zeros(Nobs_d+Nobs_n)

    first_obs=np.zeros(Ned_d+Ned_n)
    last_obs=np.zeros(Ned_d+Ned_n)
    ro_mean=np.zeros(Ned_d+Ned_n)
    r_max_avg=np.zeros(Ned_d+Ned_n)
    life_time=np.zeros(Ned_d+Ned_n)
    eddy_origin=np.zeros(Ned_d+Ned_n)
    eddy_death=np.zeros(Ned_d+Ned_n)

    #### DYNED
    data=nc4.Dataset(path_dyned)
    Atlas=data['Atlas']   ### Obs variables
    x_cen[:Nobs_d]=Atlas['x_cen'][:] ;y_cen[:Nobs_d]=Atlas['y_cen'][:]
    x_max[:Nobs_d]=Atlas['x_max'][:] ;y_max[:Nobs_d]=Atlas['y_max'][:]
    track[:Nobs_d]=Atlas['track'][:]
    time[:Nobs_d]=Atlas['time'][:]
    r_max[:Nobs_d]=Atlas['r_max'][:]
    v_max[:Nobs_d]=Atlas['v_max'][:]
    life_cycle[:Nobs_d]=Atlas['life_cycle'][:]
    ro[:Nobs_d]=Atlas['Ro'][:]
    dSSH[:Nobs_d]=Atlas['dssh_max'][:]  
    first_obs[:Ned_d]=Atlas['index_of_first_observation'][:]
    last_obs[:Ned_d]=Atlas['last_obs'][:]

    ro_mean[:Ned_d]=Atlas['ro_mean'][:]   ### eddy variables
    r_max_avg[:Ned_d]=Atlas['r_max_avg'][:]
    life_time[:Ned_d]=Atlas['life_time'][:]
    eddy_origin[:Ned_d]=Atlas['eddy_origin'][:]
    eddy_death[:Ned_d]=Atlas['eddy_death'][:]
    Ned=Atlas.dimensions['eddies'].size
    data.close()

    ####  NEODYN
    Atlas=nc4.Dataset(path_neo)  ### Obs variables
    x_cen[Nobs_d:]=Atlas['x_cen'][:] ;y_cen[Nobs_d:]=Atlas['y_cen'][:]
    x_max[Nobs_d:]=Atlas['x_max'][:] ;y_max[Nobs_d:]=Atlas['y_max'][:]

    track[Nobs_d:]=Atlas['track'][:]
    time[Nobs_d:]=Atlas['time'][:]
    r_max[Nobs_d:]=Atlas['r_max'][:]
    v_max[Nobs_d:]=Atlas['v_max'][:]
    life_cycle[Nobs_d:]=Atlas['life_cycle'][:]
    ro[Nobs_d:]=Atlas['Ro'][:]

    first_obs[Ned_d:]=Atlas['index_of_first_observation'][:]+Nobs_d  ## Warning here ! specific Neodyn
    last_obs[Ned_d:]=Atlas['last_obs'][:]+Nobs_d
    ro_mean[Ned_d:]=Atlas['ro_mean'][:]   ### eddy variables
    r_max_avg[Ned_d:]=Atlas['r_max_avg'][:]
    life_time[Ned_d:]=Atlas['life_time'][:]
    eddy_origin[Ned_d:]=Atlas['eddy_origin'][:]
    eddy_death[Ned_d:]=Atlas['eddy_death'][:]
    dSSH[Nobs_d:]=Atlas['dssh'][:]  ## In NEODYN v2, dssh is dssh_max by default

    new_index=Atlas['eddy_index_nb'][:]
    Atlas.close()

    first_obs=first_obs.astype(int)
    last_obs=last_obs.astype(int)
    return first_obs,last_obs,x_cen,y_cen,x_max,y_max,track,time,r_max,v_max,life_cycle,ro,dSSH,ro_mean,r_max_avg,life_time,eddy_origin,eddy_death,new_index

