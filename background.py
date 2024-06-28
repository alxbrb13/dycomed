#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May  1 17:37:04 2020

@author: alexandre

This script takes as input the profiles previously merged in a common netCDF file by script 'merge_cora_vXXX.py',
compute the associated background and put it into the same netCDF file

####### INPUT
netcdf_data     str     path to netCDF dataset
prof_dim_name   str     name of data raws dimension in netcdf_data
z_dim_name      str     name of vertical dimension in netcdf_data (if applicable)
z_vars          list of str # variables as vertical profiles in netcdf_data, assumed as 2D [#prof,#z]
surf_vars       list of str # variables as point data (e.g. surface only) in netcdf_data, assumed as 1D [#prof]

day_interval    float   time interval, in days
dist_threshold  float   maximal distance from the profile, in kilometers (good guess is at least 4-5 deformation radius)
min_nb_data     int     minimal number of profile in the background, otherwise no background computed
year_max        int     maximal delay in years between the profiles for which the background is computed and profiles constituting this background.

###### OUTPUT : for each variable 'Var' listed in either z_vars or surf_vars, 
        corresponding background information are added in the SAME netCDF file 'netcdf_data' :

'Var_anom'       
'Var_back'
'Var_std'

"""
import numpy as np
import netCDF4 as nc4
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tools_dyco import  distance

### paths
netcdf_data='./Dataset_v0.nc'  ## path to dataset
prof_dim_name='Nprof'          ## name of data rows dimension in netcdf_data
z_dim_name='grid_depth'       ## name of vertical dimension in netcdf_data (if applicable)

### variables names
z_vars=['PSAL','TEMP','sigma_pot']  ### variables as vertical profiles, assumed as 2D in netcdf dataset [#prof,#z]
surf_vars=[]                       ### variable as point data (e.g. only in surface), assumed as 1D in netcdf dataset [#prof]

### Background computation params
dist_threshold=150
day_interval=30
min_nb_data=20
year_max=1

# %% Loading NetCDF data

f_merge = nc4.Dataset(netcdf_data,'r', format='NETCDF4')
xprof=f_merge['Longitude'][:]
yprof=f_merge['Latitude'][:]     ### in degrees
timeprof=f_merge['Time'][:]      ### should be in days 

### info from collocation step (eddy_link.py)
prof2obs=f_merge['index_dyned_obs'][:]
eddy_tag=f_merge['eddy_tag'][:]

MainDict={}  ### Loading variables 
for var in z_vars+surf_vars:
    MainDict[var+'-prof']=f_merge[var][:]

Nprof=f_merge.dimensions[prof_dim_name].size
if len(z_vars)>0:
    Nz=f_merge.dimensions[z_dim_name].size
f_merge.close()

# yearprof=2000+timeprof*20/7304

# %% Listing profiles constituting the background

DictList={}
DictList['back_index']=[]

OutEddy=(prof2obs==-1)
## back_index = for each profile, index of profiles available to build the background
for i in tqdm(range(Nprof)):
    
    ### Conditions
    DayInterv=(((timeprof-timeprof[i])%365.25>365-day_interval) + ((timeprof-timeprof[i])%365.25<day_interval))
    YearSel=(np.abs(timeprof-timeprof[i])<year_max*365+day_interval)
    DistThres = (distance(xprof[i], xprof, yprof[i], yprof)<dist_threshold)
    DictList['back_index']+=[np.where(OutEddy & DayInterv & DistThres)[0]]  # & Sal_present    

BackNb=np.array([ len(l) for l in DictList['back_index']])

# %% Check Background numbers

plt.figure(0, figsize=(8,6))
plt.hist(BackNb,bins=np.arange(0,min_nb_data+1,5),color='r',label='Not enough profiles for background')
plt.hist(BackNb,bins=np.arange(min_nb_data,10*min_nb_data,5),color='g',label='Enough profiles for background')
plt.xlim([0,min_nb_data*10]) ; plt.legend()
plt.title('Background profiles number histogram')
plt.xlabel('Number of profiles as background')
plt.ylabel('Number of profiles')

print('Number of profiles with no consistent background :'+str(len(np.where(BackNb<min_nb_data)[0])))
#%% Small map
fig=plt.figure(0,figsize=(18,10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

plt.plot(xprof[BackNb>min_nb_data],yprof[BackNb>min_nb_data],'og', label='Enough profiles for background')
plt.plot(xprof[BackNb<min_nb_data],yprof[BackNb<min_nb_data],'or', label='Not enough profiles for background')
plt.legend(fontsize=14) ; plt.grid()
ax.coastlines()

# plt.xlim(CoorLon) ; plt.ylim(CoorLat)
plt.title('region of interest profiles')


# %% Computing mean background

for var in surf_vars:
    MainDict[var+'-back']=np.ones(np.shape(MainDict[var+'-prof']))*(-1)
    MainDict[var+'-std']=np.ones(np.shape(MainDict[var+'-prof']))*(-1)

for var in z_vars:
    MainDict[var+'-back']=np.ones(np.shape(MainDict[var+'-prof']))*(-1)
    MainDict[var+'-std']=np.ones(np.shape(MainDict[var+'-prof']))*(-1)
    MainDict[var+'-occ']=np.zeros(np.shape(MainDict[var+'-prof']))


for i in tqdm(range(Nprof)):
    back_index=DictList['back_index'][i]
    
    for var in z_vars:
        MainDict[var+'-back'][i]=np.nanmean(MainDict[var+'-prof'][back_index,:], axis=0).data
        MainDict[var+'-std'][i]=np.nanstd(MainDict[var+'-prof'][back_index,:], axis=0).data  
        MainDict[var+'-occ'][i]=np.sum(~np.isnan(MainDict[var+'-prof'][back_index,:]),axis=0)   ## masked values do not count !
        
    for var in surf_vars:
        MainDict[var+'-back'][i]=np.nanmean(MainDict[var+'-prof'][back_index]).data
        MainDict[var+'-std'][i]=np.nanstd(MainDict[var+'-prof'][back_index]).data  
            
# %% Nan in background if not enough profiles in background, for each vertical level

if len(z_vars)>0:
    # Nz=np.shape(MainDict[z_vars[0]+'-prof'])[1]
    # occur_raw=np.array([occur500]*id500+[occur700]*(id700-id500)+[occur_bottom]*(Nz-id700))
    OccurM=np.repeat(np.array([min_nb_data]*Nz)[np.newaxis,:],Nprof,axis=0)

    for var in z_vars:
        MainDict[var+'-back'][MainDict[var+'-occ']<OccurM]=np.nan

        print('WARNING : * '+var+' * flagging '+str(len(np.where(MainDict[var+'-occ']<OccurM)[0]))+' background value as NaN because not enough backgroudn data')

for var in surf_vars:
    MainDict[var+'-back'][BackNb<min_nb_data]=np.nan
    print('WARNING : * '+var+' * flagging '+str(len(np.where(BackNb<min_nb_data)[0]))+' background value as NaN because not enough backgroudn data')

# Discarding profiles with not enough background profiles
# T_orig[Back_nb_orig<min_nb_data]=np.nan
# T_std[Back_nb_smart<min_nb_data]=np.nan

# %% Create Variables

f_merge = nc4.Dataset(netcdf_data,'a', format='NETCDF4')
f_merge.createVariable('background_number', 'i4', 'Nprof' )

for var in z_vars:
    f_merge.createVariable(var+'_anom', 'f4', (prof_dim_name,z_dim_name) )
    f_merge.createVariable(var+'_back', 'f4', (prof_dim_name,z_dim_name) )
    f_merge.createVariable(var+'_std', 'f4', (prof_dim_name,z_dim_name) )
for var in surf_vars:
    f_merge.createVariable(var+'_anom', 'f4', prof_dim_name )
    f_merge.createVariable(var+'_back', 'f4', prof_dim_name)
    f_merge.createVariable(var+'_std', 'f4', prof_dim_name )

f_merge['background_number'].units='Number of profiles in the background. \n Background filled of NaN if less than '+str(min_nb_data)+' profiles'

# %%

f_merge['background_number'][:]=BackNb

for var in z_vars+surf_vars:
    f_merge[var+'_anom'][:]=MainDict[var+'-prof']-MainDict[var+'-back']
    f_merge[var+'_back'][:]=MainDict[var+'-back']
    f_merge[var+'_std'][:]=MainDict[var+'-std']
# %%
f_merge.close()
