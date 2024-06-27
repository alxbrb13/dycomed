#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Fri May  1 17:37:04 2020

@author: alexandre

This script takes as input the profiles previously merged in a common netCDF file by script 'merge_cora_vXXX.py',
compute the associated background and put it into the same netCDF file

- "Original background is computed  for each profile as the average
of all profiles OUTSIDE eddies (outside last closed SSH contour), within a distance 'dist_threshold' and within a time interval 'day_interval' modulo 365 days
, that is for all years in the whole time period.
- "Smart" background retrieves only profiles within 100km +/- 30 days within the same year. Number of average profiles do not exceed "min_nb_prof",
Additional profiles are retrieved from other years. See details in the code

Steps done in this script :
    - MLD computations
    - Background (original + "smart")


NEW November 2020 : backgrounds gather only profiles less than "year_max" years to avoid strong decadal variations.
NEW November 2020 : number of non-NaN data depends on depth due to higher variability in the thermocline, see "occurXXX"
NEW April 2021 : - Adding Original & "Smart background"  (with QC on smart background)
                 - Correction on old profiles, XBT profiles removed from background computation
                 - Reduced minimal number of data in the upper column (0-500m) for salinity compared to temperature (see occur500_sal)
 
INPUT
netcdf_data     str     name of the netCDF file merging all profiles

day_interval    float   time interval, in days
dist_threshold  float   maximal distance from the profile, in kilometers
occur500       int   minimal data number for temperature below which a background measurement is flagged as NaN, from the surface to 500m.
occur500_sal   int   minimal data number for salinity below which a background measurement is flagged as NaN, from the surface to 500m.
occur700       int   minimal data number below which a background measurement is flagged as NaN, from 500m to 700m.
occurbottom     int   minimal data number below which a background measurement is flagged as NaN, below 700m to the bottom
min_nb_prof     int     minimal number of profile in the background, discarded otherwise
year_max        int     maximal delay in years - for smart background - between the profiles for which the background is computed and profiles constituting this background.

OUTPUT : new variables in the SAME netCDF file 'netcdf_data' :
'data_type'
'gridded_temp_anom'
'gridded_sal_anom'
'gridded_sigma_pot_anom'

'gridded_temp_background'
'gridded_sal_background'
'gridded_ptemp_background'
'gridded_sigma_pot_background'
'gridded_temp_back_std'
'gridded_sal_back_std'
'gridded_sigma_pot_back_std'

"""

# import datetime as dt
import os
import numpy as np
import netCDF4 as nc4
from tqdm import tqdm
#import gsw
import matplotlib.pyplot as plt
import sys
sys.path.append('/home6/datahome/abarboni/DYNED-NRT/')
from tools_dyned_these import  distance


### Parametres de Briac : « dans une période de ± 30 jours quelque soit l'année et à moins de 150km du profil en question »

recomputation =True ## If background already in netCDF and simply recomputation
Keys=['original','smart']
region ='MED' ## ARA or MED
path_work='/home6/datawork/abarboni/OceanData/'

if region =='ARA':
    dist_threshold_orig=150
    dist_close=100
    day_interval=30
    min_nb_prof=20
    year_max=5
    occur500=min_nb_prof ; occur700=10 ; occur_bottom=5
    occur500_sal=20
    netcdf_data='DYCOMED_ARA_2000-2015_TEST.nc'
    
if region =='MED':
    dist_threshold_orig=150
    dist_close=100
    day_interval=30
    min_nb_prof=30
    year_max=1
    occur500=min_nb_prof ; occur700=10 ; occur_bottom=5
    occur500_sal=20
    netcdf_data='DYCOMED_2000-2021_v4_+back.nc'
    netcdf_new='DYCOMED_2000-2021_v4_+back.nc'
# %% Loading NetCDF data

### Eastern Med profiles to compute background
f_merge = nc4.Dataset(path_work+netcdf_data,'r', format='NETCDF4')
xprof=f_merge['Longitude'][:]
yprof=f_merge['Latitude'][:]
timeprof=f_merge['Days'][:]
yearprof=2000+timeprof*20/7304
prof2obs=f_merge['index_dyned_obs'][:]
eddy_tag=f_merge['eddy_tag'][:]

Tprof=f_merge['gridded_temp'][:]
PTprof=f_merge['gridded_ptemp'][:]
Sprof=f_merge['gridded_sal'][:]
Dprof=f_merge['gridded_sigma_pot'][:]
depth=f_merge['Depth_ref'][:]
mldT=f_merge['mld_from_temp'][:]
mldSigma=f_merge['mld_from_sigma'][:]
f_merge.close()
Nprof=len(xprof)

id500=np.argmin(np.abs(depth-500))
id700=np.argmin(np.abs(depth-700))

# %% Listing profiles constituting the background
ListDir=os.listdir(path_work)
if not recomputation : # 'DictSmart+Original_'+region+'.npy' in ListDir:
    DictList=np.load('DictSmart+Original_'+region+'.npy', allow_pickle=True).item()
else:
    DictList={}
    for h in range(len(Keys)):
        DictList['back_'+Keys[h]]=[]
        DictList['max_nb_'+Keys[h]]=[]
        DictList['QC_'+Keys[h]]=[]
    
    OutEddy=(prof2obs==-1)
    ## back_index = for each profile, index of profiles available to build the background
    for i in tqdm(range(0, len(xprof))):
        
        ### Conditions
        DayInterv=(((timeprof-timeprof[i])%365.25>365-day_interval) + ((timeprof-timeprof[i])%365.25<day_interval))
        YearSel=(np.abs(timeprof-timeprof[i])<year_max*365+day_interval)
        YearAnnual=(np.abs(timeprof-timeprof[i])<365+30)
        YearSame=(np.abs(timeprof-timeprof[i])<30)
        DistThres = (distance(xprof[i], xprof, yprof[i], yprof)<dist_threshold_orig)
        DistClose = (distance(xprof[i], xprof, yprof[i], yprof)<dist_close)
        
        
        DictList['back_original']+=[np.where(OutEddy & DayInterv & DistThres)[0]]  # & Sal_present    
        ## Building Smart background
        smart_qc=1
        back_smart=np.where( OutEddy & DayInterv & DistClose & YearSame )[0]
        if len(back_smart)<min_nb_prof:
            smart_qc=2
            back_smart=np.where(OutEddy & DayInterv & DistClose & YearAnnual )[0]
            if len(back_smart)<min_nb_prof:
                smart_qc=3
                back_smart=np.where(OutEddy & DayInterv & DistThres & YearSel )[0]
                if len(back_smart)<min_nb_prof:
                    smart_qc=4
        DictList['back_smart']+=[back_smart]
        DictList['QC_smart']+=[smart_qc]    
        
        for k in range(len(Keys)):
            DictList['max_nb_'+Keys[k]]+=[len(DictList['back_'+Keys[k]][-1])]
        
    for k in range(len(Keys)):
        DictList['max_nb_'+Keys[k]]=np.array(DictList['max_nb_'+Keys[k]])   
    np.save(path_work+'DictSmart+Original_'+region+'.npy',DictList)
# %% Check Background numbers
plt.figure(0, figsize=(8,6))
# plt.hist(max_year,bins=300, histtype='step',color='m',label='Yearly background')
plt.hist(DictList['max_nb_smart'],bins=100, histtype='step',color='orange',label='Smart background')
plt.hist(DictList['max_nb_original'],bins=100, histtype='step',color='b',label='2000-2019 background')
plt.xlim([0,300]) ; plt.legend()
plt.title('Background profiles number histogram')
plt.xlabel('Number of profiles as background')
plt.ylabel('Number of profiles') ; plt.xlim([0,500])
plt.savefig(path_work+'Profiles_Histo.png')
plt.close()
print('Number of profiles with no consistent background :'+str(len(np.where(np.array(DictList['max_nb_smart'])<min_nb_prof)[0])))
# %% Check Background Quality

plt.figure(1)
plt.hist(DictList['QC_smart'])
plt.xticks(np.arange(1,5),np.arange(1,5).astype(str))
plt.title('Smart Background quality', size=16)
plt.savefig(path_work+'Profiles_QC.png')
plt.close()

# %% Computing mean background
T_orig=np.ones((len(xprof),len(depth)))*(-1)
PT_orig=np.ones((len(xprof),len(depth)))*(-1)
S_orig=np.ones((len(xprof),len(depth)))*(-1)
D_orig=np.ones((len(xprof),len(depth)))*(-1)

T_smart=np.ones((len(xprof),len(depth)))*(-1)
PT_smart=np.ones((len(xprof),len(depth)))*(-1)
S_smart=np.ones((len(xprof),len(depth)))*(-1)
D_smart=np.ones((len(xprof),len(depth)))*(-1)

T_std=np.ones((len(xprof),len(depth)))*(-1)
S_std=np.ones((len(xprof),len(depth)))*(-1)
D_std=np.ones((len(xprof),len(depth)))*(-1)

Socc_orig=np.zeros((len(xprof),len(depth)))
Tocc_orig=np.zeros((len(xprof),len(depth)))
Socc_smart=np.zeros((len(xprof),len(depth)))
Tocc_smart=np.zeros((len(xprof),len(depth)))

# Back_nb_orig=np.zeros(len(xprof))
# Back_nb_smart=np.zeros(len(xprof))
Back_nb_orig=DictList['max_nb_original']
Back_nb_smart=DictList['max_nb_smart']

mldT_back=np.zeros(len(xprof))
mldSigma_back=np.zeros(len(xprof))

for i in tqdm(range(len(xprof))):
    back_original=DictList['back_original'][i]
    back_smart=DictList['back_smart'][i]
    
    T_orig[i]=np.nanmean(Tprof[back_original,:], axis=0).data
    PT_orig[i]=np.nanmean(PTprof[back_original,:], axis=0).data
    S_orig[i]=np.nanmean(Sprof[back_original,:], axis=0).data
    D_orig[i]=np.nanmean(Dprof[back_original,:], axis=0).data
    
    T_smart[i]=np.nanmean(Tprof[back_smart,:], axis=0).data
    PT_smart[i]=np.nanmean(PTprof[back_smart,:], axis=0).data
    S_smart[i]=np.nanmean(Sprof[back_smart,:], axis=0).data
    D_smart[i]=np.nanmean(Dprof[back_smart,:], axis=0).data
    
    T_std[i]=np.nanstd(Tprof[back_smart,:], axis=0).data
    S_std[i]=np.nanstd(Sprof[back_smart,:], axis=0).data
    D_std[i]=np.nanstd(Dprof[back_smart,:], axis=0).data    
    
    Tocc_orig[i]=np.sum(~np.isnan(Tprof[back_original,:]),axis=0)
    Socc_orig[i]=np.sum(~np.isnan(Sprof[back_original,:]),axis=0)   ## masked values do not count !
    
    Tocc_smart[i]=np.sum(~np.isnan(Tprof[back_smart,:]),axis=0)
    Socc_smart[i]=np.sum(~np.isnan(Sprof[back_smart,:]),axis=0)   ## masked values do not count !
    
    # Median Background MLD and not Mean !
    mldT_back[i]=np.nanmedian(mldT[back_smart])
    mldSigma_back[i]=np.nanmedian(mldSigma[back_smart])

# %% Nan values  & Data classification

### Selective filter in depth on minimal number of value
# Warning : depends on depth vector size
occur_raw=np.array([occur500]*id500+[occur700]*(id700-id500)+[occur_bottom]*(len(depth)-id700))
OccurM=np.repeat(occur_raw[np.newaxis,:],len(xprof),axis=0)
occur_raw_sal=np.array([occur500_sal]*id500+[occur700]*(id700-id500)+[occur_bottom]*(len(depth)-id700))
OccurMS=np.repeat(occur_raw_sal[np.newaxis,:],len(xprof),axis=0)

PT_smart[Socc_smart<OccurMS]=np.nan
T_smart[Tocc_smart<OccurM]=np.nan
S_smart[Socc_smart<OccurMS]=np.nan
D_smart[Socc_smart<OccurMS]=np.nan

PT_orig[Socc_orig<OccurMS]=np.nan
T_orig[Tocc_orig<OccurM]=np.nan
S_orig[Socc_orig<OccurMS]=np.nan
D_orig[Socc_orig<OccurMS]=np.nan

T_std[Tocc_smart<OccurM]=np.nan
S_std[Socc_smart<OccurMS]=np.nan
D_std[Socc_smart<OccurMS]=np.nan

print('WARNING : flagging '+str(len(np.where(Tocc_orig<OccurM)[0]))+' data in ORIGINAL as NaN')
print('WARNING : flagging '+str(len(np.where(Tocc_smart<OccurM)[0]))+' data in SMART as NaN')
print('in '+str(len(np.unique(np.where(Tocc_smart<OccurM)[0])))+' profiles')

# %% Discarding profiles with not enough background profiles

PT_orig[Back_nb_orig<min_nb_prof]=np.nan
T_orig[Back_nb_orig<min_nb_prof]=np.nan
S_orig[Back_nb_orig<min_nb_prof]=np.nan
D_orig[Back_nb_orig<min_nb_prof]=np.nan

PT_smart[Back_nb_smart<min_nb_prof]=np.nan
T_smart[Back_nb_smart<min_nb_prof]=np.nan
S_smart[Back_nb_smart<min_nb_prof]=np.nan
D_smart[Back_nb_smart<min_nb_prof]=np.nan

T_std[Back_nb_smart<min_nb_prof]=np.nan
S_std[Back_nb_smart<min_nb_prof]=np.nan
D_std[Back_nb_smart<min_nb_prof]=np.nan

print('WARNING : discarding '+str(len(np.where(Back_nb_smart<min_nb_prof)[0]))+' profiles having less than '+str(min_nb_prof)+' background profiles')

# %% Create Variables
f_merge = nc4.Dataset(path_work+netcdf_new,'w', format='NETCDF4')
if not recomputation:
    f_merge.createVariable('QC_smart_back', 'i1', 'Nprof')
    f_merge.createVariable('background_number_original', 'i4', 'Nprof' )
    f_merge.createVariable('background_number_smart', 'i4', 'Nprof' )
    f_merge.createVariable('background_mld_from_temp', 'f4', 'Nprof' )
    f_merge.createVariable('background_mld_from_sigma', 'f4', 'Nprof' )
    
    f_merge.createVariable('temp_anom', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('sal_anom', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('sigma_anom', 'f4', ('Nprof','grid_depth') )
    
    f_merge.createVariable('temp_back_original', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('sal_back_original', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('ptemp_back_original', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('sigma_back_original', 'f4', ('Nprof','grid_depth') )
    
    f_merge.createVariable('temp_back_smart', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('sal_back_smart', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('ptemp_back_smart', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('sigma_back_smart', 'f4', ('Nprof','grid_depth') )
    
    f_merge.createVariable('temp_back_std_smart', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('sal_back_std_smart', 'f4', ('Nprof','grid_depth') )
    f_merge.createVariable('sigma_back_std_smart', 'f4', ('Nprof','grid_depth'))

# %%
f_merge['QC_smart_back'].units = 'Smart background QC : 1 = only same year ; 2 = +/- one year max ; 3 = +/- 5 years max ; 4 = Not enough profiles for background, even +/- 5 years.'

f_merge['background_number_original'].units='Number of profiles in the background - original method. \n Background filled of NaN if less than '+str(min_nb_prof)+' profiles'
f_merge['background_number_smart'].units='Number of profiles in the background - smart method. \n Background filled of NaN if less than '+str(min_nb_prof)+' profiles'

f_merge['background_mld_from_temp'].units='Mean MLD in the (smart) background, computed over temperature profiles'
f_merge['background_mld_from_sigma'].units='Mean MLD in the (smart) background, computed over potential density profiles'

f_merge['temp_anom'].units = 'In situ Temperature anomaly, in degC, interpolated on depth_ref vector \n Computed on Smart background \n Assumed to be equivalent at Potential temperature anomaly'
f_merge['sal_anom'].units = 'Practical Salinity anomaly, in PSU, interpolated on depth_ref vector  \n Computed on Smart background'
f_merge['sigma_anom'].units = 'Potential density anomaly, in kg/m³, interpolated on depth_ref vector  \n Computed on Smart background'

f_merge['temp_back_original'].units = 'In situ Temperature background, in degC, interpolated on depth_ref vector \n Original method'
f_merge['sal_back_original'].units = 'Practical Salinity background, in PSU, interpolated on depth_ref vector \n Original method'
f_merge['ptemp_back_original'].units = 'Potential Temperature background, in degC, interpolated on depth_ref vector \n Original method'
f_merge['sigma_back_original'].units = 'Potential density background -1000, in kg/m³, interpolated on depth_ref vector \n Original method'

f_merge['temp_back_smart'].units = 'In situ Temperature background, in degC, interpolated on depth_ref vector \n Smart method'
f_merge['sal_back_smart'].units = 'Practical Salinity background, in PSU, interpolated on depth_ref vector \n Smart method'
f_merge['ptemp_back_smart'].units = 'Potential Temperature background, in degC, interpolated on depth_ref vector \n Smart method'
f_merge['sigma_back_smart'].units = 'Potential density background -1000, in kg/m³, interpolated on depth_ref vector \n Smart method'

f_merge['temp_back_std_smart'].units = 'In situ Temperature standard deviation, in degC, interpolated on depth_ref vector \n Assumed to be equivalent at Potential temperature background standard deviation'
f_merge['sal_back_std_smart'].units = 'Practical Salinity standard deviation, in PSU, interpolated on depth_ref vector'
f_merge['sigma_back_std_smart'].units = 'Potential density standard deviation, in kg/m³, interpolated on depth_ref vector'

# %%
f_merge['QC_smart_back'][:]=DictList['QC_smart']

f_merge['background_number_original'][:]=Back_nb_orig
f_merge['background_number_smart'][:]=Back_nb_smart
f_merge['background_mld_from_temp'][:]=mldT_back
f_merge['background_mld_from_sigma'][:]=mldSigma_back

f_merge['temp_anom'][:]=Tprof-T_smart
f_merge['sal_anom'][:]=Sprof-S_smart
f_merge['sigma_anom'][:]=Dprof-D_smart

f_merge['temp_back_original'][:]=T_orig
f_merge['sal_back_original'][:]=S_orig
f_merge['ptemp_back_original'][:]=PT_orig
f_merge['sigma_back_original'][:]=D_orig

f_merge['temp_back_smart'][:]=T_smart
f_merge['sal_back_smart'][:]=S_smart
f_merge['ptemp_back_smart'][:]=PT_smart
f_merge['sigma_back_smart'][:]=D_smart

f_merge['temp_back_std_smart'][:]=T_std
f_merge['sal_back_std_smart'][:]=S_std
f_merge['sigma_back_std_smart'][:]=D_std
# %%
f_merge.close()
