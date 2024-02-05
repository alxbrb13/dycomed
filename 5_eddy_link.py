# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:16:31 2021

@author: alexandre

Steps done in this script
    - Make colocalisation with DYNED Atlas
    - Compute eddy tag (-2 / -1 / 0 / 1 / 2)
    
New from November 2020 :
    - Tag observations to eddies in Dyned Atlas, differenciating last closed SSH contour and maximal speed radius.
    (Routine link_eddies_end)
    - Compute MLD from temperature and density profile (respective threshold 0.1 degC and 0.03 kg/m^3)

New from May 2021 :
    - New eddy tag (-2 / -1 / 0 / 1 / 2) = former eddy tag (0/1/2)*sign(Ro)
    
New November 2021 :
    - Colocalization at ± x days, stored in variable 'extended_dyned_obs'
    - New eddy tag=(-3 /-2 /-1 / 0 / 1 / 2 / 3) => ± 3 for "ambiguous" colocalization
    
New January 2022 :
    - Crossed colocalisation DYNED - NEODYN
    
New December 2023 :
    - Near Real time application

"""
import datetime as dt
import numpy as np
import os
import netCDF4 as nc4
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('/home6/datahome/abarboni/DYNED-NRT/')
from tools_dycomed import MLD_threshold, newlink, str2juld, juld2str, distance
from tools_dyned_these import load_bathy, plot_bathy
# from tools_dyned_these import distance, MLD_threshold #, unique_prof, link_eddies_end, 
# import gsw
from matplotlib import rcParams
plt.style.use(u'default')
rcParams['contour.negative_linestyle'] = 'solid' 

delay=2         ## days for extended colocalization before/after day of cast (colocalization over (2*delay+1) days )
refresh=6   ## refreshed days (typically 7 for D+6 NRT SSH)
region ='MED' ## ARA or MED
#dateF=dt.date(2020,1,1)  ## starting date of NRT detections (NEODYN Atlas)
CoorLon=[-6,36] ; CoorLat=[30,46] ### Mediterranean

if region =='ARA':
    search_radius=200
    path_dyned='/media/alexandre/HDD/Documents/These/Eddies_ARA/LocalData/dyned_atlas_3D_arabie_20000101_20151231_20190225.nc'
    filename='DYCOMED_ARA_2000-2015_v1_withArgos.nc'
    
if region =='MED':
    search_radius=150
    path_obs='/home6/datawork/abarboni/DYCOMED/SSH_18_MED/'
    file_obs='PET_Atlas_untracked_'
    path_prof='/home6/datawork/abarboni/DYCOMED/PROF/'
    file_prof='CORA_NRT_MED_merged_'

# %%
path_topo='/home6/datawork/abarboni/OceanData/'
lon_t,lat_t,z_t=load_bathy(path_topo+'ETOPO2v2c_f4.nc', CoorLon, CoorLat)

# %%
#### finding last merged obs dataset
ListDir=os.listdir(path_obs) ; Atlas=[]
for name in ListDir:
    if name[:len(file_obs)]==file_obs:
        Atlas+=[name]
Atlas=np.sort(Atlas) ;
name_atlas=Atlas[0]

# %%
### Loading eddy obs
f=nc4.Dataset(path_obs+name_atlas,'r')
Pol=f['polarity'][:]
x_cen=f['x_cen'][:]
y_cen=f['y_cen'][:]
x_max=f['x_max'][:]
y_max=f['y_max'][:]
x_end=f['x_end'][:]
y_end=f['y_end'][:]
time_eddy=f['time_eddy'][:]
f.close()

# %%
### finding last merged profile dataset
ListDir=os.listdir(path_prof) ; Merged=[]
for name in ListDir:
    if name[:len(file_prof)]==file_prof:
        Merged+=[name]
Merged=np.sort(Merged) ;

name_merged=Merged[0] ; date_m=name_merged[-11:-3]
print('Date of previously merged profiles dataset :'+date_m)

# %% Loading NetCDF data
### Loading profiles
f_merge = nc4.Dataset(path_prof+name_merged,'r', format='NETCDF4')
time_common=f_merge['Days'][:]
juldnew=str2juld(date_m)-delay-refresh

print('\n Updating profiles colocalization from '+juld2str(juldnew))

select=(time_common>=juldnew)  ### selecting only new profiles
#yearprof=2000+(time_common/365.25)
#select=(yearprof>=2022)  ### selecting only new profiles

print('\n Colocalization computed for '+str(len(np.where(select)[0]))+' profiles')

### Quick fix
time_common=f_merge['Days'][select]
x_common=f_merge['Longitude'][select]
y_common=f_merge['Latitude'][select]

file_name=f_merge['file_name'][select]
Tprof=f_merge['gridded_temp'][select]
# PTprof=f_merge['gridded_ptemp'][:]
Sprof=f_merge['gridded_sal'][select]
Dprof=f_merge['gridded_sigma_pot'][select]
depth=f_merge['Depth_ref'][:]
f_merge.close()
Nprof=len(x_common)
# %% Linking profiles & eddies
day_common=time_common - time_common%1  ## convert in integer days
day_common[(time_common%1)>0.5]+=1
day_common=day_common.astype(int)

new2obs=-1*np.ones((Nprof,2*delay+1))
newdist=-1*np.ones((Nprof,2*delay+1))
newflag=np.zeros((Nprof,2*delay+1))


# %% Computing colocalization

new2obs, newdist, newflag = newlink(x_common, y_common, day_common, x_cen, y_cen, x_max, y_max, x_end, y_end, time_eddy, delay=delay)
new2obs[new2obs==1999999]=-1
Sum=np.sum(newflag,axis=1) # np.sum(new2obs!=-1,axis=1)

# %% [markdown]
#
# plt.figure(0)
# plt.hist(np.sum(newflag,axis=1), bins=20)
# plt.title('Sum of eddy flag +/- 2 days', size=18)

# %%
## 1D flag
flageddy=np.zeros(Nprof)
flageddy[Sum>0]=3 ## Default : flag 3 if ambiguous
## then :
flageddy[np.where(np.sum(newflag[:]==1,axis=1)==(2*delay+1))[0]]=1 ## flag 1 if always between Rend and Rmax
flageddy[np.where(Sum>=2*(2*delay+1-1))[0]]=2 ## flag 2 if inside eddy 4 times over 5

## 1D observation index
ctd2obs=-1*np.ones(Nprof)
ID=np.where(flageddy>0)[0]
for i in ID:  ## Select obs index as close as possible from the day of cast :
    if new2obs[i,2]!=-1:
        ctd2obs[i]=new2obs[i,2]
    elif new2obs[i,1]!=-1:
        ctd2obs[i]=new2obs[i,1]
    elif new2obs[i,3]!=-1:
        ctd2obs[i]=new2obs[i,3]
    elif new2obs[i,0]!=-1:
        ctd2obs[i]=new2obs[i,0]
    elif new2obs[i,4]!=-1:
        ctd2obs[i]=new2obs[i,4]
ctd2obs=ctd2obs.astype(int)

## 1D distance, averaged over +/- 2 days
disteddy=np.nanmedian(newdist,axis=1)
# %%
## DYNED eddy ID when sampled (New from 20200901 version) :
#prof_track=np.zeros(Nprof)

##  New eddy tag  (New from 20210601  version) showing eddy polarity
eddy_tag=np.zeros(Nprof)
for i in range(Nprof):
    if ctd2obs[i]!=-1:
        #prof_track[i]=track[ctd2obs[i]].astype(int)
        if Pol[ctd2obs[i]]<0:  ## changing size depending on polarity
            eddy_tag[i]=-1
        else:
            eddy_tag[i]=1
        if flageddy[i]==2:
            eddy_tag[i]=eddy_tag[i]*2
        if flageddy[i]==3:
            eddy_tag[i]=eddy_tag[i]*3
#prof_track[ctd2obs==-1]=-1
#prof_track=prof_track.astype(int)
# %% Some stats on Prof2Obs

plt.figure(1)
plt.hist(eddy_tag, bins=14)
plt.xticks([-3,-2,-1,0,1,2,3],['Ambig','Inside','Inbetween','Outside','Inbetween','Inside','Ambig'])
plt.title('Number of profiles')

# %%
Out=len(np.where((np.abs(eddy_tag)==0))[0])
Amb=len(np.where((np.abs(eddy_tag)==1) + (np.abs(eddy_tag)==3))[0])
AE=len(np.where((eddy_tag==2))[0])
CE=len(np.where((eddy_tag==-2))[0])
print('\n  Inside Anticyclone : '+str(AE)+' \n Inside Cyclone : '+str(CE)+' \n Ambiguous : '+str(Amb)+' \n Outside : '+str(Out))

# %% Mixed layer depth calculation on temperature and temperature
mld=np.zeros(Nprof) ; mldT=np.zeros(Nprof)

## At least one of the 5 top measurements are not Nan
for i in tqdm(range(Nprof)):
    mld[i]=MLD_threshold(Dprof[i,1:], depth[1:], delta=0.03, surf_accept=5)
    mldT[i]=MLD_threshold(Tprof[i,1:], depth[1:], delta=0.1, surf_accept=5)

# %% Check plot [markdown]
# density=True
# plt.figure(0)
# for i in range(20):
#     s=20*i+10000 #20*i+600
#     if density :
#         plt.plot(Dprof[s,:], -1*depth, '-')
#         if ~np.isnan(mld[s]): ### Otherwise black dot appear even if mld is nan
#             h=np.argmin(np.abs(mld[s]-depth))
#             plt.plot(Dprof[s,h],-1*depth[h],'ok')
#         if ~np.isnan(mldT[s]):
#             h=np.argmin(np.abs(mldT[s]-depth))
#             plt.plot(Dprof[s,h],-1*depth[h],'or')
#     else:
#         plt.plot(Tprof[s,:], -1*depth, '-')
#         if ~np.isnan(mldT[s]): ### Otherwise black dot appear even if mld is nan
#             h=np.argmin(np.abs(mldT[s]-depth))
#             plt.plot(Tprof[s,h],-1*depth[h],'or')
#         if ~np.isnan(mldT[s]):
#             h=np.argmin(np.abs(mld[s]-depth))
#             plt.plot(Tprof[s,h],-1*depth[h],'ok')
#             
# plt.ylim([-80,0])
# %% Salinity ### New November 2021
Sal_present=(np.sum(~np.isnan(Sprof),axis=1)>1)

# %% Creating new NetCDF variables
f_merge = nc4.Dataset(path_prof+name_merged,'a', format='NETCDF4')

if 'delay' not in f_merge.dimensions: ### if no colocalization ever done
    f_merge.createDimension('delay', 2*delay+1)
    f_merge.createVariable('index_dyned_obs', 'i4', 'Nprof')
    f_merge.createVariable('extended_dyned_obs', 'i4', ('Nprof','delay'))

    f_merge.createVariable('eddy_distance', 'f4', 'Nprof')
    f_merge.createVariable('eddy_tag', 'i4', 'Nprof')
    #f_merge.createVariable('eddy_number', 'i4', 'Nprof')

    f_merge.createVariable('mld_from_sigma', 'f4', 'Nprof')
    f_merge.createVariable('mld_from_temp', 'f4', 'Nprof')
    f_merge.createVariable('Salinity_measured', 'i1', 'Nprof')

    ## Meta data information
    f_merge['index_dyned_obs'].description = 'if =-1 not in eddy, if x!= obs index in obs Atlas'
    f_merge['extended_dyned_obs'].description = 'Dyned Atlas observation number extended to +/- '+str(delay)+' days, if in eddy, =-1 otherwise'
    f_merge['eddy_tag'].description = 'Eddy tag levels : 0 = outside eddy, 1 = within last closed contour, 2 = within max speed radius, \n 3 = Ambiguous (change value at +/- 2 days). Sign is eddy polarity : negative for AE, positive for CE'
    f_merge['eddy_distance'].description = 'eddy center distance in kilometers, averaged over +/- 2 days, if in eddy, =-1 otherwise'
    #f_merge['eddy_number'].description = 'DYNED eddy ID number, if in eddy, =-1 otherwise'

    f_merge['mld_from_sigma'].description = 'Mixed layer depth computed from threshold on density'
    f_merge['mld_from_temp'].description = 'Mixed layer depth computed from threshold on temperature'
    f_merge['Salinity_measured'].description = '1 if Salinity measurements, else =0'


f_merge['index_dyned_obs'][select]=ctd2obs
f_merge['extended_dyned_obs'][select]=new2obs
f_merge['eddy_distance'][select]=disteddy
f_merge['eddy_tag'][select]=eddy_tag
#f_merge['eddy_number'][select]=prof_track

f_merge['mld_from_sigma'][select]=mld
f_merge['mld_from_temp'][select]=mldT
f_merge['Salinity_measured'][select]=Sal_present.astype(int)

f_merge.close()



# %%
Datatype=[]
for file in file_name:
    Datatype+=[file.split('_')[2]]
Datatype=np.array(Datatype)
#Items=np.zeros(len(x_common))
#Items[Datatype=='PF']='d'
#Items[Datatype=='']='x'

# %%

Items=['c']*100+['r']*14
Items=np.array(Items)

# %%
FigSize=(24,10) ;LS=16
fig=plt.figure(2, figsize=FigSize) ; plt.grid()
ax=plt.subplot(111)
plt.title('New profiles since ', size=20)
plot_bathy(lon_t,lat_t,z_t)

#plt.plot(x_common[Datatype=='PF'],y_common[Datatype=='PF'],'r',marker='o', linestyle='')
#plt.plot(x_common[Datatype=='GL'],y_common[Datatype=='GL'],'k',marker='d', linestyle='')
plt.scatter(x_common,y_common, c=Items)
plt.tick_params(labelsize=LS)
#map_chart(ax, CoorLon, CoorLat, 0.2, 0.2, n_x=1, n_y=1, stepy=0)
plt.xlim(CoorLon) ; plt.ylim(CoorLat) #; plt.legend(loc=3,fontsize=LS)
plt.xlabel('Longitude [\N{DEGREE SIGN}E]',size=LS) ; plt.ylabel('Longitude [\N{DEGREE SIGN}E]',size=LS) 


# %%
