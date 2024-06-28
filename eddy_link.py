# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:16:31 2021

@author: alexandre

Steps done in this script
    - Make colocalisation with DYNED Atlas
    - Tag observations to eddies in Dyned Atlas, differenciating last closed SSH contour and maximal speed radius.
    (Routine link_eddies_end)
    - Colocalization at ± x days, stored in variable 'extended_dyned_obs'
    - New eddy tag=(-2 /-1 / 0 / 1 / 2 ) => ± 1 for "ambiguous" colocalization 

"""
import datetime as dt
import numpy as np
import os
import netCDF4 as nc4
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from tools_dyco import link_eddies

CoorLon=[-6,36] ; CoorLat=[30,46] ### Mediterranean

#### Colocation parameters
search_radius=150 ## radius to look for center of likely collocated eddy obs. A good guess is a twice the deformation radius.
                 # ( large values of search radius slows down the algo) 
delay=2           ## days for extended colocalization before/after day of cast (colocalization over (2*delay+1) days )
track_provided=True  ## Are eddy obs already linked between timesteps in an eddy track ?

### Paths
path_atlas='./DYNED_Atlas_3D_2000_2019.nc'
path_data='./Dataset_v0.nc'
# %%
### Loading eddy obs (example with Dyned atlas)
fd=nc4.Dataset(path_atlas,'r')
f=fd['Atlas']
Pol=f['Ro'][:]   ### +1 if obs is cyclone  ; -1 if anticyclone
x_cen=f['x_cen'][:]
y_cen=f['y_cen'][:]
x_max=f['x_max'][:]
y_max=f['y_max'][:]
x_end=f['x_end'][:]
y_end=f['y_end'][:]
time_eddy=f['time'][:]
if track_provided:
    track=f['track'][:]
fd.close()

# %% Loading NetCDF data
### Loading profiles, Note that here only position data are used
f_merge = nc4.Dataset(path_data,'r', format='NETCDF4')
time_cast=f_merge['Time'][:]
x_cast=f_merge['Longitude'][:]
y_cast=f_merge['Latitude'][:]
f_merge.close()
Nprof=len(x_cast)

day_cast=time_cast - time_cast%1  ## convert in integer days
day_cast[(time_cast%1)>0.5]+=1
day_cast=day_cast.astype(int)

#%% Checking time range of in situ data is covered by eddy observation

Filter_time=np.in1d(day_cast,np.unique(time_eddy))
print(str(len(np.where(Filter_time)[0]))+' (out of '+str(len(x_cast))+') profiles covered by eddy atlas')

if len(np.where(~Filter_time)[0])>0:
    print("########## \n WARNING : the remaining "+str(len(np.where(~Filter_time)[0]))+" profiles will be flagged as 'outside-eddy' by default"
          + '\n Did you check time reference is the same for eddy atlas and in situ data ? \n ############')
# x_cast=x_cast[Filter_time]
# y_cast=y_cast[Filter_time]
# day_cast=day_cast[Filter_time]
# %% Linking profiles & eddies

ctd2obs,eddy_dist, eddy_tag = link_eddies(x_cast, y_cast, day_cast, x_cen, y_cen, x_max, y_max, x_end, y_end, time_eddy, delay=delay, search_radius=search_radius)

Sum=np.sum(eddy_tag,axis=1) # np.sum(new2obs!=-1,axis=1)

# %% 1D tag, obs_index and distance
eddy_tag_1d=np.zeros(Nprof)
eddy_tag_1d[Sum>0]=3 ## Default : flag 3 if ambiguous
## then :
eddy_tag_1d[np.where(np.sum(eddy_tag[:]==1,axis=1)==(2*delay+1))[0]]=1 ##% flag 1 if always between Rend and Rmax
eddy_tag_1d[np.where(Sum>=2*(2*delay+1-1))[0]]=2 ## flag 2 if inside eddy 80 % of the time

## 1D observation index
ctd2obs_1d=-1*np.ones(Nprof)
ID=np.where(eddy_tag_1d>0)[0]
for i in ID:  ## Select obs index as close as possible from the day of cast 

    if ctd2obs[i,delay]!=-1:
        ctd2obs_1d[i]=ctd2obs[i,delay]
    elif ctd2obs[i,delay-1]!=-1:
        ctd2obs_1d[i]=ctd2obs[i,delay-1]
    elif ctd2obs[i,delay+1]!=-1:
        ctd2obs_1d[i]=ctd2obs[i,delay+1]
    else:  ## if not day of cast +/- 1 day, takes the first available
        ctd2obs_1d[i]=ctd2obs[i,ctd2obs[i,:]!=-1][0]
ctd2obs_1d=ctd2obs_1d.astype(int)

##  Distance from in situ data to eddy center, averaged over +/- 2 days
eddy_dist_1d=np.nanmedian(eddy_dist,axis=1)
# %%  New eddy tag showing eddy polarity + track collocation
eddy_tag_pol=np.zeros(Nprof)

for i in range(Nprof):
    if ctd2obs_1d[i]!=-1:
        if Pol[ctd2obs_1d[i]]<0:  ## changing size depending on polarity
            eddy_tag_pol[i]=-1
        else:
            eddy_tag_pol[i]=1

        if eddy_tag_1d[i]==2:
            eddy_tag_pol[i]=eddy_tag_pol[i]*2
        # if eddy_tag_1d[i]==3:
        #     eddy_tag[i]=eddy_tag[i]*3
        
if track_provided:
    prof_track=track[ctd2obs_1d].astype(float)
    prof_track[ctd2obs_1d==-1]=-1
    prof_track=prof_track.astype(int)
# %% Some stats on Prof2Obs

plt.figure(1)
plt.hist(eddy_tag_pol, bins=4*delay+1)
plt.xticks([-2,-1,0,1,2],['Inside \n Anticyclone','Ambiguous','Outside','Ambiguous','Inside \n Cyclone'])
plt.title('Number of profiles')

Out=len(np.where((np.abs(eddy_tag_pol)==0))[0])
Amb=len(np.where((np.abs(eddy_tag_pol)==1))[0])
AE=len(np.where((eddy_tag_pol==-2))[0])
CE=len(np.where((eddy_tag_pol==+2))[0])
print('\n  Inside Anticyclone : '+str(AE)+' \n Inside Cyclone : '+str(CE)+' \n Ambiguous : '+str(Amb)+' \n Outside : '+str(Out))


# %% Creating new NetCDF variables
f_merge = nc4.Dataset(path_data,'a', format='NETCDF4')

# if 'delay' not in f_merge.dimensions: ### if no colocalization ever done
f_merge.createDimension('delay', 2*delay+1)
f_merge.createVariable('index_dyned_obs', 'i4', 'Nprof')
f_merge.createVariable('extended_dyned_obs', 'i4', ('Nprof','delay'))

f_merge.createVariable('eddy_distance', 'f4', 'Nprof')
f_merge.createVariable('eddy_tag', 'i4', 'Nprof')

## Meta data information
f_merge['index_dyned_obs'].description = 'if =-1 not in eddy, if x!= obs index in obs Atlas'
f_merge['extended_dyned_obs'].description = 'Eddy Atlas observation number extended to +/- '+str(delay)+' days, if in eddy, =-1 otherwise'
f_merge['eddy_tag'].description = 'Eddy tag levels : 0 = outside eddy, 1 = Ambiguous (change value at +/- 2 days or inbetween Rmax and Rend), 2 = within max speed radius. \n Sign is eddy polarity : negative for AE, positive for CE'
f_merge['eddy_distance'].description = 'eddy center distance in kilometers, averaged over +/- 2 days, if in eddy, =-1 otherwise'


f_merge['index_dyned_obs'][:]=ctd2obs_1d
f_merge['extended_dyned_obs'][:]=ctd2obs
f_merge['eddy_distance'][:]=eddy_dist_1d
f_merge['eddy_tag'][:]=eddy_tag_pol

if track_provided:
    f_merge.createVariable('eddy_number', 'i4', 'Nprof')
    f_merge['eddy_number'].description = 'Eddy ID track number, if in eddy, =-1 otherwise'
    f_merge['eddy_number'][:]=prof_track

f_merge.close()
