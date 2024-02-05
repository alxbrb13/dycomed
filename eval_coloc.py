# +
import os
import numpy as np
import netCDF4 as nc4
from tqdm import tqdm
from tools_dyned_these import  distance
#import gsw
import matplotlib.pyplot as plt

recomputation =False ## If background already in netCDF and simply recomputation
Keys=['original','smart']
region ='MED' ## ARA or MED
path_work='/media/alexandre/HDD/Documents/These/OceanData/Cruise Data/'

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
    year_max=5
    occur500=min_nb_prof ; occur700=10 ; occur_bottom=5
    occur500_sal=20
    netcdf_data='DYCOMED_2000-2021_v3_DT+NRT.nc'

# +

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
