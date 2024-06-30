#%%
"""
created on 2023-12-05

@author: Barboni Alexandre

Merge to similar datasets containing in situ data. Name of variables to be merged should be the same, and with same vertical dimensions (size and name)

Only Distinct data are kept while merging. The distinction is made with subroutine unique_prof,
Two data having same position at +/- 1 km and 10 min are considered as duplicates. 

Time in inputs is assumed as julian days (but can be integer or float). By default julian days from 2000-1-1

"""
import netCDF4 as nc4 
import numpy as np
import datetime as dt
# import os
from tools_dyco import unique_prof

path_work='./'
file_0='Dataset_v0.nc'   ### For example : CORA delayed time profiles ('cmems_obs-ins_glo_phy-temp-sal_my_cora_irr')
file_1='Dataset_v1.nc'   ### for example : CORA Near-Real time profiles ('cmems_obs-ins_med_phybgcwav_mynrt_na_irr')
file_merged='Dataset_v0+1.nc'

z_vars_to_merge=['PSAL','TEMP','sigma_pot','pot_temp']  ## variables with vertical dimensions
z_vars_type=['f4','f4','f4','f4']                 ## data type of variable to merge to be recored in ne w netcdf.
# 'f' for float, 'i' for integer, 'S' for string. Number is for data size, usually 4 is fine.
surf_vars_to_merge=['file_name']           ## variables with NO vertical dimensions
surf_vars_type=['S4']    

z_var='depth'  ## vertical coordinate

#%%
MainDict={}
DictUnits={}
### Previous data 
f0 = nc4.Dataset(path_work+file_0,'r', format='NETCDF4')
z_axis0=f0[z_var][:]
time0=f0['Time'][:]
x0=f0['Longitude'][:]
y0=f0['Latitude'][:]
for var in z_vars_to_merge + surf_vars_to_merge:
    MainDict[var+'-0']=f0[var][:]
    DictUnits[var+'-dim_name']=f0[var].dimensions
f0.close()

f1 = nc4.Dataset(path_work+file_1,'r', format='NETCDF4')
z_axis1=f1[z_var][:]
time1=f1['Time'][:]
x1=f1['Longitude'][:]
y1=f1['Latitude'][:]
for var in z_vars_to_merge + surf_vars_to_merge:
    MainDict[var+'-1']=f1[var][:]
f1.close()
#%%
if np.any(z_axis0!= z_axis1):
    raise ValueError

#%%
N0=np.shape(MainDict[z_vars_to_merge[0]+'-0'])[0]  ## number of rows, 1st file
N1=np.shape(MainDict[z_vars_to_merge[0]+'-1'])[0]  ## ________________ 2nd file

var=z_vars_to_merge[0]
surf_dim=DictUnits[var+'-dim_name'][0]
z_dim=DictUnits[var+'-dim_name'][1]

profid0=np.zeros(N0).astype(str)
for i in range(N0):   
    profid0[i]=unique_prof(time0[i], x0[i],y0[i])
profid1=np.zeros(N1).astype(str)
for i in range(N1):   
    profid1[i]=unique_prof(time1[i], x1[i],y1[i])
#%% Checking profile not already ther with Prof ID string
select=~np.in1d(profid1, profid0)
print('\n ################## \n New profiles in dataset 1 : '+str(len(np.where(select)[0]))+' out of '+str(len(profid1)))

time_m=np.array(list(time0)+list(time1[select]))
x_m=np.array(list(x0)+list(x1[select]))
y_m=np.array(list(y0)+list(y1[select]))
profid_m=np.array(list(profid0)+list(profid1[select]))

for var in z_vars_to_merge+surf_vars_to_merge:
    MainDict[var+'-m']=np.array(list(MainDict[var+'-0'])+list(MainDict[var+'-1'][select]))
#%%
### New merged file
print('Creating merged file :'+file_merged)
f = nc4.Dataset(path_work+file_merged,'w', format='NETCDF4')
f.description = "Profiles from different sources merged per group"

f.createDimension('Nprof', len(x_m))
f.createDimension('grid_depth', len(z_axis0))

lon = f.createVariable('Longitude', 'f4', surf_dim)
lat = f.createVariable('Latitude', 'f4', surf_dim)
time  = f.createVariable('Days', 'f4', surf_dim)

#### Shared dimensions
depth_nc = f.createVariable(z_var, 'f4', z_dim)

#### Merged variables
for i,var in enumerate(z_vars_to_merge):
    f.createVariable(var, z_vars_type[i], DictUnits[var+'-dim_name'])
for i,var in enumerate(surf_vars_to_merge):
    f.createVariable(var, surf_vars_type[i], DictUnits[var+'-dim_name'])
#%%
lon.units = 'degrees East' 
lat.units = 'degrees North'
time.units = 'days since 2000-01-01'

lon[:]=x_m[np.argsort(time_m)]
lat[:]=y_m[np.argsort(time_m)]
time[:]=time_m[np.argsort(time_m)]
depth_nc[:]=z_axis0

#### Merged variables
for var in z_vars_to_merge+surf_vars_to_merge:
    f[var][:]=MainDict[var+'-m'][np.argsort(time_m)]
    # f[var][:].units = DictUnits[var]
# idprof[:]=profid_m[np.argsort(time_m)]

f.description = 'In situ database \n Updated '+dt.date.today().strftime('%d/%m/%Y')
f.close()


