#%%
"""
created on 2023-12-05

@author: Barboni Alexandre

Download new index_latest file
List available profile
Compare with existing database
Download new profiles

"""
import netCDF4 as nc4 
import numpy as np
import datetime as dt
import os
from tools_dyco import unique_prof

path_work='./'
file_0='Dataset_v0.nc'
file_1='Dataset_v1.nc'
file_merged='Dataset_v0+1.nc'


# -


#%%

### Previous data 
f0 = nc4.Dataset(path_work+file_0,'r', format='NETCDF4')
N0=f0.dimensions['Nprof'].size
depth=f0['Depth_ref'][:]
time0=f0['Days'][:]
x0=f0['Longitude'][:]
y0=f0['Latitude'][:]
file0=f0['file_name'][:]
T0=f0['gridded_temp'][:]
S0=f0['gridded_sal'][:]
PT0=f0['gridded_ptemp'][:]
profid0=np.zeros(N0).astype(str)
for i in range(N0):   ### recomputing prof Id for latest file
    profid0[i]=unique_prof(time0[i], x0[i],y0[i])
Sigma0=f0['gridded_sigma_pot'][:]

f1 = nc4.Dataset(path_work+file_1,'r', format='NETCDF4')
N1=f1.dimensions['Nprof'].size
time1=f1['Days'][:]
x1=f1['Longitude'][:]
y1=f1['Latitude'][:]
file1=f1['file_name'][:]
T1=f1['gridded_temp'][:]
S1=f1['gridded_sal'][:]
PT1=f1['gridded_ptemp'][:]
profid1=np.zeros(N1).astype(str)
for i in range(N1):   ### recomputing prof Id for latest file
    profid1[i]=unique_prof(time1[i], x1[i],y1[i])
#=f1['prof_id'][:]   ## Adding also prof_id
Sigma1=f1['gridded_sigma_pot'][:]

#%% Checking profile not already ther with Prof ID string
select=~np.in1d(profid1, profid0)
print('New profiles : '+str(len(np.where(select)[0]))+' out of '+str(len(profid1)))

time_m=np.array(list(time0)+list(time1[select]))
x_m=np.array(list(x0)+list(x1[select]))
y_m=np.array(list(y0)+list(y1[select]))
file_m=np.array(list(file0)+list(file1[select]))
T_m=np.array(list(T0)+list(T1[select]))
S_m=np.array(list(S0)+list(S1[select]))
PT_m=np.array(list(PT0)+list(PT1[select]))
Sigma_m=np.array(list(Sigma0)+list(Sigma1[select]))
profid_m=np.array(list(profid0)+list(profid1[select]))

#%%
### New merged file
print('Creating merged file :'+file_merged)
f = nc4.Dataset(path_work+file_merged,'w', format='NETCDF4')
f.description = "Profiles from different sources merged per group"

f.createDimension('Nprof', len(x_m))
f.createDimension('grid_depth', len(depth))

lon = f.createVariable('Longitude', 'f4', 'Nprof')
lat = f.createVariable('Latitude', 'f4', 'Nprof')
time  = f.createVariable('Days', 'f4', 'Nprof')

cast_name = f.createVariable('cast_name' , 'S4', 'Nprof')
file_name = f.createVariable('file_name', 'S4', 'Nprof')
depth_nc = f.createVariable('Depth_ref', 'i4', 'grid_depth')

Temp_interp = f.createVariable('gridded_temp', 'f4', ('Nprof','grid_depth') )
Sal_interp = f.createVariable('gridded_sal', 'f4', ('Nprof','grid_depth') )
PTemp_interp = f.createVariable('gridded_ptemp', 'f4', ('Nprof','grid_depth') )
Sigma_interp = f.createVariable('gridded_sigma_pot', 'f4', ('Nprof','grid_depth') )

idprof = f.createVariable('prof_id' , 'S4', 'Nprof')

#%%
lon.units = 'degrees East' 
lat.units = 'degrees North'
time.units = 'days since 2000-01-01'
cast_name.units = 'Cast name'

depth_nc.units = 'Reference depth vector for interpolation, in meters'
Temp_interp.units = 'In situ Temperature, in degC, interpolated on depth_ref vector'
Sal_interp.units = 'Practical Salinity, in PSU, interpolated on depth_ref vector'
PTemp_interp.units = 'Potential Temperature, in degC, interpolated on depth_ref vector'
Sigma_interp.units = 'Potential density -1000, in kg/m³, interpolated on depth_ref vector'

lon[:]=x_m[np.argsort(time_m)]
lat[:]=y_m[np.argsort(time_m)]
time[:]=time_m[np.argsort(time_m)]
file_name[:]=file_m[np.argsort(time_m)]
depth_nc[:]=depth

Temp_interp[:]=T_m[np.argsort(time_m)]
Sal_interp[:]=S_m[np.argsort(time_m)]
PTemp_interp[:]=PT_m[np.argsort(time_m)]
Sigma_interp[:]=Sigma_m[np.argsort(time_m)]

idprof[:]=profid_m[np.argsort(time_m)]
f.description = 'In situ database \n Updated '+dt.date.today().strftime('%d/%m/%Y')
f.close()


