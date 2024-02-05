# +
"""
created on 2023-12-05

@author: Barboni Alexandre

Download new index_latest file
List available profile
Compare with existing database
Download new profiles

"""
import netCDF4 as nc4 
import h5py
from tqdm import tqdm
import numpy as np
import datetime as dt
import os

import sys
sys.path.append('/home6/datahome/abarboni/DYNED-NRT/')
from tools_dycomed import juldHour2str

path_work='/home6/datawork/abarboni/DYCOMED/PROF/'
file_ini='CORA_NRT_MED_Initial_20231204.nc'
file_latest='CORA_NRT_MED_latest'
file_merged='CORA_NRT_MED_merged'


# -

def profile_id(t,x,y,p):
    return juldHour2str(t)+'X'+'%05d'%(float(x%360)*100)+'Y'+'%05d'%(float(y)*100)+'P'+p


# +
### Find latest file
ListDir=os.listdir(path_work) ; Latest=[]

for name in ListDir:
    if name[:19]==file_latest:
        Latest+=[name]
Latest=np.sort(Latest)
name_latest=Latest[-1] ; datestr=name_latest[-11:-3] ;
print('Date of latest NRT profiles :'+datestr)

# +
### Find previous merged file
ListDir=os.listdir(path_work) ; Merged=[]

for name in ListDir:
    if name[:len(file_merged)]==file_merged:
        Merged+=[name]
Merged=np.sort(Merged) ;
if len(Merged)==0:
    print('No merged file, starting form initial : '+file_ini)
    name_merged=file_ini ; date_m=name_merged[-11:-3]
elif len(Merged)==1:
    name_merged=Merged[0] ; date_m=name_merged[-11:-3]
    print('Date of previously merged file :'+date_m)
if len(Merged)>1:
    print('Warning : more than one merged profiles datset')
# -

### Previous data 
f0 = nc4.Dataset(path_work+name_merged,'r', format='NETCDF4')
N0=f0.dimensions['Nprof'].size
depth=f0['Depth_ref'][:]
time0=f0['Days'][:]
x0=f0['Longitude'][:]
y0=f0['Latitude'][:]
cast0=f0['cast_name'][:]
file0=f0['file_name'][:]
T0=f0['gridded_temp'][:]
S0=f0['gridded_sal'][:]
PT0=f0['gridded_ptemp'][:]
profid0=f0['prof_id'][:]   ## Adding also prof_id
Sigma0=f0['gridded_sigma_pot'][:]

f1 = nc4.Dataset(path_work+name_latest,'r', format='NETCDF4')
N1=f1.dimensions['Nprof'].size
time1=f1['Days'][:]
x1=f1['Longitude'][:]
y1=f1['Latitude'][:]
cast1=f1['cast_name'][:]
file1=f1['file_name'][:]
T1=f1['gridded_temp'][:]
S1=f1['gridded_sal'][:]
PT1=f1['gridded_ptemp'][:]
profid1=np.zeros(N1).astype(str)
for i in range(N1):   ### recomputing prof Id for latest file
    profid1[i]=profile_id(time1[i], x1[i],y1[i],cast1[i])
#=f1['prof_id'][:]   ## Adding also prof_id
Sigma1=f1['gridded_sigma_pot'][:]

### Checking profile not already ther with Prof ID string
select=~np.in1d(profid1, profid0)
print('New profiles : '+str(len(np.where(select)[0]))+' out of '+str(len(profid1)))

time_m=np.array(list(time0)+list(time1[select]))
x_m=np.array(list(x0)+list(x1[select]))
y_m=np.array(list(y0)+list(y1[select]))
cast_m=np.array(list(cast0)+list(cast1[select]))
file_m=np.array(list(file0)+list(file1[select]))
T_m=np.array(list(T0)+list(T1[select]))
S_m=np.array(list(S0)+list(S1[select]))
PT_m=np.array(list(PT0)+list(PT1[select]))
Sigma_m=np.array(list(Sigma0)+list(Sigma1[select]))
profid_m=np.array(list(profid0)+list(profid1[select]))

# +
### New merged file
print('Creating merged file :'+file_merged+'_'+datestr)
f = nc4.Dataset(path_work+file_merged+'_'+datestr+'.nc','w', format='NETCDF4')
f.description = "Med profiles from different sources merged per group"

#coragrp = f.createGroup('CORA')
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

# +
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
cast_name[:]=cast_m[np.argsort(time_m)]
file_name[:]=file_m[np.argsort(time_m)]
depth_nc[:]=depth

Temp_interp[:]=T_m[np.argsort(time_m)]
Sal_interp[:]=S_m[np.argsort(time_m)]
PTemp_interp[:]=PT_m[np.argsort(time_m)]
Sigma_interp[:]=Sigma_m[np.argsort(time_m)]

idprof[:]=profid_m[np.argsort(time_m)]
f.description = 'Vertically interpolated CORA-NRT database \n Updated '+dt.date.today().strftime('%d/%m/%Y')
f.close()
# -
### Removing previous merged file
if len(Merged)>0:
    print('Removing previous merged file :'+file_merged+'_'+date_m)
    os.remove(path_work+file_merged+'_'+date_m+'.nc')

    ### Adding profile ID
#    f=nc4.Dataset(path_work+file_ini,'a')
#    #idprof = f.createVariable('prof_id' , 'S4', 'Nprof')
 #   time0=f['Days'][:]  ;  x0=f['Longitude'][:]  ;  y0=f['Latitude'][:]
  #  cast0=f['cast_name'][:]
    #name0=f['file_name'][:]
   # id0=[]
    #for i in tqdm(range(len(x0))):
     #   id0+=[profile_id(time0[i], x0[i],y0[i],cast0[i])]
    #id0=np.array(id0)
    #f['prof_id'][:]=id0
    #f.close()


