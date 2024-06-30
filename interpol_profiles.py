#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:27:40 2024

@author: alexandre
"""

import netCDF4 as nc4
import matplotlib.pyplot as plt
import numpy as np
import gsw
from tqdm import tqdm
import os
import cartopy.crs as ccrs
from tools_dyco import str2juld, interp_local_z
import matplotlib.path as mpp

#### Path & folder
path_prof='./INSITU_DATA_singlefiles/'  ## folder where original files where downloaded
newfile='./Dataset_v0.nc'        

#### Variable names
main_var='PSAL'  ## Main variables of interest : data at each original grid cell is not considered if this variable is not present.
z_var='PRES'  # variable considered as the z_axis
sal_var='PSAL'
temp_var='TEMP'
other_vars=[sal_var,temp_var] ## Minimal is to keep [sal_var, temp_var]
# other_vars=[sal_var,temp_var,'DOXY'] ## Possibility to add 'DOXY', 'NO3' , etc.

#### Reference time
time_start='20150101'  ## your data time range
time_end='20181231'
ref_date='19500101' ### Reference date of the ORIGINAL files (Default = CORA with  1950-1-1)

### Geographical selection
CoorLon=[-6,36] ; CoorLat=[30,46]  ### Default : Med Sea
polygon =False ## #Optional : provide a polygon for more accurate selection
if polygon:
    poly_box=np.array([[42,5],[78,5],[78,26],[42,26],[42,5]])
else:
    poly_box=np.array([[CoorLon[0],CoorLat[0]],[CoorLon[1],CoorLat[0]],[CoorLon[1],CoorLat[1]],[CoorLon[0],CoorLat[1]],[CoorLon[0],CoorLat[0]]])
    path_box=mpp.Path(poly_box, closed=True)

#### Vertical grid
min_depth=400 ### minimal depth to be reached by a profile
dz=2
z_i=np.arange(5,1000,dz)  ### common interpolation vertical grid
interpol_mode='linear'          ### interpolation mode : 'gap' = grid step in z_i with no value are kept as nan
                                               #  other  = grid step in z_i with no value are interpolated, using interp.interp1d(kind=interpol_mode)

#%%  Interpolation file by file

List=np.sort(os.listdir(path_prof+'/'))
if other_vars==['']:
    vars_list=list(np.unique([main_var,z_var,sal_var,temp_var]))
    extract_list=list(np.unique([main_var,sal_var,temp_var]))
else:
    vars_list=list(np.unique([main_var,z_var,sal_var,temp_var]+other_vars))
    extract_list=list(np.unique([main_var,z_var,sal_var,temp_var]+other_vars))

time_offset=str2juld('20000101')-str2juld(ref_date)  ### Difference in julian days between 2000-1-1 (ref for output dataset) and downloaded files

Time=[] ; Lat=[] ; Lon=[] ;
Lat_rej=[] ; Lon_rej=[] ; File_rej=[]
File_name=[]

# Var_stack=[] ;
Vars_dict={}
for var in vars_list:
    Vars_dict[var+'-stack']=[]

for name in tqdm(List):
    f=nc4.Dataset(path_prof+'/'+name)
    
    if 'TIME' in f.variables.keys(): ### 2 possibilities for time variable name
        Time_str='TIME'
    else:
        Time_str='JULD'
        
    ### Filter on location & time
    if polygon:
        Filter_reg=path_box.contains_points(np.array([f['LONGITUDE'][:],f['LATITUDE'][:]]).T)
    else:
        Filter_reg=np.array([True]*len(f[Time_str]))
    Filter_date=(f[Time_str][:] >= time_offset + str2juld(time_start)) & (f[Time_str][:] <= time_offset + str2juld(time_end))
    if np.any(Filter_reg & Filter_date):
        print(name+' profiles in region and time period of interest : '+str(len(np.where(Filter_reg & Filter_date)[0])))

        for var in vars_list:
            Vars_dict[var+'-init']=f[var][Filter_reg & Filter_date]
            Vars_dict[var+'-QC']=(f[var+'_QC'][Filter_reg & Filter_date]).astype(str)

        ### Filter on quality control
        Filter_QC=(Vars_dict[main_var+'-QC']!='1') & (Vars_dict[main_var+'-QC']!='2') & (Vars_dict[main_var+'-QC']!='3')
        Vars_dict[main_var+'-init'][Filter_QC]=np.nan
        for var in [z_var]+other_vars:
            Vars_dict[var+'-init'][(Vars_dict[var+'-QC']!='1') & (Vars_dict[var+'-QC']!='2')]=np.nan

        ### Ensuring there is at least one valid data
        if len(Vars_dict[main_var+'-init'])>0:
            ### Removing column wihtout any value
            Filter_any_value=(np.sum(~Vars_dict[main_var+'-init'].mask & ~np.isnan(Vars_dict[main_var+'-init']),axis=1)>0) & (np.max(Vars_dict[z_var+'-init'],axis=1)>min_depth)
            for var in vars_list:
                Vars_dict[var+'-init']=Vars_dict[var+'-init'][Filter_any_value]
            
            N0=len(Vars_dict[main_var+'-init'])
            Time+=list(f[Time_str][Filter_reg & Filter_date][Filter_any_value])
            Lat+=list(f['LATITUDE'][Filter_reg & Filter_date][Filter_any_value])
            Lon+=list(f['LONGITUDE'][Filter_reg & Filter_date][Filter_any_value])
            File_name+=[name[:-3]]*N0
            
            for p in range(N0):
                for var in extract_list:
                    Vars_dict[var+'-stack']+=[list(interp_local_z(Vars_dict[z_var+'-init'][p],Vars_dict[var+'-init'][p],z_i,dz,interpol_mode=interpol_mode))]
 
            ## Listing rejected
            Nrej=len(np.where(~Filter_any_value)[0])
            Lat_rej+=list(f['LATITUDE'][Filter_reg & Filter_date][~Filter_any_value])
            Lon_rej+=list(f['LONGITUDE'][Filter_reg & Filter_date][~Filter_any_value])
            File_rej+=[name[:-3]]*Nrej
            print('\n Profiles rejected : '+str(Nrej))
        else: ## In case all are rejected
            Nrej=len(Filter_QC)
            Lat_rej+=list(f['LATITUDE'][Filter_reg & Filter_date])
            Lon_rej+=list(f['LONGITUDE'][Filter_reg & Filter_date])
            File_rej+=[name[:-3]]*Nrej
            print('\n Profiles rejected : '+str(Nrej))
            
    f.close()
    
Time=np.array(Time)-time_offset ## in days since 2000-1-1
Lat=np.array(Lat) ; Lon=np.array(Lon)
Lat_rej=np.array(Lat_rej) ; Lon_rej=np.array(Lon_rej)
for var in extract_list:
    Vars_dict[var+'-stack']=np.array(Vars_dict[var+'-stack'])


#%% Computing state variable
N1=np.shape(Vars_dict[sal_var+'-stack'])[0]
SA_prof = gsw.SA_from_SP(Vars_dict[sal_var+'-stack'],np.repeat(z_i[np.newaxis,:],N1,axis=0), 
                         np.repeat(Lon[:, np.newaxis],len(z_i),axis=1), np.repeat(Lat[:, np.newaxis],len(z_i),axis=1))
                        
#Conservative temp
CT_prof = gsw.CT_from_t(SA_prof, Vars_dict[temp_var+'-stack'], np.repeat(z_i[np.newaxis,:],N1,axis=0))
#PotTemp
PT_prof = gsw.pt0_from_t(SA_prof, Vars_dict[temp_var+'-stack'], np.repeat(z_i[np.newaxis,:],N1,axis=0))
Sigma_prof=gsw.density.sigma0(SA_prof,CT_prof)

#%% Small map
fig=plt.figure(0,figsize=(10,7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

plt.plot(Lon_rej,Lat_rej,'or', label='No Data')
plt.plot(Lon,Lat,'og', label='Good')
plt.legend(fontsize=15) ; plt.grid()
ax.coastlines()

plt.xlim(CoorLon) ; plt.ylim(CoorLat)
plt.title('region of interest profiles')
# plt.savefig(, kwargs)

#%% Writing 2D netcdf

#Change write mode to 'a' if you want to append a preexisting dataset
f = nc4.Dataset(newfile,'w', format='NETCDF4')
f.description = "Some in situ data in region XXXX"
f.contact ='your_mail'

f.createDimension('Nprof', N1)
f.createDimension('grid_depth', len(z_i))

lon = f.createVariable('Longitude', 'f4', 'Nprof')
lat = f.createVariable('Latitude', 'f4', 'Nprof')
time  = f.createVariable('Time', 'f4', 'Nprof')

file_name = f.createVariable('file_name', 'S4', 'Nprof')
depth_nc = f.createVariable('depth', 'f4', 'grid_depth')

for var in extract_list:
    f.createVariable(var, 'f4', ('Nprof','grid_depth') )
### Extra computed var
PTemp_interp = f.createVariable('pot_temp', 'f4', ('Nprof','grid_depth') )
Sigma_interp = f.createVariable('sigma_pot', 'f4', ('Nprof','grid_depth') )

#%%

lon.units = 'degrees East' 
lat.units = 'degrees North'
time.units = 'days since 2000-01-01'
depth_nc.units = 'Reference depth vector for interpolation, in meters'
file_name.units = 'original single file name'
f[temp_var].units = 'In situ Temperature, in degC, interpolated on depth vector'
f[sal_var].units = 'Practical Salinity, in PSU, interpolated on depth vector'
PTemp_interp.units = 'Potential Temperature, in degC, interpolated on depth_ref vector'
Sigma_interp.units = 'Potential density -1000, in kg/m³, interpolated on depth_ref vector'

lon[:]=Lon
lat[:]=Lat
time[:]=Time
file_name[:]=np.array(File_name)
depth_nc[:]=z_i

for var in extract_list:
    f[var][:]=Vars_dict[var+'-stack']
PTemp_interp[:]=PT_prof
Sigma_interp[:]=Sigma_prof
f.close()
