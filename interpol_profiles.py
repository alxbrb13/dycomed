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
import datetime as dt
from matplotlib import cm, ticker, rcParams, colors
from tqdm import tqdm
import os
# import scipy.ndimage as nd
import scipy.interpolate as interp

from tools_dyco import str2juld, interp_local_z
import matplotlib.path as mpp

rcParams['pcolor.shading']='auto' ; rcParams['contour.negative_linestyle'] = 'solid' 


time_start='20150101'
time_end='20181231'
CoorLon=[-6,36] ; CoorLat=[30,46]  ### Default : Med Sea
polygon =False ## #Optional : provide a polygon for more accurate selection
if polygon:
    poly_box=np.array([[42,5],[78,5],[78,26],[42,26],[42,5]])
    path_box=mpp.Path(poly_box, closed=True)


path_prof='./INSITU_DATA_singlefiles/'  ## folder where original files where downloaded

main_var='PSAL'  ## Main variables of interest : data at each original grid cell is not considered if this variable is not present.

ref_date='19500101' ### Reference date of the ORIGINAL files (Default = CORA with  1950-1-1)
time_offset=str2juld('20000101')-str2juld(ref_date)  ### Difference in julian days between 2000-1-1 (ref for output dataset) and downloaded files

min_depth=700 ### minimal depth to be reached by a profile
dz=1
z_i=np.arange(5,2000,dz)  ### common interpolation vertical grid

newfile='./Dataset_v0.nc'


#%%

List=np.sort(os.listdir(path_prof+'/'))

Time=[] ; Lat=[] ; Lon=[] ;
Lat_rej=[] ; Lon_rej=[] ; File_rej=[]
File_name=[]

Var_stack=[] ; Temp_stack=[] ; Psal_stack=[]

for name in tqdm(List):
    f=nc4.Dataset(path_prof+'/'+name)
    
    Psal_str='PSAL' ; Temp_str='TEMP' ; Pres_str='PRES'
    if 'TIME' in f.variables.keys():
        Time_str='TIME'
    else:
        Time_str='JULD'
        
    ### Filter on location
    if polygon:
        Filter_reg=path_box.contains_points(np.array([f['LONGITUDE'][:],f['LATITUDE'][:]]).T)
    else:
        Filter_reg=np.array([True]*len(f[Time_str]))
    Filter_date=(f[Time_str][:] >= time_offset + str2juld(time_start)) & (f[Time_str][:] <= time_offset + str2juld(time_end))
    if np.any(Filter_reg & Filter_date):
        print(name+' profiles in region and time period of interest : '+str(len(np.where(Filter_reg & Filter_date)[0])))

        Pres_init=f[Pres_str][Filter_reg & Filter_date]
        Pres_QC=f[Pres_str+'_QC'][Filter_reg & Filter_date]
        Var_init=f[main_var][Filter_reg & Filter_date]
        Var_QC=f[main_var+'_QC'][Filter_reg & Filter_date]
        Temp_init=f[Temp_str][Filter_reg & Filter_date]
        Psal_init=f[Psal_str][Filter_reg & Filter_date]
        Temp_QC=f[Temp_str+'_QC'][Filter_reg & Filter_date]
        Psal_QC=f[Psal_str+'_QC'][Filter_reg & Filter_date]
        
        ### First filter on quality control
        Var_QC=Var_QC.astype(str) ; Temp_QC=Temp_QC.astype(str) ; Psal_QC=Psal_QC.astype(str) ; Pres_QC=Pres_QC.astype(str)
        Filter_QC=(Var_QC!='1') & (Var_QC!='2') & (Var_QC!='3')
        Var_init[Filter_QC]=np.nan
        Pres_init[(Pres_QC!='1') & (Pres_QC!='2')]=np.nan
        Temp_init[(Temp_QC!='1') & (Temp_QC!='2')]=np.nan
        Psal_init[(Psal_QC!='1') & (Psal_QC!='2')]=np.nan
        
        if len(Var_init)>0:
            ### Removing column wihtout any value
            Filter_any_value=(np.sum(~Var_init.mask & ~np.isnan(Var_init),axis=1)>0) & (np.max(Pres_init,axis=1)>min_depth)
            
            Pres_init=Pres_init[Filter_any_value]
            
            
            Var_init=Var_init[Filter_any_value]
            Temp_init=Temp_init[Filter_any_value]
            Psal_init=Psal_init[Filter_any_value]
            
            N0=len(Var_init)
            Time+=list(f[Time_str][Filter_reg & Filter_date][Filter_any_value])
            Lat+=list(f['LATITUDE'][Filter_reg & Filter_date][Filter_any_value])
            Lon+=list(f['LONGITUDE'][Filter_reg & Filter_date][Filter_any_value])
            File_name+=[name[:-3]]*N0
            
            for p in range(N0):
                Var_stack+=[list(interp_local_z(Pres_init[p],Var_init[p],z_i,dz))]
                Temp_stack+=[list(interp_local_z(Pres_init[p],Temp_init[p],z_i,dz))]
                Psal_stack+=[list(interp_local_z(Pres_init[p],Psal_init[p],z_i,dz))]
                
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
Var_stack=np.array(Var_stack)
Temp_stack=np.array(Temp_stack)
Psal_stack=np.array(Psal_stack)


#%%
SA_prof = gsw.SA_from_SP(Psal_stack,np.repeat(z_i[np.newaxis,:],len(Psal_stack),axis=0), 
                         np.repeat(Lon[:, np.newaxis],len(z_i),axis=1), np.repeat(Lat[:, np.newaxis],len(z_i),axis=1))
                        
#Conservative temp
CT_prof = gsw.CT_from_t(SA_prof, Temp_stack, np.repeat(z_i[np.newaxis,:],len(Psal_stack),axis=0))
#PotTemp
PT_prof = gsw.pt0_from_t(SA_prof, Temp_stack, np.repeat(z_i[np.newaxis,:],len(Psal_stack),axis=0))
Sigma_prof=gsw.density.sigma0(SA_prof,CT_prof)

#%%
plt.figure(0,figsize=(10,7))

plt.plot(Lon_rej,Lat_rej,'or', label='No Data')
plt.plot(Lon,Lat,'og', label='Good')
plt.legend(fontsize=15) ; plt.grid()
plt.xlim(CoorLon) ; plt.ylim(CoorLat)
plt.title('region of interest profiles')
# plt.savefig(, kwargs)

#%%
#Change write mode to 'a' if you want to append a preexisting dataset
f = nc4.Dataset(newfile,'w', format='NETCDF4')
f.description = "Some in situ data in region XXXX"
f.contact ='your_mail'

f.createDimension('Nprof', len(Var_stack))
f.createDimension('grid_depth', len(z_i))

lon = f.createVariable('Longitude', 'f4', 'Nprof')
lat = f.createVariable('Latitude', 'f4', 'Nprof')
time  = f.createVariable('Time', 'f4', 'Nprof')

file_name = f.createVariable('file_name', 'S4', 'Nprof')
depth_nc = f.createVariable('depth', 'f4', 'grid_depth')

mainVar_interp = f.createVariable('dOxy', 'f4', ('Nprof','grid_depth') )
Temp_interp = f.createVariable('temp_IS', 'f4', ('Nprof','grid_depth') )
Sal_interp = f.createVariable('psal', 'f4', ('Nprof','grid_depth') )
PTemp_interp = f.createVariable('ptemp', 'f4', ('Nprof','grid_depth') )
Sigma_interp = f.createVariable('sigma_pot', 'f4', ('Nprof','grid_depth') )

#%%

lon.units = 'degrees East' 
lat.units = 'degrees North'
time.units = 'days since 2000-01-01'
depth_nc.units = 'Reference depth vector for interpolation, in meters'
Temp_interp.units = 'In situ Temperature, in degC, interpolated on depth vector'
Sal_interp.units = 'Practical Salinity, in PSU, interpolated on depth vector'
PTemp_interp.units = 'Potential Temperature, in degC, interpolated on depth_ref vector'
Sigma_interp.units = 'Potential density -1000, in kg/m³, interpolated on depth_ref vector'

lon[:]=Lon
lat[:]=Lat
time[:]=Time
file_name[:]=np.array(File_name)
depth_nc[:]=z_i

mainVar_interp[:]=Var_stack
Temp_interp[:]=Temp_stack
Sal_interp[:]=Psal_stack
PTemp_interp[:]=PT_prof
Sigma_interp[:]=Sigma_prof
f.close()
