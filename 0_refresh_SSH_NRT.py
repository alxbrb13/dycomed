#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Apr 16 14:28:48 2021

@author: alexandre

This script contactenates previously downloaded NRT AVISO SSH in a common netCDF file

Option refresh to refresh the last reprocessed SSH files

v2: append to existing netCDF file instead of creating a new one
    takes into account NRT tracks : 'optimal' mask more restricted
            
"""
import os
import shutil
import urllib.request as request
import netCDF4 as nc4
import h5py
import scipy.ndimage as nd
from contextlib import closing
# from matplotlib import rcParams
import datetime as dt
from ftplib import FTP
import matplotlib.pyplot as plt
# # import gsw
from tqdm import tqdm
import numpy as np

import sys
sys.path.append('/home6/datahome/abarboni/DYNED-NRT/DYCOMED_scripts/')
import secretcodes
mdp=secretcodes.cmems_password
user=secretcodes.cmems_username

refresh=True
# year=2023
#date_update='20210326'  ## date up to which ADT is updated
date_update=dt.datetime.strftime(dt.date.today(),'%Y%m%d')

refresh_delay=7        ## index of last previous file refreshed
## Selection criteria 
# CoorLon=[-6,36] ; CoorLat=[30,46] ### Mediterranean
# CoorLon=[15,36] ; CoorLat=[30,40] ### East Mediterranean
# CoorLon=[30,78] ; CoorLat=[0,32]  ## Arabian sea (initial 1/8 product)
# CoorLon=[43,78] ; CoorLat=[0,26]   ### 'Arabian sea without Persic gulf and red sea'
#CoorLon=[-1,10];CoorLat=[36,45]    ## Western Mediterranean
#CoorLon=[10,25];CoorLat=[30,41] ## Ionian Sea
CoorLon=[-6,36]; CoorLat=[30,46]

region_long = 'Med Sea'
region='MED'
new_filename='_time_merged_L4_NRT_'+region+'_'
variables=['adt','ugos','vgos'] ## variables to be merged in 1 file

## Folder with all raw NRT daily files
pathwork='/home6/datawork/abarboni/DYCOMED/SSH_18_MED/DATA/'
#name_local='adt_uv_geos_duacs_nrt_allsat_l4_'   ### Name on local folder
name_local='nrt_europe_l4_crop_'

distant_path='/Core/SEALEVEL_EUR_PHY_L4_NRT_OBSERVATIONS_008_060/dataset-duacs-nrt-europe-merged-allsat-phy-l4/'

# %% ############### END USER DEFINED VARIABLES #############################
is0=25 ## index in string beginning date in ftp name
isF=33 ## index in string ending date
is1=len(name_local)## index in string beginning date in local name
is2=is1+8 ## index in string ending date

ListDir=np.sort(os.listdir(pathwork))
dates=[]
if len(ListDir)>0:
    for s in ListDir:
        dates+=[s[is1:is2]]
    dates=np.array(dates)
    date_before=ListDir[-1][is1:is2]
    print('Previous file found, last date : '+date_before) 
else:
    date_before='20230101'
    print('No previous files, updating from default : '+date_before)
# %% ############### END USER DEFINED VARIABLES #############################
date0=dt.datetime.strptime(date_before,'%Y%m%d')
dateF=dt.datetime.strptime(date_update,'%Y%m%d')

if date0 >= dateF:
    print('Chosen date already updated')
elif (date_update!=date_before) & refresh:

    DeltaT=(dateF-date0).days
    date_to_be_updated=[]
    for t in np.arange(-refresh_delay+1,DeltaT+1):
        date_to_be_updated+=[(date0+dt.timedelta(days=int(t))).strftime('%Y%m%d')]

    month_select=[]
    for date in date_to_be_updated:
        month_select+=[date[:6]]
        #year_select+=[date[:4]]
    month_select=np.unique(month_select) 

# %% ############### END USER DEFINED VARIABLES #############################
###############################################
## Downloading nrt daily file per ftp and cropping it to the MED sea only
    names_select=[]
    #for y in year_select:
    for m in month_select:
        print(' \n YEAR = '+m[:4]+' - MONTH = '+m[4:])

        ftp=FTP('nrt.cmems-du.eu')   
        ftp.login(user=user, passwd=mdp)
        ftp.cwd(distant_path+m[:4]+'/'+m[4:]+'/')
        names=ftp.nlst()  ### Listing available data in ftp directory

        for s in names:
            if s[is0:isF] in date_to_be_updated:
                # names_select+=[s]
                names_select+=[name_local+s[is0:]]  ## string for merging, in local names

                postdate=(dt.datetime.strptime(s[isF+1:isF+9],'%Y%m%d')-dt.datetime.strptime(s[is0:isF],'%Y%m%d')).days


                already=np.where(dates==s[is0:isF])[0]
                if len(already)>0:  #if s in ListDir:
                    str_to_delete=ListDir[already[0]]  ### <== assumes here only one file to delete !
                    print(region+' : Removing '+str_to_delete)   ## deleting previous file, in order to avoid 2 versions 
                    os.remove(pathwork+str_to_delete)
                    print(region+' : Updating : '+s[is0:isF]+' - day+'+str(postdate))  
                else:
                    print(region+' : New file : '+s[is0:isF]+' - day+'+str(postdate))

                with closing(request.urlopen('ftp://abarboni1:cmemsNico13@@nrt.cmems-du.eu:'+distant_path+m[:4]+'/'+m[4:]+'/'+s)) as r:
                    with open(pathwork+'/'+s, 'wb') as f:
                        shutil.copyfileobj(r, f)

                ##### Cropping + renaming
                os.system('ncks -O -d latitude,'+str(CoorLat[0])+'.0,'+str(CoorLat[1])+'.0 -d longitude,'+str(CoorLon[0])+'.0,'+str(CoorLon[1])+'.0 -v adt,ugos,vgos '+pathwork+s+' '+pathwork+name_local+s[is0:])
                #os.system('cp '+pathwork+name_local+s[is0:]+' /home6/scratch/abarboni/eftp/RAWDATA/DUACS/'+name_local+s[is0:])
                os.remove(pathwork+s)
# %%
#for s in tqdm(ListDir):
  #  os.system('ncks -O -d latitude,'+str(CoorLat[0])+'.0,'+str(CoorLat[1])+'.0 -d longitude,'+str(CoorLon[0])+'.0,'+str(CoorLon[1])+'.0 -v adt,ugos,vgos '+pathwork+s+' '+pathwork+name_local+s[is0:])
    #os.remove(pathwork+s)

# %% ############### END USER DEFINED VARIABLES #############################
######################## 
ListDir=np.sort(os.listdir(pathwork))  ## re-listing pathwork after update
Nf=len(ListDir)

info=[] ; units=[]
f0=nc4.Dataset(pathwork+ListDir[0],'r')
lat1=f0['latitude'][:] ; lon1=f0['longitude'][:]

adt0=f0['adt'][0]
Nx = len(lon1) ; Ny = len(lat1)
for var in variables:
    info+=[f0[var].standard_name]
    units+=[f0[var].units]
f0.close()
# %% ################ Creating grid file for ADT
#### Creating grid file with mask (re-done each time because not always the same in NRT data)
f_mask = nc4.Dataset(pathwork+'../grid'+new_filename+'.nc','w', format='NETCDF4')
f_mask.title ='grid parameter in region '+region_long
f_mask.contact = 'alexandre.barboni@lmd.ipsl.fr'
f_mask.description = 'AVISO DUACS L4 NRT SSH land mask'

f_mask.createDimension('lon', Nx)
f_mask.createDimension('lat', Ny)
lon_nc = f_mask.createVariable('lon', 'f4', 'lon')
lat_nc = f_mask.createVariable('lat', 'f4', 'lat')
mask_nc  = f_mask.createVariable('mask', 'i4', ('lat','lon'))
mask2_nc  = f_mask.createVariable('mask_min', 'i4', ('lat','lon'))
mask2_nc.description = 'minimal mask (D+6 data)'

Mtemp=np.zeros((Ny,Nx))
Mmin=np.zeros((Ny,Nx))
print('Checking NRT mask')
f=nc4.Dataset(pathwork+ListDir[-7],'r', format='NETCDF4')  # minimal mask for D+6 data
Mmin=(f['adt'][0].mask).astype(int)
f.close()
for j,name in enumerate(tqdm(ListDir[-7:])):
    f=nc4.Dataset(pathwork+name,'r', format='NETCDF4')  # sometime more masked values
    Mtemp+=(f['adt'][0].mask).astype(int)
    Mtemp+=(f['ugos'][0].mask).astype(int)
    Mtemp+=(f['vgos'][0].mask).astype(int)
    f.close()

# %%
###  Showing only the MED sea (discarding black sea)
Mbig=(Mtemp==0).astype(int)
Seas=Mbig.astype(bool)

M,Nmax=nd.label(Seas)  ## Finding connected sea surfaces
Sizes=[len(Seas[M==x]) for x in range(1,Nmax)]  ## Identifying biggest sea
Mask=np.ones(np.shape(Seas))
Mask[M==np.argmax(Sizes)+1]=0
Mask=Mask.astype(bool)

Mbig2=(Mmin==0).astype(int)
Seas2=Mbig2.astype(bool)
M,Nmax=nd.label(Seas2)  ## Finding connected sea surfaces
Sizes=[len(Seas2[M==x]) for x in range(1,Nmax)]  ## Identifying biggest sea
Mask2=np.ones(np.shape(Seas2))
Mask2[M==np.argmax(Sizes)+1]=0
Mask2=Mask2.astype(bool)

#plt.subplot(211)
#plt.imshow(Mask) ;
#plt.subplot(212)
#plt.imshow(Mask2) ; plt.colorbar()

# %%
mask_nc[:]=Mask ##(~adt0.mask).astype(int)
mask2_nc[:]=Mask2
lon_nc[:]=lon1 ; lat_nc[:]=lat1
f_mask.close()

# %%
#### Flagging Outside Med Sea
for name in tqdm(names_select):#tqdm(ListDir):
    f=nc4.Dataset(pathwork+name,'a')
    A=f['adt'][0]
    A[Mask2]=np.nan
    B=f['ugos'][0]
    B[Mask2]=np.nan
    C=f['vgos'][0]
    C[Mask2]=np.nan
    f['adt'][0]=A ; f['ugos'][0]=B ; f['vgos'][0]=C
    #plt.imshow(f['adt'][0])
    f.close()


# %% Checking previous merged file
##### Updating previous files
if date0 >= dateF:
    print('Chosen date already updated')
elif (date_update!=date_before) & refresh:
    
    

    Listmerge=os.listdir(pathwork+'../') ; Listgood=[]
    for s_m in Listmerge:
        if s_m[-23-len(new_filename):-len(new_filename)+1]==new_filename:
            Listgood+=[s_m]   
    if len(Listgood)>0:
        print('Updating existing merged file')
        last_date='20230101' ; date_base='20230101'  ## assuming already at least one file in 2023
        for s_m in Listgood:
            if int(last_date)<int(s_m[-11:-3]):
                last_date=s_m[-11:-3] ; date_base=s_m[-23:-15]  ## looking for bounds date
    

    #%%#################### Appending previous common NetCDF file with separated variable
        print('Appending '+str(len(names_select))+' new days')
        for i,var in enumerate(variables):
            print('Merging daily files for '+var)
            f_merge = nc4.Dataset(pathwork+'../'+var+new_filename+date_base+'_to_'+last_date+'.nc','a', format='NETCDF4')
            date_already=f_merge['date_str'][:] ; update=0
            for j,name in enumerate(tqdm(names_select)):
                if date_to_be_updated[j] in date_already:
                    idj=np.where(date_already==date_to_be_updated[j])[0][0]
                    update+=1 ## update counts how many indexes are already there
                else:
                    idj=len(date_already)-update+j
                f=nc4.Dataset(pathwork+name,'r', format='NETCDF4')
                f_merge[var][idj]=f[var][0]
                f_merge['date_str'][idj]=name[is1:is2]
                f_merge['time'][idj]=f['time'][0]-(dt.date(2000,1,1)-dt.date(1950,1,1)).days
                f.close()
            f_merge.close()
            ### Updating bounds date
            os.rename(pathwork+'../'+var+new_filename+date_base+'_to_'+last_date+'.nc', pathwork+'../'+var+new_filename+date_base+'_to_'+date_update+'.nc')
  
    #%%#################### If not already there, creating common NetCDF file 
           
    else:
        for i,var in enumerate(variables):
            print('Merging daily files for '+var)
            # str0=pathwork+'../adt_time_merged_L4_NRT_ARA20230101_to_20230202.nc'
            f_merge = nc4.Dataset(pathwork+'../'+var+new_filename+ListDir[0][is1:is2]+'_to_'+ListDir[-1][is1:is2]+'.nc','w', format='NETCDF4')
            f_merge.title = var+ ' data in region '+region
            f_merge.contact = 'alexandre.barboni@shom.fr'
            f_merge.description = 'AVISO DUACS L4 NRT SSH at 1/8 degree in region '+region
            
            f_merge.createDimension('lon', size=0)
            f_merge.createDimension('lat', size=0)
            f_merge.createDimension('time', size=0)
            
            lon_nc = f_merge.createVariable('lon', 'f4', 'lon')
            lat_nc = f_merge.createVariable('lat', 'f4', 'lat')
            time_nc  = f_merge.createVariable('time', 'f4', 'time')
            date_nc = f_merge.createVariable('date_str', 'S4', 'time')
            VAR  = f_merge.createVariable(var, 'f4', ('time','lat','lon'))
            
            ### Meta data information
            lon_nc.units = 'degrees East' 
            lat_nc.units = 'degrees North'
            time_nc.units = 'days since 2000-1-1. Convention : 0.000 = 2000-01-01 at 0h00'
            date_nc.units = 'date string'
            VAR.units=units[i]
            VAR.comment=info[i]
            
            # Filling netcdf
            lon_nc[:]=lon1
            lat_nc[:]=lat1
            for j,name in enumerate(tqdm(ListDir)):
                f=nc4.Dataset(pathwork+name,'r', format='NETCDF4')
                VAR[j]=f[var][0]
                date_nc[j]=name[is1:is2]
                time_nc[j]=f['time'][0]-(dt.date(2000,1,1)-dt.date(1950,1,1)).days
                f.close()
            f_merge.close()

# %%
