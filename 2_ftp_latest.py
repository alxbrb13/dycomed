#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
created on 2023-12-05

@author: Barboni Alexandre

Download new index_latest file
List available profile
Compare with existing database
Download new profiles

"""
## Importing some standard packages
import matplotlib.pyplot as plt
import csv
import shutil
import urllib.request as request
from contextlib import closing

import netCDF4 as nc4 
from tqdm import tqdm
import numpy as np
from matplotlib import cm
from ftplib import FTP
from matplotlib import rcParams
import datetime as dt
## Optional
#import gsw
import os
import secretcodes
mdp=secretcodes.cmems_password
user=secretcodes.cmems_username

plt.style.use(u'default')
rcParams['contour.negative_linestyle'] = 'solid' 
header=6   ##↕ header size in index_latest.txt

## Paths to set
path_work='/home6/datawork/abarboni/DYCOMED/PROF/'
dir_down='LATEST_'+dt.datetime.strftime(dt.date.today(),format='%Y%m%d')
path_down=path_work+dir_down
file_merged='CORA_NRT_MED_merged'

distant_path='/Core/INSITU_MED_PHYBGCWAV_DISCRETE_MYNRT_013_035/cmems_obs-ins_med_phybgcwav_mynrt_na_irr/'

Datatype=['CT','GL','PF','XB']
CoorLon=[-6,37] ; CoorLat=[30,46]

## Legend = #catalog_id,file_name,geospatial_lat_min,geospatial_lat_max,geospatial_lon_min,geospatial_lon_max,
  ##time_coverage_start,time_coverage_end,institution,date_update,data_mode,parameters

# %%
if dir_down not in os.listdir(path_work):
    os.mkdir(path_down)

# %%
#### Downloading index_lastest.txt
ftp=FTP('nrt.cmems-du.eu')   
ftp.login(user=user, passwd=mdp)
ftp.cwd(distant_path)
with closing(request.urlopen('ftp://'+user+':'+mdp+'@nrt.cmems-du.eu:'+distant_path+'index_latest.txt')) as r:
    with open(path_work+'index_latest.txt', 'wb') as f:
        shutil.copyfileobj(r, f)

# %% Loading metadata
paths=[]; Datestr=[] ; Lati=[] ; Latf=[] ; Loni=[] ; Lonf=[] ; platforms=[] ; types =[]# ; profs=[]
with open(path_work+'index_latest.txt', newline='') as csvfile:
    fileread = csv.reader(csvfile, delimiter=',') 
    for x in range(header):
        next(fileread)  ## Skipping first line
    for row in fileread :
        paths+=[row[1]] ; Loni+=[row[4]] ;  Lonf+=[row[5]]  ; Lati+=[row[2]]  ; Latf+=[row[3]] 
        Datestr+=[row[6][:4]+row[6][5:7]+row[6][8:10]]
        #profs+=[row[1].split('/')[-1].split('_')[1]]
        types+=[row[1].split('/')[-1].split('_')[2]]
        platforms+=[row[1].split('/')[-1].split('_')[3]]
Loni=np.array(Loni).astype(float)
Lonf=np.array(Lonf).astype(float)
Lati=np.array(Lati).astype(float)
Latf=np.array(Latf).astype(float)
types=np.array(types).astype(str)
paths=np.array(paths).astype(str)
platforms=np.array(platforms).astype(str)
Datestr=np.array(Datestr).astype(str)
N0=len(paths)

# %%
select_loc=((Loni>CoorLon[0])+ (Lonf>CoorLon[0])) & ((Loni<CoorLon[1])+(Lonf<CoorLon[1])) & ((Lati>CoorLat[0])+ (Latf>CoorLat[0])) & ((Lati<CoorLat[1])+(Latf<CoorLat[1]))
select_data=np.in1d(types,Datatype)
print('\n good location : '+str(len(np.where(select_loc)[0]))+' files')
print('\n good platform type : '+str(len(np.where(select_data)[0]))+' files')

good_paths=paths[select_loc & select_data]
Lon1=Loni[select_loc & select_data]  ### Warning ; takes only initial lat here
Lat1=Lati[select_loc & select_data]  ### Warning ; takes only initial lat here
Datestr1=Datestr[select_loc & select_data]  ### Warning ; takes only initial lat here
plat1=platforms[select_loc & select_data]  ### Warning ; takes only initial lat here


# %%
def str2juld(s):
    return (dt.date(int(s[0:4]),int(s[4:6]),int(s[6:8]))-dt.date(2000,1,1)).days

def juld2str(juld, format='%Y%m%d'):
    return dt.datetime.strftime(dt.date(2000,1,1)+dt.timedelta(days=int(juld)),format=format)

def profile_id(t,x,y,p):
    return juld2str(t)+'X'+'%05d'%(float(x%360)*100)+'Y'+'%05d'%(float(y)*100)+'P'+p

def profile_id_from_date(d,x,y,p):
    return d+'X'+'%05d'%(float(x%360)*100)+'Y'+'%05d'%(float(y)*100)+'P'+p


# %%
id1=[]
for i in tqdm(range(len(Lon1))):
    id1+=[profile_id_from_date(Datestr1[i], Lon1[i],Lat1[i],plat1[i])]
id1=np.array(id1)

# %%
### Find latest merged file
ListDir=os.listdir(path_work) ; Latest=[]

for name in ListDir:
    if name[:19]==file_merged:
        Latest+=[name]
Latest=np.sort(Latest)   ### Warning : selecting LATEST merged file here !
name_latest=Latest[-1] ; datestr=name_latest[-11:-3] ;
print('Date of merged NRT profiles :'+datestr)

# %%
### Loading previous profiles dataset
f=nc4.Dataset(path_work+name_latest,'r')
#id0=f['prof_id'][:]
time0=f['Days'][:] ; x0=f['Longitude'][:] ; y0=f['Latitude'][:]
cast0=f['cast_name'][:] ; file0=f['file_name'][:]
id0=[]
for i in tqdm(range(len(x0))):
    id0+=[profile_id(time0[i], x0[i],y0[i],cast0[i])]
id0=np.array(id0)

# %%
### 2nd selection => check if not already there
paths2=good_paths[~np.in1d(id1,id0)]
print('new files :'+str(len(np.where(~np.in1d(id1,id0))[0]))+' out of '+str(len(Lon1)))

# %%
ftp=FTP('nrt.cmems-du.eu')   
ftp.login(user='abarboni1', passwd='cmemsNico13@')

for path in tqdm(paths2):
    date=path.split('/')[-2]
    file=path.split('/')[-1]
    print('downloading : '+file)
    with closing(request.urlopen('ftp://'+user+':'+mdp+'@nrt.cmems-du.eu:'+distant_path+'latest/'+date+'/'+file)) as r:
        with open(path_down+'/'+file, 'wb') as f:
            shutil.copyfileobj(r, f)

# %%
