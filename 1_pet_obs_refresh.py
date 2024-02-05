# +
"""
Created on 2023-12-01

@author: alexandre

This script uses NRT AVISO SSH to detect eddies with PyEddyTracker algorithm and store it in NetCDF file

Option refresh to proceed only the last SSH files
            
"""

import datetime as dt # datetime
from matplotlib import pyplot as plt
from numpy import arange
import numpy as np
import netCDF4 as nc4
import h5py
import os
from tqdm import tqdm
import sys
sys.path.append('/home6/datahome/abarboni/DYNED-NRT/')
from tools_dycomed import str2juld, juld2str
#from tools_nav import THS_NAV, NAV_only, datestr2num

from py_eddy_tracker import data
from py_eddy_tracker.dataset.grid import RegularGridDataset
#date_update=dt.datetime.strftime(dt.date.today(),'%Y%m%d')

add_refresh=6 ## additional list refreshment (to reprocess obs at D+6)

# +
path_work = '/home6/datawork/abarboni/DYCOMED/SSH_18_MED/DATA/'
path_out= '/home6/datawork/abarboni/DYCOMED/SSH_18_MED/OBS/'

### Automatically list available SSH and already processed obs
ListDir=np.sort(os.listdir(path_work))
ListOut=np.sort(os.listdir(path_out))
Dates_out=[] ; Julds=[]
for name in ListOut:
    Dates_out+=[name[3:11]]
    Julds+=[str2juld(name[3:11])]
Dates_out=np.array(Dates_out) ; Julds=np.array(Julds)
last_obs=Dates_out[-1]
last_ssh=ListDir[-1].split('_')[4]
Dtime=str2juld(last_ssh)- str2juld(last_obs)  ### New days for obs to be computed 
print('\n New days for obs to be computed : '+str(Dtime))
print('\n refreshing in addition past '+str(add_refresh)+' days')
# -

for name in tqdm(ListDir[-Dtime-add_refresh:]):  ## also refresh last 'add_refresh' days (typically Dtime +6)
    dstr=name.split('_')[4]
    dproc=name.split('_')[-1][:-3]
    date = dt.datetime(int(dstr[:4]), int(dstr[4:6]), int(dstr[6:]))
    
    if dstr in Dates_out:      ### Removing previous file if it exists !
        for file in ListOut[np.where(Dates_out==dstr)[0]]:
            os.system('rm '+path_out+file)
                      
                                ### Detection
    g = RegularGridDataset(
        path_work+name,"longitude","latitude")
    g.add_uv("adt")            ### Recompute U/V !
    g.bessel_high_filter("adt", 500)  ### Apply 500km wvelength high-pass filter
    a, c = g.eddy_identification("adt", "u", "v", date, 0.002, shape_error=55)
    ## Saving outputs
    a.to_netcdf(nc4.Dataset(path_work+'../OBS/AE_'+dstr+'_from_'+dproc+'.nc','w'))
    c.to_netcdf(nc4.Dataset(path_work+'../OBS/CE_'+dstr+'_from_'+dproc+'.nc','w'))

#### Looking for previous Atlas (untracked)
ListAtlas=np.sort(os.listdir(path_out+'../'))
files=[]
for name in ListAtlas:
    if name[:20]=='PET_Atlas_untracked_':
        files+=[name]
files

#### Listing Obs already there (including new ones)
ListOut=np.sort(os.listdir(path_out))
Dates_out=[] ; Julds=[]
for name in ListOut:
    Dates_out+=[name[3:11]]
    Julds+=[str2juld(name[3:11])]
Dates_out=np.array(Dates_out) ; Julds=np.array(Julds)
ListDate=np.unique(Dates_out)
dateobs0=Dates_out[0] ; dateobsF=Dates_out[-1]


def collect_obs(ListDate,dateobs,ListOut,path_out):
    xcen=[] ; ycen=[] ; xmax=[] ; ymax=[] ; xend=[] ; yend=[] ; timee=[] ; Pol=[]
    for d in tqdm(ListDate):
        for pol in np.where(dateobs==d)[0]:
            f=nc4.Dataset(path_out+ListOut[pol])
            No=f.dimensions['obs'].size
            juld=str2juld(d)
            xmax+=list(f['speed_contour_longitude'][:].data)
            ymax+=list(f['speed_contour_latitude'][:].data)
            xend+=list(f['effective_contour_longitude'][:].data)
            yend+=list(f['effective_contour_latitude'][:].data)
            xcen+=list(f['longitude'][:].data)
            ycen+=list(f['latitude'][:].data)
            timee+=[juld]*No
            if ListOut[pol][:2]=='AE':
                Pol+=[-1]*No
            else:
                Pol+=[1]*No
    xcen=np.array(xcen) ;ycen=np.array(ycen)
    xend=np.array(xend) ; yend=np.array(yend)
    xmax=np.array(xmax) ; ymax=np.array(ymax)
    timee=np.array(timee) ; Pol=np.array(Pol)
    Not=len(timee)
### Longitude with convention range = +/-180
    xcen[xcen>180]-=360
    xend[xend>180]-=360
    xmax[xmax>180]-=360
    return xcen, ycen, xmax, ymax, xend, yend, timee, Pol, Not


### if no Atlas creating it
if len(files)==0:
    
    print('\n  \n Creating new untracked Atlas from scratch : PET_Atlas_untracked_'+dateobs0+'_'+dateobsF)
    ### Collecting obs
    xcen, ycen, xmax, ymax, xend, yend, timee, Pol, Not=collect_obs(ListDate[:],Dates_out,ListOut,path_out)
    #### Creating new netcdf
    f=nc4.Dataset(path_out+'../PET_Atlas_untracked_'+dateobs0+'_'+dateobsF+'.nc','w')
    f.description='Merged PET obs from '+dateobs0+' to '+dateobsF
    f.createDimension('Nobs', None)
    f.createDimension('Nsample', 50)

    x_cen = f.createVariable('x_cen', 'f4', 'Nobs')
    y_cen = f.createVariable('y_cen', 'f4', 'Nobs')
    time_e  = f.createVariable('time_eddy', 'i4', 'Nobs')
    x_end = f.createVariable('x_end', 'f4', ('Nobs','Nsample'))
    y_end = f.createVariable('y_end', 'f4', ('Nobs','Nsample'))
    x_max = f.createVariable('x_max', 'f4', ('Nobs','Nsample'))
    y_max = f.createVariable('y_max', 'f4', ('Nobs','Nsample'))
    pol_ed = f.createVariable('polarity', 'i4', 'Nobs')
    
    x_cen[:]=xcen ; y_cen[:]=ycen
    x_end[:]=xend ; y_end[:]=yend
    x_max[:]=xmax ; y_max[:]=ymax
    time_e[:]=timee ; pol_ed[:]=Pol
    f.close()
else:    

    print('\n  \n Updating untracked Atlas : '+files[0]+'\n until '+dateobsF)
    f=nc4.Dataset(path_out+'../'+files[0],'a')
    f.description='Merged PET obs from '+dateobs0+' to '+dateobsF
    time_e_ini=f['time_eddy'][:]
    id0=np.where(time_e_ini==Julds[-Dtime-add_refresh])[0][0]

    xcen, ycen, xmax, ymax, xend, yend, timee, Pol, Not=collect_obs(ListDate[-Dtime-add_refresh:],Dates_out,ListOut,path_out)
    f['x_cen'][id0:]=xcen
    f['y_cen'][id0:]=ycen
    f['x_max'][id0:]=xmax
    f['y_max'][id0:]=ymax
    f['x_end'][id0:]=xend
    f['y_end'][id0:]=yend
    f['time_eddy'][id0:]=timee
    f['polarity'][id0:]=Pol
    f.close()
    os.rename(path_out+'../'+files[0],path_out+'../PET_Atlas_untracked_'+dateobs0+'_'+dateobsF+'.nc')


