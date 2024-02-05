# +
""" 
This script download all availalbe data with types listed in Datatype
from Copernicus monthly release
Over years listed in years

"""
import os
import shutil
import urllib.request as request
from contextlib import closing
#import netCDF4 as nc4
from ftplib import FTP
import numpy as np
import matplotlib.pyplot as plt
import tarfile
from matplotlib import rcParams
import datetime as dt
from tqdm import tqdm

import secretcodes

distant_path='/Core/INSITU_MED_PHYBGCWAV_DISCRETE_MYNRT_013_035/cmems_obs-ins_med_phybgcwav_mynrt_na_irr/monthly/'
path_work='/home6/datawork/abarboni/DYCOMED/ARGO/'
years=np.arange(2021,2024).astype(str)
Datatype=['PF'] # 'CT','GL','XB']
mdp=secretcodes.cmems_password
user=secretcodes.cmems_username
# -

ftp=FTP('nrt.cmems-du.eu')   
ftp.login(user=user, passwd=mdp)
for Type in Datatype:
    ftp.cwd(distant_path+Type+'/')
    months=ftp.nlst()  ### Listing available data in ftp directory

    good_months=[]
    for month in months:
        if month[:4] in years:
            good_months+=[month]

    for month in tqdm(good_months):
        print(' \n MONTH = '+month) 
        ftp.cwd(distant_path+'PF/'+month)
        listfiles=ftp.nlst()
        for file in listfiles:
        # urllib.request.urlretrieve('ftp://logincmems:passwordcmems@my.cmems-du.eu:'+distant_path, 'CORA-5.2-mediterrane-2000.tgz')
            with closing(request.urlopen('ftp://'+user+':'+mdp+'@nrt.cmems-du.eu:'+distant_path+Type+'/'+month+'/'+file)) as r:
                with open(path_work+file, 'wb') as f:
                    shutil.copyfileobj(r, f)
