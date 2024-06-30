#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:13:12 2024

@author: alexandre
"""

import copernicusmarine as cm
import os
import numpy as np

#### You should also be logged in the copernicusmarine toolbox.
# This might require in command line(for bash) : $ export COPERNICUSMARINE_CACHE_DIRECTORY="path_where_you_can_write"
# (alternatively for csh ) $ setenv COPERNICUSMARINE_CACHE_DIRECTORY "path_where_you_can_write"
# then : $ copernicusmarine login 

#%% Default example : CORA Database 2015-2018 in the Med Sea (delayed-time)
  
path_work='./INSITU_DATA_singlefiles/'
os.makedirs(path_work,exist_ok=True)
  
data_types=['GL','XB','PF']
region='mediterrane'  ## available options : artic, baltic, blacksea, global, mediterrane, northwesternshelf, southwestshelf
years=np.arange(2015,2018+1).astype(str)

for TYPE in data_types:
    for year in years:
        print(" \n  ##### Downloading : "+TYPE+' - year : '+year+'######')
        cm.get( 
               dataset_id='cmems_obs-ins_glo_phy-temp-sal_my_cora_irr',
               filter='*'+region+'/'+year+'/CO_DMQCGL01_'+year+'*_PR_'+TYPE+'*',
               output_directory=path_work,
               no_directories=True,
               force_download=True
               )
#%%     
### Example 2 : Profiles in the Med Sea, Near-Real-Time dataset, history datapart
path_work='./NRT_CORA_singlefiles/'
os.makedirs(path_work,exist_ok=True)
  
data_types=['CT','GL','PF','XB']
for TYPE in data_types:
    print(" ##### Downloading : "+TYPE+' ######')
    cm.get( 
           dataset_id='cmems_obs-ins_med_phybgcwav_mynrt_na_irr',
           dataset_part='history',
           filter='/'+TYPE+'/GL_PR_'+TYPE+'*',
           output_directory=path_work,
           no_directories=True,
           force_download=True
           )

#%%
### Example 3 : Delayed Time BGC Argo float data containing Oxygen

data_types=['PF']  ## NB : also lots of data in CTD files, but files indexing is not always in 'GL_PR_CT*'
for TYPE in data_types:
    print(" ##### Downloading : "+TYPE+' ######')
    cm.get( 
           dataset_id='cmems_obs-ins_glo_bgc-ox_my_na_irr',
           dataset_part='history',
           filter='Data_In_microlmolL/'+TYPE+'/GL_PR_'+TYPE+'*',
           output_directory=path_work,
           no_directories=True,
           force_download=True
           )
    
#%%
### Example 4 : Drifter positions (SVP)

cm.get( 
       dataset_id='cmems_obs-ins_glo_phy-cur_my_adcp_irr',
       output_directory=path_work,
       no_directories=True,
       #force_download=True
       )