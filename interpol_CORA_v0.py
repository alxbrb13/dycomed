#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Apr 16 14:28:48 2021

@author: alexandre

This script is intended to download CORA NEAR REAL TIME database, it is mostly based on get_CORA_ftp.py
It uses the 'history' or 'monthly' NRT release

Profiles selection and extraction is made on location (grid coordinates bounds), QC check, and meeting of given depth criteria
Additional checks are successively made (notably on temperature gradient)
If no pressure data is given, takes DEPTH instead (XBT data)

At last profiles are interpolated on a common depth vector, saved in a common netcdf file per year

Different from get_CORA_DT_xx.py:
    - works per platform type (input of Datatype)
    - consider adjusted variables on TEMP instead of DEPH/PRES
    - Take into account potential factor
    
New March 2022 : check if profiles are within MED sea, which is not always the case from NRT source
New December 2023 : NRT batch mode on Datarmor

"""
import os
# import shutil
# import urllib.request as request
# from contextlib import closing
import netCDF4 as nc4
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
# import tarfile
from matplotlib import rcParams
import datetime as dt
import gsw
from tqdm import tqdm

import sys
sys.path.append('/home6/datahome/abarboni/DYNED-NRT/')

from tools_dycomed import interpol_profiles_column_filter
from tools_dyned_these import distance, load_bathy, plot_bathy #, interpol_quality, unique_prof #, link_eddies_end
#from tools_dyned_m2 import map_chart

plt.style.use(u'default')
rcParams['contour.negative_linestyle'] = 'solid' 

#Datatype=['GL','XB','CT']#,'PF','XX']

## Selection criteria
min_first=20  # Maximal depth of first profile, in meters
min_depth=400  # Minimal depth reached, in meters
min_nb_lev=40  # minimal number of levels IN MIN_DEPTH
shallowest_value=50 # Depth of shallowest value
sensiv_neg=2       # Maximal negative temperature (in degC) jump to be check further for suspicious jumps (removed only if only 0 above)
sensiv_pos=10       # Maximal positive temperature jump # Warning : sensiv_pos is expected to be higher than sensiv_neg
sensiv_warn=4      # Suspicious positive temperature jump
temp_range=[4,35] # Normal temperature range, flagged as NaN if outside
sal_range=[30,42] # Same for salinity

temp_eject=12     # Upper part minimal temperature (upper_part in meters)
upper_part=200

path_topo='/home6/datawork/abarboni/OceanData/'

"""##  old product  Name : Core/INSITU_MED_NRT_OBSERVATIONS_013_035/
Target file on CMEMS : /Core/INSITU_MED_PHYBGCWAV_DISCRETE_MYNRT_013_035/cmems_obs-ins_med_phybgcwav_mynrt_na_irr/
=> monthly
=> Datatyp GL / CT / XB / (SM)
"""

CoorLon=[-6,36] ; CoorLat=[30,46] ### Mediterranean
depth = np.array(list(np.arange(5,300,5))+list(np.arange(300,2010,10))).astype(float) ## Be careful ! Different depth vectors !

datestr=dt.datetime.strftime(dt.date.today(),format='%Y%m%d')
dir_down='LATEST_'+datestr
path_work='/home6/datawork/abarboni/DYCOMED/'+dir_down+'/'
path_save='/home6/datawork/abarboni/DYCOMED/'



# %% Bathymetry loading (only for plots, skipped if not necessary)
lon_t,lat_t,z_t=load_bathy(path_topo+'ETOPO2v2c_f4.nc', CoorLon, CoorLat)
#lon_t,lat_t,z_t=load_bathy(path_topo+'GEBCO_Med/gebco_2019_n46.0_s30.0_w-7.0_e37.0.nc', CoorLon, CoorLat, xname='lon', yname='lat',zname='elevation')

###  Showing only the MED sea
Land=(z_t>0).astype(bool)
M,Nmax=nd.measurements.label(~Land)  ## Finding connected sea surfaces
Sizes=[len(Land[M==x]) for x in range(1,Nmax)]  ## Identifying biggest sea
Mask=np.ones(np.shape(Land))
Mask[M==np.argmax(Sizes)+1]=0
Mask=Mask.astype(bool)
# %%
######  Only to unpack initial set-up
#for Type in Datatype:
 #   directlist=np.sort(os.listdir(path_work+Type))
  #  for month in directlist:
   #     listfile=os.listdir(path_work+Type+'/'+month)
    #    for file in listfile:
     #       os.system('mv '+path_work+Type+'/'+month+'/'+file+' '+path_work+file)

    # %%
    #Type='CT'
    #print('\n mode = '+Type+' \n')

    x_prim=[] ; y_prim=[] ; time_raw=[] ; name_prim=[] ; nbprof_list=[]
    max_prof_pres=[] ;max_prof_depth=[] ; min_prof_pres=[] ;min_prof_depth=[] ; nb_pres=[] ; nb_depth=[]
    time_QC=[] ; pos_QC=[] ;  psal_present=[] ; Nlev=[] ; Nprof=[] 
    nb_pres_adj=[] ; nb_deph_adj=[] ; adj_present=[]
    directlist=np.sort(os.listdir(path_work))

    for name in tqdm(directlist[:]):#filelist[:]:
        f = nc4.Dataset(path_work+name)#direct+'/'+name)
        N1=f.dimensions['DEPTH'].size
        ### First file selection on the number of available levels
        if N1>min_nb_lev:
            N0=f.dimensions['TIME'].size
            Nlev+=[N1]*N0 #Nprof+=[N0] ; 
            x_prim+=list(f['LONGITUDE'][:])
            y_prim+=list(f['LATITUDE'][:])
            time_raw+=list(f['TIME'][:])
            time_QC+=list(f['TIME_QC'][:])
            pos_QC+=list(f['POSITION_QC'][:])
    
            name_prim+=[name]*N0
            for j in range(N0):
                nbprof_list+=[j]
                
            if 'PRES' in f.variables.keys():     ## Check if pressure
                min_prof_pres+=list(np.min(f['PRES'][:], axis=1).data)
                max_prof_pres+=list(np.max(f['PRES'][:], axis=1).data)
                if np.shape(f['PRES'][:].mask)!=():
                    nb_pres+=list(np.sum(~f['PRES'][:].mask,axis=1))
                else:
                    nb_pres+=[N1]*N0
            else:
                max_prof_pres+=[np.nan]*N0 ; min_prof_pres+=[np.nan]*N0
                nb_pres+=[np.nan]*N0
                
            if 'DEPH' in f.variables.keys():    ### Check if Depth
                min_prof_depth+=list(np.min(f['DEPH'][:], axis=1).data)
                max_prof_depth+=list(np.max(f['DEPH'][:], axis=1).data)
                if np.shape(f['DEPH'][:].mask)!=():
                    nb_depth+=list(np.sum(~f['DEPH'][:].mask,axis=1))
                else:
                    nb_depth+=[N1]*N0
            else:         
                max_prof_depth+=[np.nan]*N0 ; min_prof_depth+=[np.nan]*N0
                nb_depth+=[np.nan]*N0
                
            if 'PRES_ADJUSTED' in f.variables.keys(): ## Check if Pressure Adjusted
                if np.shape(f['PRES_ADJUSTED'][:].mask)!=():
                    nb_pres_adj+=list(np.sum(~f['PRES_ADJUSTED'][:].mask,axis=1))
                else:
                    nb_pres_adj+=[N1]*N0
            else:
                nb_pres_adj+=[np.nan]*N0
                
            if 'DEPH_ADJUSTED' in f.variables.keys():   ## Check if Depth Adjusted
                if np.shape(f['DEPH_ADJUSTED'][:].mask)!=():
                    nb_deph_adj+=list(np.sum(~f['DEPH_ADJUSTED'][:].mask,axis=1))
                else:
                    nb_deph_adj+=[N1]*N0
            else:
                nb_deph_adj+=[np.nan]*N0
                
            ## Check if Depth Adjusted
            if 'TEMP_ADJUSTED' in f.variables.keys():
                adj_present+=[N1]*N0
            else:
                adj_present+=[np.nan]*N0                    
            if 'PSAL' in f.variables.keys():
                psal_present+=[True]*N0
            else:
                psal_present+=[False]*N0
        f.close()

    # %% Metadata conversion
    time_QC=np.array(time_QC)
    pos_QC=np.array(pos_QC)
    
    ### Putting in days since 2000-1-1 with 2000-1-1 as day 0 (same as DYNED, v. 2018 and later)
    time_prim=np.array(time_raw)-18262 # +1
    x_prim=np.array(x_prim)
    y_prim=np.array(y_prim)
    max_prof_pres=np.array(max_prof_pres) ; min_prof_pres=np.array(min_prof_pres)
    max_prof_depth=np.array(max_prof_depth) ; min_prof_depth=np.array(min_prof_depth)
    
    max_prof_pres[max_prof_pres>10000]=np.nan ;  min_prof_pres[min_prof_pres>10000]=np.nan
    max_prof_depth[max_prof_depth>10000]=np.nan ;  min_prof_depth[min_prof_depth>10000]=np.nan
    nb_pres=np.array(nb_pres).astype(float)
    nb_depth=np.array(nb_depth).astype(float)
    nb_pres_adj=np.array(nb_pres_adj).astype(float)
    nb_deph_adj=np.array(nb_deph_adj).astype(float)
    nb_pres[nb_pres==0]=np.nan
    nb_depth[nb_depth==0]=np.nan
    nb_pres_adj[nb_pres_adj==0]=np.nan
    nb_deph_adj[nb_deph_adj==0]=np.nan
    adj_present=np.array(adj_present).astype(float)
    adj_present[adj_present==0]=np.nan
    
    # nb_temp_adj=np.array(nb_temp_adj).astype(float)
    # nb_temp_adj[nb_temp_adj==0]=np.nan
# %%
plt.plot(time_raw)

    # %% Filter on quality and loc area
    GoodTime=((time_QC==1) + (time_QC==2) + (time_QC==5)) & ~np.isnan(time_prim) & (time_prim>0)
    #### Pos_QC==8 should be avoided !!
    GoodLoc=((pos_QC==1) + (pos_QC==2) + (pos_QC==5)) & ~np.isnan(x_prim) & ~np.isnan(y_prim)
    select_loc= (x_prim> CoorLon[0]) & (x_prim<CoorLon[1]) & (y_prim>CoorLat[0]) & (y_prim<CoorLat[1]) & GoodTime & GoodLoc
    
    ### Filter on nb_xxx and not nb_xxx_adj : assumes that if nb_xxx is not nan, nb_xxx_adj is not either.
    select_max=(max_prof_depth>min_depth) + (max_prof_pres>min_depth)
    select_min=(min_prof_depth<min_first) + (min_prof_pres<min_first)
    select_level =(nb_depth>min_nb_lev) + (nb_pres>min_nb_lev) #& ((nb_depth<10000) + np.isnan(nb_depth))
    
    select_second=select_loc & select_level & select_min & select_max ## boolean array
    time_second=time_prim[select_second]
    x_second=x_prim[select_second]
    y_second=y_prim[select_second]

    print('\n At least '+str(min_nb_lev)+' levels : '+str(len(np.where(select_level)[0])))
    print('\n '+str(min_depth)+'m reached : '+str(len(np.where(select_max)[0])))
    print('\n First value above '+str(min_first)+'m : '+str(len(np.where(select_min)[0])))
    print('\n good location : '+str(len(np.where(select_loc)[0])))
    print(' \n'+str(len(np.where(select_second)[0]))+' good profiles availables (out of '+str(len(x_prim))+')')
    # %% Check double with minimal distance compared to CORA-DT
    ### Adjusted
    Adjusted=(~np.isnan(nb_deph_adj) + ~np.isnan(nb_pres_adj))
    Adjusted_var=(~np.isnan(adj_present))
    print(str(len(np.where(Adjusted)[0]))+' profiles with adjusted PRES/DEPTH')
    print(str(len(np.where(Adjusted_var)[0]))+' profiles with adjusted TEMP/SAL variables')

    # %% Detecting profiles on land or outside Med sea
    Onshore=np.zeros(len(x_second)).astype(bool)
    select_quater=np.zeros(len(x_prim)) ; B=np.where(select_second)[0]
    for i in tqdm(range(len(x_second))):
        Onshore[i]=Mask[np.argmin(np.abs(y_second[i]-lat_t)),np.argmin(np.abs(x_second[i]-lon_t))]
        if Onshore[i]==0:
            select_quater[B[i]]=1
    select_quater=select_quater.astype(bool)
    time_quater=time_prim[select_quater]
    x_quater=x_prim[select_quater]
    y_quater=y_prim[select_quater]
    print('\n Check outisde Med : '+str(len(x_quater))+' (out of '+str(len(x_second))+') profiles available')

    # plt.figure(2, figsize=FigSize) ; plt.grid()
    # ax=plt.subplot(111) ; plt.title('Profiles location :', size=18)
    # plt.plot(x_ter[Onshore], y_ter[Onshore],'or', ms=5)
    # # plt.plot(x_prim[pos_QC==8], y_prim[pos_QC==8],'or', ms=5)    
    # plot_bathy(lon_t,lat_t,z_t)
    # plt.tick_params(labelsize=15)
    # map_chart(ax, CoorLon, CoorLat, 0.2, 0.2, n_x=1, n_y=1, stepy=0)
    # plt.xlim(CoorLon) ; plt.ylim(CoorLat)
    # %% Extracting and interpolating P-T-S data
    index_new=np.where(select_quater)[0]
    Nnew=len(index_new)
    Nmax=len(depth)
    ### D_interp : TEMP_IS/PSAL/SIGMA/TEMP_POT
    D_interp=np.zeros((Nnew,Nmax,4))
    
    name_array=np.zeros(Nnew, dtype='U40')
    
    scale=0
    unrealist_data=0
    i=0   ## i = index in selected profiles, k = index in raw prim file
    all_nan=np.zeros(Nnew)


    # %% Extracting and interpolating P-T-S data
    name_f=name_prim[0]
    f = nc4.Dataset(path_work+'/'+name_f)
    for k in tqdm(index_new):
        if name_prim[k]!=name_f:
            name_f=name_prim[k]
            f = nc4.Dataset(path_work+'/'+name_f)
        ### Adjusted variables
        adjusted=Adjusted[k]
        adjusted_var=Adjusted_var[k]
        if adjusted:
            strextra='_ADJUSTED'
        else:
            strextra=''
            
        if adjusted_var:
            strextra_var='_ADJUSTED'
        else:
            strextra_var=''
            
        index=nbprof_list[k]
        Nlev_prof=Nlev[k]
            
        ### Reading Name
        Name =f.id
        name_array[i]=Name+'-'+str(index)
        #name_list+=Name
        
        ### Erasing previous data
        PTS_prof=np.zeros((Nlev_prof,3))
        PTS_prof_QC=np.zeros((Nlev_prof,3), dtype='S2')
        
        ### TEMP
        if 'TEMP'+strextra_var in f.variables.keys():
            PTS_prof[:,1]=f['TEMP'+strextra_var][index,:Nlev_prof]
            PTS_prof_QC[:,1]=f['TEMP'+strextra_var+'_QC'][index,:Nlev_prof]        
        
        ### PRES-DEPTH
        if ~np.isnan(nb_pres[k]):
            PTS_prof[:,0]=f['PRES'+strextra][index,:Nlev_prof]
            PTS_prof_QC[:,0]=f['PRES'+strextra+'_QC'][index,:Nlev_prof]
            
        # Assumes that if pressure not given, depth must be given
        else:
            PTS_prof[:,0]=f['DEPH'+strextra][index,:Nlev_prof]
            PTS_prof_QC[:,0]=f['DEPH'+strextra+'_QC'][index,:Nlev_prof]
            
        ### SALINITY
        if psal_present[k]:
            PTS_prof[:,2]=f['PSAL'+strextra_var][index,:Nlev_prof]
            PTS_prof_QC[:,2]=f['PSAL'+strextra_var+'_QC'][index,:Nlev_prof]

        ### Quality control check
        PTS_prof[(PTS_prof_QC!=b'1') & (PTS_prof_QC!=b'2') & (PTS_prof_QC!=b'5') & (PTS_prof_QC!=b'8')]=np.nan
        
        ### If potential scale factor not already taken into account
        if (np.nanmean(PTS_prof[:,1])>10000) & (np.nanmean(PTS_prof[:,1])<30000):
            PTS_prof[:,1]=PTS_prof[:,1]/1000
            scale+=1
        if (np.nanmean(PTS_prof[:,2])>20000) & (np.nanmean(PTS_prof[:,1])<40000):
            PTS_prof[:,2]=PTS_prof[:,2]/1000
        

        ### 2nd Checking irrealistic data
        Irr = (PTS_prof[:,1]<temp_range[0]) + (PTS_prof[:,1]>temp_range[1])
        unrealist_data+=np.sum(Irr)
        PTS_prof[Irr ,1]=np.nan
        PTS_prof[(PTS_prof[:,2]<sal_range[0]) + (PTS_prof[:,2]>sal_range[1]) ,2]=np.nan

        ################### Test mode ################
        # plt.plot(PTS_prof[:,1],-PTS_prof[:,0])
        # plt.ylim([-200,0])
        # Press_for_Temp=np.copy(PTS_prof[:,0])
        # Press_for_Sal=np.copy(PTS_prof[:,0])Also new :
        # Press_for_Temp[np.isnan(PTS_prof[:,1])]=np.nan
        # Press_for_Sal[np.isnan(PTS_prof[:,1])]=np.nan

        ## Changing condition : instead of removing all-nan, require a minimum of non-nan values
        # if ~np.all(np.isnan(PTS_prof[:,1])) & ~np.all(np.isnan(PTS_prof[:,0])):
        if (np.sum(~np.isnan(PTS_prof[:,1]))>min_nb_lev/2) & (np.sum(~np.isnan(PTS_prof[:,0]))>min_nb_lev/2):
            D_interp[i,:,0]=interpol_profiles_column_filter(PTS_prof[:,0], PTS_prof[:,1], depth)    
            all_nan[i]=False
            ### Computing density and potential temperature if salinity present
            if psal_present[k] & ~np.all(np.isnan(PTS_prof[:,2])):
                #Absolute Salinity
                SA_prof = gsw.SA_from_SP(PTS_prof[:,2],PTS_prof[:,0], x_quater[i], y_quater[i])
                                        
                #Conservative temp
                CT_prof = gsw.CT_from_t(SA_prof, PTS_prof[:,1], PTS_prof[:,0])
                #PotTemp
                PT_prof = gsw.pt0_from_t(SA_prof, PTS_prof[:,1], PTS_prof[:,0])
                Sigma_prof=gsw.density.sigma0(SA_prof,CT_prof)
    
                #### Interpolation
                D_interp[i,:,1]=interpol_profiles_column_filter(PTS_prof[:,0], PTS_prof[:,2], depth)
                D_interp[i,:,2]=interpol_profiles_column_filter(PTS_prof[:,0], Sigma_prof , depth)
                D_interp[i,:,3]=interpol_profiles_column_filter(PTS_prof[:,0], PT_prof , depth)

            else:
                D_interp[i,:,1]=np.nan
                D_interp[i,:,2]=np.nan
                D_interp[i,:,3]=np.nan
        else:
            D_interp[i,:,0]=np.nan
            D_interp[i,:,1]=np.nan
            D_interp[i,:,2]=np.nan
            D_interp[i,:,3]=np.nan
            all_nan[i]=True
        
        i+=1
    print('\n WARNING : '+str(unrealist_data)+' unrealistic data, flagged as NaNs')
    print('\n WARNING : '+str(len(np.where(all_nan)[0]))+' profiles with only NaNs')
    print('\n WARNING : '+str(scale)+' profiles rescaled')

    # %% 2nd checking on unrealistic values
    D_ter=np.copy(D_interp)
    idupper=np.argmin(np.abs(depth-upper_part))
    SuspectIrr=np.unique(np.where(D_ter[:,:idupper,0]<temp_eject)[0])

    print('\n WARNING :'+str(len(SuspectIrr))+' profiles with irrealistic data removed')
    
    if len(SuspectIrr)>0:
        plt.figure(0,figsize=(10,8))  
        for i in range(len(SuspectIrr)):
            plt.plot(D_ter[SuspectIrr[i],:,0], -depth)
        plt.ylim([-300,0]) ; plt.grid()
        plt.ylabel('Depth [m]',size=15)
        plt.xlabel('Temperature [degC]',size=15)
        plt.text(5,-250, str(len(SuspectIrr))+' profiles removed',fontsize=15)
        plt.title(' 2nd Check : Irrealistic value < '+str(temp_eject)+' degC above '+str(upper_part)+'m',size=16)
        #plt.savefig(path_save+'1_CheckIrrealist_'+Type+'.png')
        #plt.close()
    D_ter[SuspectIrr]=np.nan

# %% 3rd check on Negative temperature gradient threshold
    
    Tdiff=np.diff(D_ter[:,:,0],axis=1) #; susjump=0 ; suspectbis=0
    SuspectJump=np.unique(np.where(Tdiff>sensiv_neg)[0])
    print('\n WARNING :'+str(len(SuspectJump))+' profiles with suspicious negative jumps cropped')

    if len(SuspectJump)>0:
        plt.figure(0,figsize=(10,8))
        for h in range(len(SuspectJump)):
            maxsus=np.where(Tdiff[SuspectJump[h]]>sensiv_neg)[0][0]
            D_ter[SuspectJump[h],:maxsus+3,:]=np.nan
            plt.plot(D_ter[SuspectJump[h],:,0], -depth)
        plt.ylim([-300,0]) ; plt.grid()
        plt.ylabel('Depth [m]',size=15)
        plt.xlabel('Temperature [degC]',size=15)
        plt.text(18,-250, str(len(SuspectJump))+' profiles cropped',fontsize=15)
        plt.title('3rd Check : negative Temp jump > '+str(sensiv_neg)+' degC',size=16)
        #plt.savefig(path_save+'2_CheckNegJump_'+Type+'.png')
        #plt.close()

    # %% 4th check on Postive temperature gradient threshold
    # D_cinq=np.copy(D_quat)
    Tdiff=np.diff(D_ter[:,:,0],axis=1)
    SuspectJump=np.unique(np.where(np.abs(Tdiff)>sensiv_pos)[0])
    print('\n WARNING :'+str(len(SuspectJump))+' profiles with suspicious positive jumps cropped')
    if len(SuspectJump)>0:
        plt.figure(0,figsize=(10,8))
        for h in range(len(SuspectJump)):
            maxsus=np.where(np.abs(Tdiff[SuspectJump[h]])>sensiv_pos)[0][0]
            D_ter[SuspectJump[h],:maxsus+5,:]=np.nan
            plt.plot(D_ter[SuspectJump[h],:,0], -depth)
            plt.plot(D_interp[SuspectJump[h],:,0], -depth,'--')
        plt.ylim([-300,0]) ; plt.grid()
        plt.ylabel('Depth [m]',size=15)
        plt.xlabel('Temperature [degC]',size=15)
        plt.text(18,-250, str(len(SuspectJump))+' profiles cropped',fontsize=15)
        plt.title('4th Check : positive Temp jump > '+str(sensiv_pos)+' degC',size=16)
        #plt.savefig(path_save+'3_CheckPosJump_'+Type+'.png')
        #plt.close()

    # %% Last check on gaps : Empirically, profiles with gaps are mostly non-reliable
    #Counting number of changes from nan to non-nan
    NanChanges=np.abs(np.diff(np.diff(np.cumsum(np.isnan(D_ter[:,:,0]).astype(int),axis=1))))
    # NoNanTop=~np.isnan(D_bis[:,-1,0])
    NoNanBottom=~np.isnan(D_ter[:,0,0])
    ChangeCount=np.sum(NanChanges,axis=1)
    # problematic profiles with gaps have 2 changes + 1 third one if bottom or top is already with a gap
    ToChange=np.where((ChangeCount>=3) + ((ChangeCount==2) & (NoNanBottom)))[0]

    print('\n WARNING :'+str(len(ToChange))+' profiles with gaps removed')
    if len(ToChange)>0:    
        plt.figure(0,figsize=(10,8))
        for h in ToChange:
            plt.plot(D_ter[h,:,0],-depth)
            D_ter[h,:,:]=np.nan
        plt.grid() ; plt.ylim([-300,0])
        plt.ylabel('Depth [m]',size=15)
        plt.xlabel('Temperature [degC]',size=15)
        plt.text(15,-250, str(len(ToChange))+' profiles removed',fontsize=15)
        plt.title(' 5th Check : profiles with gaps',size=16)
        #plt.savefig(path_save+'4_CheckGap_'+Type+'.png')
        #plt.close()

    # %% Selection on profile number
    idstop=np.argmin(np.abs(depth-min_depth))
    idup=np.argmin(np.abs(depth-shallowest_value))
    NOTNAN=~np.isnan(D_ter[:,:,0])
    to_keep=np.where((np.sum(NOTNAN[:,:idstop],axis=1)>min_nb_lev) & (np.sum(NOTNAN[:,:idup],axis=1)>0) & (np.sum(NOTNAN[:,idstop:],axis=1)>0))[0]
    
    D_bis=D_ter[to_keep]
    x_cora=x_quater[to_keep] ; y_cora=y_quater[to_keep]
    time_cora=time_quater[to_keep]
    print(str(len(x_cora))+' profiles kept (out of '+str(len(x_quater))+')')
    ### name of the platform
    name_cora=name_array[to_keep]
    
    ### Building a str name to link the CORA file
    name_plat=[]
    for j in range(len(to_keep)):
        name_plat+=[name_cora[j].split('_')[3]]
    name_plat=np.array(name_plat)

    # %%
    Tdiff=np.diff(D_bis[:,:,0],axis=1)
    Suspect=np.where(np.abs(Tdiff)>4)[0]
    plt.figure(0,figsize=(10,8))
    for h in Suspect:
        plt.plot(D_bis[h,:,0],-depth)
    plt.grid() ; plt.ylim([-150,0])
    plt.ylabel('Depth [m]',size=15)
    plt.xlabel('Temperature [degC]',size=15)
    plt.text(18,-140, str(len(Suspect))+' potential unreliable \n profiles remain',fontsize=15)
    plt.title('Remaining : positive Temp jump > '+str(sensiv_warn)+' degC',size=16)
    #plt.savefig(path_save+'5_RemainGradT_'+Type+'.png')
    #plt.close()
    print('\n WARNING :'+str(len(Suspect))+' potential unreliable profiles remain')

# %% Check plot
    
    plt.figure(0)
    plt.imshow(~np.isnan(D_bis[:,:,0]).T, aspect=10)
    plt.plot([0,len(D_bis)],[np.argmin(np.abs(depth-min_depth))]*2,'--')
    plt.yticks([49,99,129,179,229],['250','700','1000','1500','2000']) ; plt.xlabel('Profiles')
    plt.ylabel('Depth [m]')
    #plt.savefig(path_save+'NanPlot_'+Type+'.png')
    #plt.close()
    
    plt.figure(1)
    plt.hist(time_cora%365.25, bins=50) ; plt.xlim([0,365])
    plt.xlabel('Day in the year')
    plt.ylabel('Profile histogram')
    #plt.savefig(path_save+'HistDate_'+Type+'.png')
    #plt.close()
    # %%
    FigSize=(20,10)
    plt.figure(2, figsize=FigSize) ; plt.grid()
    ax=plt.subplot(111)
    plt.title('Profiles location : '+str(len(to_keep))+' profiles - Update '+datestr, size=18)
    plt.plot(x_cora, y_cora,'or', ms=5)
    plot_bathy(lon_t,lat_t,z_t)
    plt.tick_params(labelsize=15)
    #map_chart(ax, CoorLon, CoorLat, 0.2, 0.2, n_x=1, n_y=1, stepy=0)
    plt.xlim(CoorLon) ; plt.ylim(CoorLat)
    #plt.savefig(path_save+'MapProfile_'+Type+'.png')
    #plt.close()

    # %%
    plt.figure(0, figsize=(8,6)) ; plt.grid()
    Yi=int(np.min(time_cora/365.25)) ; Yf=int(np.max(time_cora/365.25))+1
    plt.hist(time_cora, bins=(Yf-Yi+1)*12, label='Near Real Time') 
    plt.xticks(ticks=np.arange(Yi*365.25,Yf*365+10,365) ,labels=list(np.arange(2000+Yi,2001+Yf).astype(str)), rotation=45)
    plt.xlim([np.min(time_cora),np.max(time_cora)])
    plt.tick_params(labelsize=14)
    plt.ylabel('Profiles per month', size=16)
    plt.title('Profiles histogram ', size=18)
    plt.tight_layout()
    #plt.savefig(path_save+'HistYear_'+Type+'.png')
    #plt.close()
# %% Creating NetCDF file
    
    #Change write mode to 'w' if not already created
    f = nc4.Dataset(path_save+'CORA_NRT_MED_latest.nc','w', format='NETCDF4')
    f.description = "Eastern Med profiles from different sources merged per group"
    
    #coragrp = f.createGroup('CORA')
    f.createDimension('Nprof', len(x_cora))
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


# %% Writing NetCDF
    
    lon.units = 'degrees East' 
    lat.units = 'degrees North'
    time.units = 'days since 2000-01-01'
    cast_name.units = 'Cast name'
    
    depth_nc.units = 'Reference depth vector for interpolation, in meters'
    Temp_interp.units = 'In situ Temperature, in degC, interpolated on depth_ref vector'
    Sal_interp.units = 'Practical Salinity, in PSU, interpolated on depth_ref vector'
    PTemp_interp.units = 'Potential Temperature, in degC, interpolated on depth_ref vector'
    Sigma_interp.units = 'Potential density -1000, in kg/m³, interpolated on depth_ref vector'
    
    lon[:]=x_cora[np.argsort(time_cora)]
    lat[:]=y_cora[np.argsort(time_cora)]
    time[:]=time_cora[np.argsort(time_cora)]
    cast_name[:]=name_plat[np.argsort(time_cora)]
    file_name[:]=name_cora[np.argsort(time_cora)]
    depth_nc[:]=depth
    
    Temp_interp[:]=D_bis[np.argsort(time_cora)][:,:,0]
    Sal_interp[:]=D_bis[np.argsort(time_cora)][:,:,1]
    PTemp_interp[:]=D_bis[np.argsort(time_cora)][:,:,3]
    Sigma_interp[:]=D_bis[np.argsort(time_cora)][:,:,2]


    # %% Saving NetCDF
    f.description = 'Vertically interpolated CORA-NRT database \n Updated '+dt.date.today().strftime('%d/%m/%Y')
    f.close()


# %%
def juld2str(juld, format='%Y%m%d'):
    return dt.datetime.strftime(dt.date(2000,1,1)+dt.timedelta(days=int(juld)),format=format)

def profile_id(t,x,y,p):
    return juld2str(t)+'X'+'%05d'%(float(x%360)*100)+'Y'+'%05d'%(float(y)*100)+'P'+p


    # %%
    ### Adding profile ID
    f=nc4.Dataset(path_save+'CORA_NRT_MED_latest.nc','a')
    idprof = f.createVariable('prof_id' , 'S4', 'Nprof')
    time0=f['Days'][:]
    x0=f['Longitude'][:]
    y0=f['Latitude'][:]
    cast0=f['cast_name'][:]
    file0=f['file_name'][:]


    # %%
    id0=[]
    for i in tqdm(range(len(x0))):
        id0+=[profile_id(time0[i], x0[i],y0[i],cast0[i])]
    id0=np.array(id0)
    idprof[:]=id0
    f.close()

# %%
