# +
import os
import netCDF4 as nc4
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import datetime as dt
from tqdm import tqdm

import sys
sys.path.append('/home6/datahome/abarboni/DYNED-NRT/')
from tools_dycomed import juld2str, str2juld
from tools_dyned_these import load_bathy, plot_bathy #, interpol_quality, unique_prof #, link_eddies_end
#from tools_dyned_m2 import map_chart

plt.style.use(u'default')
rcParams['contour.negative_linestyle'] = 'solid' 

path_topo='/home6/datawork/abarboni/OceanData/'
path_obs='/home6/datawork/abarboni/DYCOMED/SSH_18_MED/OBS/'
path_save='/home6/datahome/abarboni/DYNED-NRT/FIG/'

datestr=dt.datetime.strftime(dt.date.today(),'%Y%m%d')
last=7 ## heatmap shown over last x days
CoorLon=[-6,36] ; CoorLat=[30,46] ### Mediterranean
# -

lon_t,lat_t,z_t=load_bathy(path_topo+'ETOPO2v2c_f4.nc', CoorLon, CoorLat)

ListObs=np.sort(os.listdir(path_obs))
Ni=int(len(ListObs)/2)
CElist=ListObs[-last:] ; AElist=ListObs[Ni-last:Ni]

# +
x_ea=[] ; y_ea=[] ; x_ma=[] ; y_ma=[]  ; Noa=[] ; time_ea=[]
x_ec=[] ; y_ec=[] ; x_mc=[] ; y_mc=[]  ; Noc=[] ; time_ec=[]
for t in tqdm(range(last)):
    f=nc4.Dataset(path_obs+CElist[t],'r')  ### Cyclones
    Nc=f.dimensions['obs'].size
    Noc+=[Nc]
    x_ec+=list(f['effective_contour_longitude'][:]) ; y_ec+=list(f['effective_contour_latitude'][:])
    x_mc+=list(f['speed_contour_longitude'][:]) ; y_mc+=list(f['speed_contour_latitude'][:])
    time_ec+=[t]*Nc
    
    f=nc4.Dataset(path_obs+AElist[t],'r')  ### Anticyclones
    Na=f.dimensions['obs'].size
    Noa+=[Na]
    x_ea+=list(f['effective_contour_longitude'][:]) ; y_ea+=list(f['effective_contour_latitude'][:])
    x_ma+=list(f['speed_contour_longitude'][:]) ; y_ma+=list(f['speed_contour_latitude'][:])
    time_ea+=[t]*Na

x_ea=np.array(x_ea) ; y_ea=np.array(y_ea)
x_ma=np.array(x_ma) ; y_ma=np.array(y_ma)
Nota=np.cumsum(Noa)
x_ec=np.array(x_ec) ; y_ec=np.array(y_ec)
x_mc=np.array(x_mc) ; y_mc=np.array(y_mc)
Notc=np.cumsum(Noc)

# +
FigSize=(24,10) ;LS=16
fig=plt.figure(2, figsize=FigSize) ; plt.grid()
ax=plt.subplot(111)
plt.title('Eddy observation last '+str(last)+' days - '+datestr, size=20)
plt.plot([0],[0],'or', ms=15, label='Cyclone') ; plt.plot([0],[0],'ob', ms=15, label='Anticyclone')
plt.plot([0],[0],'--k', lw=2, label='effective contour on '+datestr) ;plt.plot([0],[0],'-k', lw=2, label='speed contour on '+datestr)
plot_bathy(lon_t,lat_t,z_t)
for i in tqdm(range(Nota[-1])):
    plt.fill(x_ma[i,:],y_ma[i,:],'b',alpha=0.2)
    if i > Nota[-2]:   ## Plotting contours but only for last day
        plt.plot(x_ea[i,:],y_ea[i,:],'--k')
        plt.plot(x_ma[i,:],y_ma[i,:],'-k')
for i in tqdm(range(Notc[-1])):
    plt.fill(x_mc[i,:],y_mc[i,:],'r',alpha=0.2)
    if i > Notc[-2]:
        plt.plot(x_ec[i,:],y_ec[i,:],'--k')
        plt.plot(x_mc[i,:],y_mc[i,:],'-k')
plt.tick_params(labelsize=LS)
#map_chart(ax, CoorLon, CoorLat, 0.2, 0.2, n_x=1, n_y=1, stepy=0)
plt.xlim(CoorLon) ; plt.ylim(CoorLat) ; plt.legend(loc=3,fontsize=LS)
plt.xlabel('Longitude [\N{DEGREE SIGN}E]',size=LS) ; plt.ylabel('Longitude [\N{DEGREE SIGN}E]',size=LS) 

cmap = mpl.cm.Greys ; bounds = np.arange(last+1) ;norm = mpl.colors.BoundaryNorm(bounds, cmap.N) #, extend='over')
CB=plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax, pad=0.01)
CB.ax.set_ylabel('Persitance [days]',size=LS)#, labelpad=-20)
CB.ax.tick_params(labelsize=LS)

plt.savefig(path_save+'Obs_last7days_'+datestr+'.png')
plt.close()
# -


