""" Functions for visualising dataset on isentropic levels
    Must be in windspharm environment                   """

#  %%
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import matplotlib.path as mpath
from PIL import Image
import io
import windspharm

# %%
def zm_theta(plobject, var='theta', startlev=34, stoplev=68,
             stoplat=60, fsize=14,
             meaning=False, time_slice=-1):
    """ Figure 2 from Garate-Lopez+2016
    Latitude vs. vertical coord zonal mean theta or extended theta
    
    Inputs: Planet object with attached data
            var can be theta or xtheta      """
    if meaning==True:
        tcube = np.mean(plobject.data['temp'][:,startlev:stoplev,:stoplat,:].values, axis=0)
        pcube = np.mean(plobject.data['pres'][:,startlev:stoplev,:stoplat,:].values, axis=0)
    else:
        tcube = plobject.data['temp'][time_slice,startlev:stoplev,:stoplat,:].values
        pcube = plobject.data['pres'][time_slice,startlev:stoplev,:stoplat,:].values

    if var=='theta':
        if plobject.model=='lmd':
            if not hasattr(plobject, 'theta'):
                plobject.calc_theta()
            theta = plobject.theta[time_slice,startlev:stoplev,:stoplat,:].values
            zmth = np.mean(theta, axis=-1)
            title = 'Potential temperature'
            llevs = np.arange(280,830,50)
            clevs = np.arange(np.min(zmth), np.max(zmth), 1)
            clevs = np.arange(280, 950, 10)
        if plobject.model=='oasis':
            p0 = 100000
            theta = tcube*((p0/pcube)**(188/900))
            zmth = np.mean(theta, axis=-1)
            title= 'Potential temperature'
            llevs = np.arange(330,830,50)
            clevs = np.arange(np.min(zmth), np.max(zmth), 1)
            clevs = np.arange(280, 950, 10)


    fig, ax = plt.subplots(figsize=(8,8))
    cs = ax.contour(plobject.lats[:stoplat], plobject.levs[startlev:stoplev], 
                    zmth, 
                    levels=llevs,
                    linewidths=0.5, colors='w')
    cf = ax.contourf(plobject.lats[:stoplat], plobject.levs[startlev:stoplev], 
                     zmth, 
                     levels=clevs,
                     cmap='nipy_spectral')
    ax.set_title(f'{title}', fontsize=fsize)
    ax.set_ylabel(f'{plobject.vert_axis} / {plobject.vert_unit}')
    ax.set_xlabel('Latitude / deg')
    ax=plt.gca()
    ax.invert_xaxis()
    if plobject.model=='lmd' or plobject.run=='isobars':
        ax.invert_yaxis()
        ax.set_yscale('log')
    ax.clabel(cs, inline=True)
    cbar = plt.colorbar(cf, orientation='horizontal',location='top')
    cbar.set_label('K')
    plt.show()

# %%
