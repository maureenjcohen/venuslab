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
from windspharm.standard import VectorWind

from venuspoles import *

# %%
def zm_theta(plobject, startlev=34, stoplev=68,
             stoplat=60, fsize=14,
             time_slice=-1):
    """ Figure 2 from Garate-Lopez+2016
    Latitude vs. vertical coord zonal mean theta or extended theta
    
    Inputs: Planet object with attached data      """
    if not hasattr(plobject, 'theta'):
                plobject.calc_theta()

    theta = plobject.theta[time_slice,startlev:stoplev,:stoplat,:].values
    zmth = np.mean(theta, axis=-1)
    llevs = np.arange(280,830,50)
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
    ax.set_title(f'Potential temperature', fontsize=fsize)
    ax.set_ylabel(f'{plobject.vert_axis.capitalize()} / {plobject.vert_unit}')
    ax.set_xlabel('Latitude / deg')
    ax=plt.gca()
    ax.invert_xaxis()
    if plobject.vert_axis == 'pressure':
        ax.invert_yaxis()
        ax.set_yscale('log')
    ax.clabel(cs, inline=True)
    cbar = plt.colorbar(cf, orientation='horizontal',location='top')
    cbar.set_label('K')
    plt.show()

# %%
def relative_vorticity(plobject, time_slice=-1):
    """ Use windspharm to calculate relative vorticity """

    winds = VectorWind(-plobject.data['vitu'][time_slice,1:,:,:].values, plobject.data['vitv'][time_slice,1:,:,:].values, rsphere=plobject.radius*1e3)
    vrt = winds.vorticity() # Get relative vorticity

    return vrt

# %%
def dthetadp(plobject, time_slice=-1):
    """ Calculate -g*dtheta/dp term from Garate-Lopez+2016 """

    if not hasattr(plobject, 'theta'):
        plobject.calc_theta() # Make sure we have a theta cube

    dth_dp = plobject.theta[time_slice,1:,:,:].differentiate(coord='presnivs')/plobject.data['pres'][time_slice,1:,:,:].differentiate(coord='presnivs')
    # Gradient of theta with respect to pressure

    theta_term = -plobject.g*dth_dp

    return theta_term


# %%
def ertelspv(plobject, time_slice=-1):
    """ Calculate Ertel's potential vorticity
        Following Garate-Lopez+2016, equation 3 """
    
    vrt = relative_vorticity(plobject, time_slice=time_slice)
    theta_term = dthetadp(plobject, time_slice=time_slice)

    pv = vrt*theta_term

    return pv

# %%
def polarpv(plobject, var='epv', lev=5, time_slice=-1):
       
    """ Project Ertel's PV onto south pole 
        Level 5 of isentropes dataset is 330 K isentrope """
    labels = {'epv':{'titleterm':'Ertel\'s potential vorticity','unit': 'PVU'},
              'vrt':{'titleterm':'Relative vorticity','unit':'s-1'},
              'theta_term':{'titleterm':'-g dtheta/dp','unit':'K kg-1 m-2'},
              'temp':{'titleterm':'Temperature','unit':'K'}}
    
    if var=='epv':        
        cube = ertelspv(plobject, time_slice=time_slice)
    elif var=='vrt':
        cube = relative_vorticity(plobject, time_slice=time_slice)
    elif var=='theta_term':
        cube = dthetadp(plobject, time_slice=time_slice)
    elif var=='temp':
        cube = plobject.data['temp'][time_slice,:,:,:]

    lon = np.linspace(-180, 180, len(plobject.lons))
    lat = np.linspace(-90, 90, len(plobject.lats))
    # Make a lon-lat grid based on the number of lon/lat columns (model resolution)

    ortho = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
    # Specify orthographic projection centered at lon/lat for north pole
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('white') # Background colour
    ax = plt.axes(projection=ortho)
    ax.set_global()
    ax.gridlines()
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    add_circle_boundary(ax)

    plimg = ax.contourf(lon, lat, cube[lev,:,:], 
                        transform=ccrs.PlateCarree(), 
    #                    levels=levels,
                        cmap='RdBu_r')
    ax.set_title(labels[var]['titleterm'] + f', h={plobject.levs[lev]} {plobject.vert_unit}', 
                 color='black',
                 y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label(labels[var]['unit'], color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
    # Create the contourfill plot and colorbar
    plt.show()


# %%
