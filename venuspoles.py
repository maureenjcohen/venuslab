# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs

# %%
def pvsnap(plobject, lev=27, time_slice=-1):
    """ Potential vorticity at north and south pole for one time output """

    pfactor = ((100000/plobject.data['presnivs'][:])**(2/7))
    theta = plobject.data['temp'][time_slice,:,:,:]*pfactor[:,np.newaxis,np.newaxis]

    theta_grad = np.gradient(theta, axis=0)
   
    pvcube = (plobject.data['eta'][time_slice,lev,:,:]*theta_grad[lev,:,:])/plobject.rhoconst

    lon = np.linspace(-180, 180, len(plobject.lons))
    lat = np.linspace(-90, 90, len(plobject.lats))
    # Make a lon-lat grid based on the number of lon/lat columns (model resolution)

    ortho = ccrs.Orthographic(central_longitude=0, central_latitude=0)
    # Specify orthographic projection centered at lon/lat for north pole
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=ortho)
    ax.set_global()
    # Create the figure
    levels=np.linspace(-3.5,3.5,30)
    plimg = ax.contourf(lon, lat, pvcube, transform=ccrs.PlateCarree(), 
#                        levels=levels,
                        cmap='RdBu', norm=TwoSlopeNorm(0))
    ax.set_title(f'Potential vorticity, h={plobject.heights[lev]} km', 
                 color='white',
                 y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('s-1', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    # Create the contourfill plot and colorbar
    plt.show()

# %%
def zetasnap(plobject, lev=27, time_slice=-1):
    """ Relative vorticity at north and south pole for one time output """
   
    zetacube = plobject.data['zeta'][time_slice,lev,:,:]
    lon = np.linspace(-180, 180, len(plobject.lons))
    lat = np.linspace(-90, 90, len(plobject.lats))
    # Make a lon-lat grid based on the number of lon/lat columns (model resolution)
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
    # Specify projection 
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.gridlines()
    # Create the figure
    levels=np.linspace(-16,8,40)
    plimg = ax.contourf(lon, lat, zetacube*1e05, transform=ccrs.PlateCarree(), 
                        levels=levels,
                        cmap='RdBu', norm=TwoSlopeNorm(0))
    ax.set_title(f'Relative vorticity, h={plobject.heights[lev]} km', 
                 color='white',
                 y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('10$^{-5}$ s-1', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    # Create the contourfill plot and colorbar
    plt.show()

# %%
def vortex_vectors(plobject, lev=27, time_slice=-1, n=2, qscale=1):

    crs = ccrs.RotatedPole(pole_longitude=0, pole_latitude=-90)
    lon = np.linspace(-180, 180, len(plobject.lons))
    lat = np.linspace(-90, -60, len(plobject.lats))

    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax.set_global()
    ax.gridlines() 

    ax.quiver(lon[::n], lat[::n], 
              plobject.data['vitu'][time_slice,lev,::n,::n],
              plobject.data['vitv'][time_slice,lev,::n,::n], 
              transform=crs,
              angles='xy', scale_units='xy', scale=qscale)  
    plt.show()
# %%
