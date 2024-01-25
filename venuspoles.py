# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs

# %%
def pvsnap(plobject, lev=40, time_slice=-1):
    """ Potential vorticity at north and south pole for one time output """

    pfactor = ((100000/plobject.data['presnivs'][:])**(2/7))
    theta = plobject.data['temp'][time_slice,:,:,:]*pfactor[:,np.newaxis,np.newaxis]

    theta_grad = np.gradient(theta, axis=0)
   
    pvcube = (plobject.data['eta'][time_slice,lev,:,:]*theta_grad[lev,:,:])/plobject.rhoconst

    lon = np.linspace(-180, 180, len(plobject.lons))
    lat = np.linspace(-90, 90, len(plobject.lats))
    # Make a lon-lat grid based on the number of lon/lat columns (model resolution)

    ortho = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
    # Specify orthographic projection centered at lon/lat for north pole
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=ortho)
    ax.set_global()
    # Create the figure
    levels=np.linspace(-0.0003,0.0003,30)
    plimg = ax.contourf(lon, lat, pvcube, transform=ccrs.PlateCarree(), 
                        levels=levels,
                        cmap='RdBu', norm=TwoSlopeNorm(0))
    ax.set_title(f'Potential vorticity of north pole, h={plobject.heights[lev]} km', 
                 color='white',
                 y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('s-1', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    # Create the contourfill plot and colorbar
    plt.show()
# %%
