# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy
import matplotlib.path as mpath

# %%
def add_circle_boundary(ax):
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

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
def vortex_vectors(plobject, levs=[20,27,33], time_slice=-1, qscale=500):

    u = plobject.data['vitu'][time_slice,:,:,:]
    v = plobject.data['vitv'][time_slice,:,:,:]
    wind_speed = np.sqrt(u**2 + v**2)
    print(wind_speed.shape)

    X, Y = np.meshgrid(plobject.lons, plobject.lats)

    fig, ax = plt.subplots(1, len(levs),
                           subplot_kw={'projection': ccrs.Orthographic(0,-90)},
                           figsize=(12,6))

    heights = []
    for l in levs:
        h = plobject.heights[l]
        heights.append(h)
   
    for lev in range(0,len(levs)):
        level_height = heights[lev]
        print(level_height)
        ws = ax[lev].imshow(wind_speed[levs[lev],:,:], 
                        cmap='YlOrBr',
                        transform=ccrs.PlateCarree())

        ax[lev].quiver(X, Y, u[levs[lev],:,:], v[levs[lev],:,:],
                       scale=qscale,
                       headlength=5,
                       headwidth=3,
                       minlength=2,
                       transform=ccrs.PlateCarree(),
                       width=0.009,
                       regrid_shape=20)
        
        ax[lev].set_title(f'{level_height} km', size=14)
        ax[lev].gridlines()
        ax[lev].set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
        add_circle_boundary(ax[lev])
        print(f'Level {level_height} added')

    cbar_ax = fig.add_axes([0.15, 0.2, 0.75, 0.02])
    cbar = fig.colorbar(ws, cax=cbar_ax, orientation='horizontal',
                        extend='both')
    cbar.set_label(label='Wind speed [m/s]', size=14)
    fig.suptitle('Venusian southern polar vortex', size=18, y=0.85)
    plt.show()
        


    
# %%

