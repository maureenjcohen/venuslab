# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy
import matplotlib.path as mpath
from PIL import Image
import io

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

    ortho = ccrs.Orthographic(central_longitude=0, central_latitude=0)
    # Specify orthographic projection centered at lon/lat for north pole
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=ortho)
    ax.set_global()
    ax.gridlines()
#    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
#    add_circle_boundary(ax)
 
    levels=np.linspace(-0.0003,0.0003,30)
    plimg = ax.contourf(lon, lat, pvcube, transform=ccrs.PlateCarree(), 
#                        levels=levels,
                        cmap='RdBu', norm=TwoSlopeNorm(0))
    ax.set_title(f'Potential vorticity, h={plobject.heights[lev]} km', 
                 color='white',
                 y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('PVU', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    # Create the contourfill plot and colorbar
    plt.show()

# %%
def zetasnap(plobject, cmin, cmax, lev=28, time_slice=-1):
    """ Relative vorticity at north and south pole for one time output """
   
    zetacube = plobject.data['zeta'][time_slice,lev,:,:]
    lon = np.linspace(-180, 180, len(plobject.lons))
    lat = np.linspace(-90, 90, len(plobject.lats))
    # Make a lon-lat grid based on the number of lon/lat columns (model resolution)
    plev = np.round(plobject.data['presnivs'][lev]*0.01) # Pressure in mb

    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
    # Specify projection 
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.gridlines()
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    add_circle_boundary(ax)
    # Create the figure
    levels=np.linspace(cmin,cmax,20)
    plimg = ax.contourf(lon, lat, zetacube*1e05, transform=ccrs.PlateCarree(), 
                        levels=levels,
                        cmap='coolwarm', norm=TwoSlopeNorm(0))
    ax.set_title(f'Relative vorticity, h={plev} mb', 
                 color='white',
                 y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('10$^{-5}$ s-1', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    # Create the contourfill plot and colorbar

    # The code block below creates a buffer and saves the plot to it.
    # This avoids having to actually save the plot to the hard drive.
    # We then reopen the 'saved' figure as a PIL Image object and output it.
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img.show()
    buf.close()

    return img

# %%
def vortex_vectors(plobject, key='zeta', levs=[25,30,35], time_slice=-1, n=2, qscale=500):

    u = plobject.data['vitu'][time_slice,:,:,:]
    v = plobject.data['vitv'][time_slice,:,:,:]

    if key=='wind':
        imcube = np.sqrt(u**2 + v**2)
        colmap = 'inferno_r'
        clabel = 'Wind speed [m/s]'
        cmin, cmax = 20, 160
    elif key=='temp':
        imcube = plobject.data['temp'][time_slice,:,:,:]
        colmap = 'Spectral_r'
        clabel = 'Air temperature [K]'
        cmin, cmax = 220, 350
    elif key=='zeta':
        imcube = plobject.data['zeta'][time_slice,:,::-1,:]*1e05
        colmap='coolwarm'
        clabel='Relative vorticity [10$^{-5}$ s$^{-1}$]'
        cmin, cmax = -5, 5
    else:
        print(f'{key} is not a valid input.')
        print('Enter wind or temp.')

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
        im = ax[lev].imshow(imcube[levs[lev],:,:], 
                        cmap=colmap,
                        transform=ccrs.PlateCarree(),
                        vmin=cmin,
                        vmax=cmax)

        ax[lev].quiver(X[::n,::n], Y[::n,::n], 
                       u[levs[lev],::n,::n], v[levs[lev],::n,::n],
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
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                        extend='both')
    cbar.set_label(label=clabel, size=14)
    fig.suptitle('Southern polar vortex', size=18, y=0.85)
    plt.show()
        


    
# %%
def animate_poles(plobject, lev, trange,
                  cmin, cmax,
                  savename='zeta_southpole.gif'):
    
    """ Function for animating the above plots,
    just insert whichever of the plotting
    functions above you want """
    
    im = []
    for t in range(trange[0], trange[1], trange[2]):
        frame_shot = zetasnap(plobject, lev=lev, time_slice=t,
                              cmin=cmin, cmax=cmax)
    # zetasnap, pvsnap, vortex_vectors, etc. goes above
        im.append(frame_shot)
    # Create PIL Image of each generated plot and append to list

    im[0].save(savename, save_all=True, append_images=im[1:], optimize=False,
               duration=0.5, loop=0)
    # Save our list of frames as a gif


# %%
