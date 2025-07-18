# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import io
from venuspoles import add_cycl_point

# %%
def sphere(inputarray, lons, lats, i, j, inputcols, ptitle, htitle, cunit,
           cmin, cmax):

    """ Creates a plot of a 2-D data array projected onto a sphere
        inputarray: The input data array. Must be 2-D
        lons: List of longitudes
        lats: List of latitudes 
        i: Initial longitude at centre of plot
        j: Initial latitude at centre of plot
        inputcols: Colormap key word
        ptitle: Plot title string
        cunit: Colorbar unit"""

    lon = np.linspace(-180, 180, len(lons))
    lat = np.linspace(-90, 90, len(lats))
    # Make a lon-lat grid based on the number of lon/lat columns (model resolution)

    ortho = ccrs.Orthographic(central_longitude=i, central_latitude=j)
    # Specify orthographic projection centered at lon/lat i, j
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=ortho)
    ax.set_global()
    # Create the figure
    levels=np.linspace(cmin, cmax, 30)
    plimg = ax.contourf(lon, lat, inputarray, transform=ccrs.PlateCarree(), 
    #                    levels=levels,
                        cmap=inputcols)
    ax.set_title(ptitle + ', ' + htitle, color='white', y=1.05, fontsize=14)
    ax.gridlines(draw_labels=True, linewidth=1.5, color='silver', alpha=0.5)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label(cunit, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    # Create the contourfill plot and colorbar
    # plt.show()
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
def animate_sphere(inputarray, lons, lats, j, irange=(0, 360, 30), inputcols='hot', 
                   ptitle='Air temperature', htitle='Height in km', cunit='W/m2', 
                   cmin=190, cmax=230,
                   savename='test.gif'):
    
    """ Take 2-D orthographic projections onto a globe and turn them into a
    rotating animation. Args as the same as for the function sphere() defined
    above, except for:
    irange: Specifies initial central longitude, final central longitude, and
    how many lons to step between frames"""

    im = []
    for i in range(irange[0], irange[1], irange[2]):
        frame_shot = sphere(inputarray, lons, lats, i, j, inputcols, ptitle, htitle, 
                            cunit, cmin, cmax)
        im.append(frame_shot)
        # Create PIL Image of each generated plot and append to list

    im[0].save(savename, save_all=True, append_images=im[1:], optimize=False,
               duration=0.5, loop=0)
    # Save our list of frames as a gif

# %%
def time_sphere(inputarray, lons, lats, i, j, trange=(500, 600, 5), inputcols='plasma',
                ptitle='Air temperature', htitle='Height in km', cunit='K',
                cmin=190, cmax=230,
                savename='timelapse.gif'):
    
    im = []
    for t in range(trange[0], trange[1], trange[2]):
        frame_shot = sphere(inputarray[t,:,:], lons, lats, i, j, inputcols, ptitle, htitle,
                            cunit, cmin, cmax)
        im.append(frame_shot)
    # Create PIL Image of each generated plot and append to list

    im[0].save(savename, save_all=True, append_images=im[1:], optimize=False,
               duration=0.5, loop=0)
    # Save our list of frames as a gif

# %%
def lonlat_frame(inputarray, lons, lats, heights, lev, time_slice, 
                 inputcols, ptitle, cunit, clevs, tday,
                 animation=False):

    if clevs is None:
        clevs = np.linspace(np.min(inputarray[time_slice,lev,:,:]), 
                            np.max(inputarray[time_slice,lev,:,:]), 
                            100)
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(lons, lats, inputarray[time_slice,lev,:,:], 
                 levels=clevs,
                 cmap=inputcols, extend='max')
    plt.title(f'{ptitle}, h={np.round(heights[lev],0)} km, day {tday}')
    plt.xlabel('Longitude / deg')
    plt.ylabel('Latitude / deg')
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')

    if animation==True:
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
    else:
        plt.show()

# %%
def animate_lonlat(inputarray, lons, lats, heights, lev, 
                   inputcols,
                   ptitle, cunit,
                   clevs, tdays,
                   savename='lonlat.gif'):

    im = []
    for t in range(0,inputarray.shape[0],1):
        frame_shot = lonlat_frame(inputarray, lons, lats, heights, lev, 
                                  t, inputcols, ptitle, cunit, clevs, tdays[t], 
                                  animation=True)
        im.append(frame_shot)

    im[0].save(savename, save_all=True, append_images=im[1:], optimize=False,
            duration=1000, loop=0)

# %%
def zm_frame(inputarray, lats, hmin, hmax, heights, time_slice, 
                 inputcols, ptitle, cunit, cmin, cmax, animation=True):
    
    zm = np.mean(inputarray[time_slice,:,:,:], axis=-1) 
    levels = np.linspace(cmin, cmax, 30)

    fig = plt.figure(figsize=(6, 6))
    plt.contourf(lats, heights[hmin:hmax], zm[hmin:hmax,:], levels=levels, 
                 cmap=inputcols, extend='max')
    plt.title(f'{ptitle}, zonal mean')
    plt.xlabel('Latitude [deg]')
    plt.ylabel('Height [km]')
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')
    if animation==True:
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
    else:
        plt.show()

# %%
def animate_zm(inputarray, lats, hmin, hmax, heights, cmin, cmax, trange=(0,4499,50), 
                   inputcols='cividis',
                   ptitle='Age of air', cunit='seconds',
                   savename='zm.gif'):

    im = []
    for t in range(trange[0], trange[1], trange[2]):
        frame_shot = zm_frame(inputarray, lats, hmin, hmax, heights, 
                             t, inputcols, ptitle, cunit, cmin, cmax)
        im.append(frame_shot)

    im[0].save(savename, save_all=True, append_images=im[1:], optimize=False,
            duration=0.5, loop=0)

# %%
def animate_akatsuki(snaps, levs, savename='test.gif'):

    ims = []
    for snap in snaps:
        fig = plt.figure(figsize=(8,6))
        plt.contourf(snap['longitude'], snap['latitude'], snap['radiance'][0,:,:], 
                     levels=levs, 
                     cmap='plasma')
        plt.title(f'Akatsuki IR1 camera 0.9 um, {snap.time.values[0]}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        cb = plt.colorbar()
        cb.set_label('mW/cm2/um/sr')

        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img.show()
        buf.close()
        
        ims.append(img)

    ims[0].save(savename, save_all=True, append_images=ims[1:], optimize=False, duration=500, loop=0)

# %%
def animate_globe(data, lev, heights, t, i=0, j=30):

    """ Input:  cube: 4-D xarray or Planet object data cube
                lev: level to be visualised 
                i: central longitude of view
                j: central latitude of view    """
    
    cube_name = data.long_name or data.name
    cube = data[:,lev,:,:]*1e6
    height = np.round(heights[lev],2)
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=i, central_latitude=j))
   # _, clon = add_cycl_point(cube, cube.lon, -1)
    #arguments
    plot_args = {
    'transform':ccrs.PlateCarree(),
    'vmin': cube.quantile(0.01), # 1st percentile
    'vmax': cube.quantile(0.99),
    'cmap': 'plasma'
    }

    default_gridlines_kw = {'linewidth' : 0.5,
            'color' : 'silver',
            'alpha': 0.5, 
            'xlabel_style': {'size':8},
            'ylabel_style': {'size':8}
            }
    
    # Define an update function that will be called for each frame
    def animate(frame):
        plimg = ax.contourf(cube.lon, cube.lat, cube[frame,:,:], **plot_args)

    ax.set_title(cube_name+ ', ' + str(height) + ' km', color='white', y=1.05, fontsize=14)
    ax.gridlines(draw_labels=True, **default_gridlines_kw)
    
    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=range(0,t), interval=200, repeat=False)

    #define the colorbar. The colorbar method needs a mappable object from which to take the colorbar
    cbar = fig.colorbar(ax.contourf(cube.lon, cube.lat, cube[0,:,:], **plot_args))
    cbar.set_label('ppm', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    #plt.show()

    # Save the animation as an mp4 file
    ani.save(f'{cube_name}_{height}km.mp4', writer='ffmpeg')
    # ani.save('myanimation.gif', writer='pillow') #alternative
# %%
