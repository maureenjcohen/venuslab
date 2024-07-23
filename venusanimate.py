# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

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
                        levels=levels,
                        cmap=inputcols)
    ax.set_title(ptitle + ', ' + htitle, color='white', y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label(cunit, color='white')
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
                 inputcols, ptitle, cunit, cmin, cmax):

    levels = np.linspace(cmin, cmax, 100)
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(lons, lats, inputarray[time_slice,lev,:,:], 
 #                levels=levels,
                 cmap=inputcols, extend='max')
    plt.title(f'{ptitle}, h={heights[lev]} km')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')

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
def animate_lonlat(inputarray, lons, lats, heights, lev, trange=(0,4499,50), 
                   inputcols='cividis',
                   ptitle='Age of air', cunit='seconds',
                   cmin=0.0, cmax=14.5,
                   savename='lonlat.gif'):

    im = []
    for t in range(trange[0], trange[1], trange[2]):
        frame_shot = lonlat_frame(inputarray, lons, lats, heights, lev, 
                                  t, inputcols, ptitle, cunit, cmin, cmax)
        im.append(frame_shot)

    im[0].save(savename, save_all=True, append_images=im[1:], optimize=False,
            duration=1, loop=1)

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
