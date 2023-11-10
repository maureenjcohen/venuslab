# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# %%
def sphere(inputarray, lons, lats, i, j, inputcols, ptitle, cunit):

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

    plimg = ax.contourf(lon, lat, inputarray, transform=ccrs.PlateCarree(), 
                        cmap=inputcols)
    ax.set_title(ptitle, color='white', y=1.05, fontsize=14)
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
def animate_sphere(inputarray, lons, lats, irange=(0, 360, 30), j, inputcols='hot', 
                   ptitle='Surface radiation', cunit='W/m2',
                   savename='test.gif'):
    
    """ Take 2-D orthographic projections onto a globe and turn them into a
    rotating animation. Args as the same as for the function sphere() defined
    above, except for:
    irange: Specifies initial central longitude, final central longitude, and
    how many lons to step between frames"""

    im = []
    for i in range(irange[0], irange[1], irange[2]):
        frame_shot = sphere(inputarray, lons, lats, i, j, inputcols, ptitle, cunit)
        im.append(frame_shot)
        # Create PIL Image of each generated plot and append to list

    im[0].save(savename, save_all=True, append_images=im[1:], optimize=False,
               duration=0.5, loop=0)
    # Save our list of frames as a gif

# %%
