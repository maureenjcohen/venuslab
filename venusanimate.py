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
def animate_globe(data, lev, heights, tf, t0=0, i=0, j=30,
                  savepath='/exomars/projects/mc5526/VPCM_full_chemistry_runs/scratch_plots/'):

    """ Input:  data: 4-D xarray or Planet object data cube
                lev: level to be visualised 
                heights: list of model heights (attr of PlObject)
                tf: final frame
                t0: first frame (default 0)
                i: central longitude of view
                j: central latitude of view   
                savepath: where to save the output """
    
    cube_name = data.long_name or data.name
    # Extract name to automatically title plot
    cube = data[:,lev,:,:]*1e6
    # Extract level data and convert to ppm
    height = np.round(heights[lev],2)
    # Get height in km rounded to 2 decimal points (for title)

    fig = plt.figure(figsize=(8, 6))
    # Create figure
    fig.patch.set_facecolor('black') 
    # Background colour
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=i, central_latitude=j))
    # Create axis with orthographic proj with given central lon and lat
    new_cube, new_lon = add_cycl_point(cube, cube.lon, -1)
    # Add cyclical point to fix discontinuity in longitudes

    # Dictionary of values for plot frames
    plot_args = {
    'transform':ccrs.PlateCarree(),
    'vmin': cube.quantile(0.01), # 1st percentile
    'vmax': cube.quantile(0.99), # 99th percentile
    'cmap': 'plasma'
    }
    # Dictionary of values for ax gridlines method
    default_gridlines_kw = {'linewidth' : 0.5,
            'color' : 'silver',
            'alpha': 0.5, 
            'xlabel_style': {'size':8},
            'ylabel_style': {'size':8}
            }
    
    # Define an update function that will be called for each frame
    def animate(frame):
        plimg = ax.contourf(new_lon, cube.lat, new_cube[frame,:,:], **plot_args)

    ax.set_title(cube_name+ ', ' + str(height) + ' km', color='white', y=1.05, fontsize=14)
    ax.gridlines(draw_labels=True, **default_gridlines_kw)
    
    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=range(t0,tf), interval=200, repeat=False)

    #Define the colorbar. The colorbar method needs a mappable object from which to take the colorbar
    cbar = fig.colorbar(ax.contourf(new_lon, cube.lat, new_cube[0,:,:], **plot_args))
    cbar.set_label('ppm', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    #plt.show()

    # Save the animation as an mp4 file
    ani.save(savepath + f'{cube_name}_{height}km.mp4', writer='ffmpeg')
    # ani.save('myanimation.gif', writer='pillow') #alternative

# %%
def animate_plume(plobject, key, lev, t0, tf, n=4, qscale=0.5,
                  savepath='/exomars/projects/mc5526/VPCM_volcanic_plumes/scratch_plots/'):

    """ Input:  plobject: Planet class object containing the data
                key: Dict key of data, either 'age' or 'aoa'
                lev: level to be visualised 
                tf: final frame
                t0: first frame  
                savepath: where to save the output 
                
                For upper atmos, try n=4, qscale=4"""
    
    height = np.round(plobject.heights[lev],2)
    # Get height in km rounded to 2 decimal points (for title)
    cube_name = plobject.data[key].long_name
    cube = plobject.data[key][t0:tf,lev,:,:]
    # Extract data for desired altitude
    interval = np.diff(cube.time_counter.values)[0]/(60*60)
    time_axis = np.round(np.arange(0,tf-t0)*interval,0)

    if key=='aoa':
        cube = cube*1e9
        unit = 'mmr ppb'
    elif key=='age':
        cube = cube/(60*60*24*360)
        unit = 'years'
    else:
        cube = cube*1e6
        unit = 'vmr ppm'

    u = plobject.data['vitu'][t0:tf,lev,:,:]
    v = plobject.data['vitv'][t0:tf,lev,:,:]
    # Extract zonal and meridional wind for desired altitude

    fig, ax = plt.subplots(figsize=(8, 6))
    # Create figure
    X, Y = np.meshgrid(plobject.lons, plobject.lats)

    # Dictionary of values for plot frames
    plot_args = {
    'vmin': cube.min(), # Min of plotted frames
    'vmax': cube.max(), # Max of plotted frames
    'cmap': 'viridis',
    'extend': 'neither'
    }
 
    quiv_args = {
    'angles': 'xy',
    'scale_units': 'xy',
    'scale': qscale,
    'color': 'white'
    }
    
    # Define an update function that will be called for each frame
    def animate(frame):
        cf = ax.contourf(plobject.lons, plobject.lats, cube[frame,:,:], **plot_args)
        q = ax.quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax.set_title(f'{cube_name}, ' + str(height) + f' km, {time_axis[frame]} hrs', color='black', y=1.05, fontsize=14)

    
    ax.quiverkey(ax.quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes', color='black')
    ax.set_title(f'{cube_name}, ' + str(height) + ' km', color='black', y=1.05, fontsize=14)
    ax.set_xlabel('Longitude / deg')
    ax.set_ylabel('Latitude / deg')
    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=range(0,tf-t0), interval=200, repeat=False)

    mask = np.where(cube >= cube.max())
    start_time = mask[0][0]
    #Define the colorbar. The colorbar method needs a mappable object from which to take the colorbar
    cbar = plt.colorbar(ax.contourf(plobject.lons, plobject.lats, cube[start_time,:,:], **plot_args))
    cbar.set_label(unit, color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
    
    #plt.show()

    # Save the animation as an mp4 file
    ani.save(savepath + f'{key}_{height}km.mp4', writer='ffmpeg')
    # ani.save('myanimation.gif', writer='pillow') #alternative

# %%
def animate_chem_plume(plobject, lev, t0, tf, n=4, qscale=1,
                  savepath='/exomars/projects/mc5526/VPCM_volcanic_plumes/scratch_plots/'):

    """ Input:  plobject: Planet class object containing the data
                key: Dict key of data, chemistry cube
                lev: level to be visualised 
                tf: final frame
                t0: first frame  
                savepath: where to save the output """
    
    height = np.round(plobject.heights[lev],2)
    # Get height in km rounded to 2 decimal points (for title)
    h2o_cube = plobject.data['h2o'][t0:tf,lev,:,:]*1e6
    co_cube = plobject.data['co'][t0:tf,lev,:,:]*1e6
    ocs_cube = plobject.data['ocs'][t0:tf,lev,:,:]*1e6
    hcl_cube = plobject.data['hcl'][t0:tf,lev,:,:]*1e6
    # Extract data for desired altitude

    u = plobject.data['vitu'][t0:tf,lev,:,:]
    v = plobject.data['vitv'][t0:tf,lev,:,:]
    # Extract zonal and meridional wind for desired altitude

    fig, ax = plt.subplots(2,2,figsize=(16, 10), sharex=True, sharey=True)
    # Create figure
    X, Y = np.meshgrid(plobject.lons, plobject.lats)

    interval = np.diff(h2o_cube.time_counter.values)[0]/(60*60)
    time_axis = np.round(np.arange(0,tf-t0)*interval,0)
 
    quiv_args = {
    'angles': 'xy',
    'scale_units': 'xy',
    'scale': qscale,
    'color': 'black'
    }
    
    # Define an update function that will be called for each frame
    def animate(frame):
        # H2O plot
        cf_h2o = ax[0,0].contourf(plobject.lons, plobject.lats, h2o_cube[frame,:,:], 
                                  cmap='Blues', 
                                  vmin=np.min(h2o_cube), vmax=45.0)
        q1 = ax[0,0].quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax[0,0].quiverkey(ax[0,0].quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes', color='black')
        ax[0,0].set_title('H2O', color='black', y=1.05, fontsize=14)
        ax[0,0].set_ylabel('Latitude / deg')

        # CO plot
        cf_co = ax[0,1].contourf(plobject.lons, plobject.lats, co_cube[frame,:,:], 
                                  cmap='Purples', vmin=np.min(co_cube), vmax=37.5)
        q2 = ax[0,1].quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax[0,1].quiverkey(ax[0,1].quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes', color='black')
        ax[0,1].set_title('CO', color='black', y=1.05, fontsize=14)

        # OCS plot
        cf_ocs = ax[1,0].contourf(plobject.lons, plobject.lats, ocs_cube[frame,:,:], 
                                  cmap='YlOrBr', vmin=np.min(ocs_cube), vmax=4.5)
        q3 = ax[1,0].quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax[1,0].quiverkey(ax[1,0].quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes', color='black')
        ax[1,0].set_title('OCS', color='black', y=1.05, fontsize=14)
        ax[1,0].set_ylabel('Latitude / deg')
        ax[1,0].set_xlabel('Longitude / deg')

        # HCl plot
        cf_hcl = ax[1,1].contourf(plobject.lons, plobject.lats, hcl_cube[frame,:,:], 
                                  cmap='Reds', vmin=np.min(hcl_cube), vmax=0.6)
        q4 = ax[1,1].quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax[1,1].quiverkey(ax[1,1].quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes', color='black')
        ax[1,1].set_title('HCl', color='black', y=1.05, fontsize=14)
        ax[1,1].set_xlabel('Longitude / deg')
        plt.subplots_adjust(wspace=0.1)
        fig.suptitle(f'Volcanic plume at {height} km, {time_axis[frame]} hrs', y=0.97, fontsize=24)

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=range(0,tf-t0), interval=200, repeat=False)
    
    cbar_h2o = plt.colorbar(ax[0,0].contourf(plobject.lons, plobject.lats, h2o_cube[0,:,:], 
                                             cmap='Blues', vmin=np.min(h2o_cube), vmax=45.0), 
                                             ax=ax[0,0])
    cbar_h2o.set_label('ppm', color='black')
    cbar_co = plt.colorbar(ax[0,1].contourf(plobject.lons, plobject.lats, co_cube[0,:,:], 
                                             cmap='Purples', vmin=np.min(co_cube), vmax=37.5), ax=ax[0,1])
    cbar_co.set_label('ppm', color='black')
    cbar_ocs = plt.colorbar(ax[1,0].contourf(plobject.lons, plobject.lats, ocs_cube[0,:,:], 
                                             cmap='YlOrBr', vmin=np.min(ocs_cube), vmax=4.5), ax=ax[1,0])
    cbar_ocs.set_label('ppm', color='black')
    cbar_hcl = plt.colorbar(ax[1,1].contourf(plobject.lons, plobject.lats, hcl_cube[0,:,:], 
                                             cmap='Reds', vmin=np.min(hcl_cube), vmax=0.6), ax=ax[1,1])
    cbar_hcl.set_label('ppm', color='black')

    # Save the animation as an mp4 file
    ani.save(savepath + f'deep_plume_{height}km.mp4', writer='ffmpeg')
    # ani.save('myanimation.gif', writer='pillow') #alternative
# %%
