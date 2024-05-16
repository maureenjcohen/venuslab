# %%
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import matplotlib.path as mpath
from PIL import Image
import io


## Utils
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
def add_cycl_point(data, coord=None, axis=-1):
        
    """ Ripped from cartopy but removing requirement for
        data to be equally spaced"""

    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError(f'The length of the coordinate does not match '
                             f'the size of the corresponding dimension of '
                             f'the data array: len(coord) = {len(coord)}, '
                             f'data.shape[{axis}] = {data.shape[axis]}.')
        delta_coord = np.diff(coord)
#        if not np.allclose(delta_coord, delta_coord[0]):
#            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value




# %%
def pvsnap(plobject, lev=40, time_slice=-1):
    """ Potential vorticity at north and south pole for one time output """

    k = 188/900
    # Gas constant for CO2, divided by specific heat used in OASIS sim
    # See Mendonca & Buchhave 2020 Table 2
    pfactor = ((100000/plobject.data['presnivs'][:])**(k))
    theta = plobject.data['temp'][time_slice,:,:,:]*pfactor[:,np.newaxis,np.newaxis]

    theta_grad = np.gradient(theta, axis=0)/np.gradient(plobject.plevs)[:,np.newaxis,np.newaxis]
    pvcube = -plobject.data['zeta'][time_slice,lev,:,:]*theta_grad[lev,:,:]*plobject.g

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
 
    levels=np.linspace(-0.6,0.6,12)
    print(np.max(pvcube[:2,:]), np.min(pvcube[:2,:]))
    plimg = ax.contourf(lon, lat, pvcube, 
                        transform=ccrs.PlateCarree(), 
    #                    levels=levels,
                        cmap='RdBu_r', norm=TwoSlopeNorm(0))
    ax.set_title(f'Potential vorticity, h={plobject.heights[lev]} km', 
                 color='black',
                 y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('PVU', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
    # Create the contourfill plot and colorbar
    plt.show()

# %%
def zetasnap(plobject, cmin=-28, cmax=5, lev=30, time_slice=-1,
             animation=False):
    """ Relative vorticity at north and south pole for one time output """
   
    zetacube = plobject.data['zeta'][time_slice,lev,:,:]
#    lon = np.linspace(-180, 180, len(plobject.lons))
#    lat = np.linspace(-90, 90, len(plobject.lats))
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
    plimg = ax.contourf(plobject.lons, plobject.lats, zetacube*1e05, 
                        transform=ccrs.PlateCarree(), 
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
def calc_zeta(plobject, lev, time_slice):

    """ Calculate relative vorticity"""

    v = plobject.data['vitv'][time_slice,lev,:,:]
    u = plobject.data['vitu'][time_slice,lev,:,:]
    
    xlon, ylat = np.meshgrid(plobject.lons, plobject.lats)
    dlat = np.deg2rad(np.abs(np.gradient(ylat, axis=0)))
    dlon = np.deg2rad(np.abs(np.gradient(xlon, axis=1)))
    rad = plobject.radius*1e3
    dy = dlat*rad
    dx = dlon*rad*(np.cos(np.deg2rad(ylat)))

    dvdx = np.gradient(v, axis=-1)/dx
    dudy = np.gradient(u, axis=-2)/dy

    zeta = dvdx - dudy

    return zeta

 # %%
def calc_dtau(plobject, lev, time_slice):

    """ Calculate vertical gradient of theta x gravity      
        This is for direct comparision with Garate-Lopez et al. 2016 """
    
    k = 188/900
    # Gas constant for CO2, divided by specific heat used in OASIS sim
    # See Mendonca & Buchhave 2020 Table 2
    pfactor = ((100000/plobject.data['presnivs'][:])**(k))
    theta = plobject.data['temp'][time_slice,:,:,:]*pfactor[:,np.newaxis,np.newaxis]

    dtheta_dp = np.gradient(theta, axis=0)/np.gradient(plobject.plevs)[:,np.newaxis,np.newaxis]
    dtau = -dtheta_dp[lev,:,:]*plobject.g

    return dtau

# %%
def calc_pv(plobject, lev, time_slice):

    """ Calculate Ertel's potential vorticity approximated
    as in Garate-Lopez et al. 2016              """

    zeta = calc_zeta(plobject=plobject, lev=lev, time_slice=time_slice)
    dtau = calc_dtau(plobject=plobject, lev=lev, time_slice=time_slice)

    pvcube = zeta*dtau

    return pvcube

# %%
def polarsnap(plobject, key, lev, time_slice=-1,
               animation=False):
    
    """ Display snapshot of desired quantity centred on south pole
        in orthographic projection"""
    
    cubedict = {'zeta': {'levels': np.linspace(-12,12,20),
                         'title': 'Relative vorticity',
                         'unit': '10$^{-5}$ s-1',
                         'cmap': 'coolwarm'},
                'dtau': {'levels': np.linspace(0, 3.8, 20),
                         'title': '-g dtheta/dp',
                         'unit': '10$^{-2}$ K kg-1 m-2',
                         'cmap': 'coolwarm'},
                'pv':   {'levels': np.linspace(-1.6, 4.8, 20),
                         'title': 'Ertel potential vorticity',
                         'unit': 'PVU',
                         'cmap': 'coolwarm'},
                'v':    {'levels': np.linspace(-30,38,40),
                         'title': 'Meridional wind',
                         'unit': 'm/s',
                         'cmap': 'RdBu_r'},
                'u':    {'levels': np.linspace(-100,0,40),
                         'title': 'Zonal wind',
                         'unit': 'm/s',
                         'cmap': 'RdBu'},
                'temp': {'levels': np.linspace(230,245,40),
                         'title': 'Air temperature',
                         'unit': 'K',
                         'cmap': 'hot'},
                'geop': {'levels': np.linspace(55000,65000,100),
                         'title': 'Geopotential height',
                         'unit': 'm',
                         'cmap': 'hot'},
                'age': {'levels': np.linspace(0,30,1),
                        'title': 'Age of air',
                        'unit': 'years',
                        'cmap': 'cividis'}}
    
    if key=='zeta':
        cube = calc_zeta(plobject=plobject, lev=lev, time_slice=time_slice)
#        cube = plobject.data['zeta'][time_slice,lev,:,:]
        cube = cube*1e5
    elif key=='dtau':
        cube = calc_dtau(plobject=plobject, lev=lev, time_slice=time_slice)
        cube = cube*1e2
    elif key=='pv':
        cube = calc_pv(plobject=plobject, lev=lev, time_slice=time_slice)
    elif key=='v':
        cube = plobject.data['vitv'][time_slice,lev,:,:]
    elif key=='u':
        cube = plobject.data['vitu'][time_slice,lev,:,:]
    elif key=='temp':
        cube = plobject.data['temp'][time_slice,lev,:,:]
    elif key=='geop':
        cube = plobject.data['geop'][time_slice,lev,:,:]
    elif key=='age':
        cube = plobject.data['age'][time_slice,lev,:,:]
        cube = cube/(60*60*24*360)
    
    proj = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
    # Specify projection 
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.gridlines()
    ax.set_extent([0, 360, -90, -60], crs=ccrs.PlateCarree())
    add_circle_boundary(ax)
    # Create the figure
    cube, clon = add_cycl_point(cube, plobject.lons, -1)
 
    plimg = ax.contourf(clon, plobject.lats, cube, 
                        transform=ccrs.PlateCarree(), 
#                        levels=cubedict[key]['levels'],
                        cmap=cubedict[key]['cmap'], 
#                        norm=TwoSlopeNorm(0)
                        )
    ax.set_title(cubedict[key]['title'] + 
                 f', {np.round(plobject.plevs[lev]*0.01,0)} mbar', 
                 color='white',
                 y=1.05, fontsize=14)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label(cubedict[key]['unit'], color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    # Create the contourfill plot and colorbar

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
        h = plobject.plevs[l]
        heights.append(h)
   
    for lev in range(0,len(levs)):
        level_height = np.round(heights[lev]*0.01,0)
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
        
        ax[lev].set_title(f'{level_height} mbar', size=14)
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
def contour_comparison(plobject, lev=30, time_slice=-1):

    """ Plot values of input fields at the southern pole at
        the same time and model level                   """
    
    air_temp = plobject.data['temp'][time_slice,lev,:,:]
#    zm_temp = np.mean(air_temp, axis=-1)
#    eddy_temp = air_temp - zm_temp[:,np.newaxis]
    div = plobject.data['div'][time_slice,lev,:,:]
    rel_vort = plobject.data['zeta'][time_slice,lev,:,:]
    zm_zeta = np.mean(rel_vort, axis=-1)
    eddy_zeta = rel_vort - zm_zeta[:,np.newaxis]

    fig, ax = plt.subplots(1, 3,
                           subplot_kw={'projection': ccrs.Orthographic(0,-90)},
                           figsize=(12,6))
    
    im_temp = ax[0].imshow(air_temp, cmap='Reds',
                      transform=ccrs.PlateCarree())    
    ax[0].set_title('Air temperature [K]', size=14)
    ax[0].gridlines()
    ax[0].set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    add_circle_boundary(ax[0])
    fig.colorbar(im_temp, ax=ax[0], orientation='horizontal')

    im_div = ax[1].imshow(div*1e6, cmap='seismic',
                      transform=ccrs.PlateCarree(), norm=TwoSlopeNorm(0))    
    ax[1].set_title('Divergence [$10^{-6}$ s-1]', size=14)
    ax[1].gridlines()
    ax[1].set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    add_circle_boundary(ax[1])
    fig.colorbar(im_div, ax=ax[1], orientation='horizontal')

    im_zeta = ax[2].imshow(eddy_zeta*1e6, cmap='coolwarm',
                      transform=ccrs.PlateCarree(), norm=TwoSlopeNorm(0))    
    ax[2].set_title('Eddy relative vorticity [$10^{-6}$ s-1]', size=14)
    ax[2].gridlines()
    ax[2].set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    add_circle_boundary(ax[2])
    fig.colorbar(im_zeta, ax=ax[2], orientation='horizontal')

    fig.suptitle(f'Southern polar vortex, {np.round(plobject.plevs[lev]*0.01,0)} mbar', size=18, y=0.9)
    plt.show()

    
# %%
def animate_poles(plobject, key, lev, trange,
                  savename='zeta_southpole.gif'):
    
    """ Function for animating the polarsnap plots """
    
    im = []
    for t in range(trange[0], trange[1], trange[2]):
        frame_shot = polarsnap(plobject, key=key, lev=lev, time_slice=t,
                               animation=True)
    # keys from polarsnap function
        im.append(frame_shot)
    # Create PIL Image of each generated plot and append to list

    im[0].save(savename, save_all=True, append_images=im[1:], optimize=False,
               duration=0.5, loop=0)
    # Save our list of frames as a gif

# %%
def zonal_plot(plobject, key, meaning=True, time_slice=-1, hmin=25, hmax=49,
               latmin=0, latmax=45,
               save=False, savename='zm_pole.png', saveformat='png'):
    
    """ Plot zonal mean temperature or geopotential height """

    cube = plobject.data[key][:,hmin:hmax,latmin:latmax,:]
    cubedict = {'temp': { 'title': 'Air temperature (zonal mean)', 
                          'levels': np.arange(160,320,4),
                          'unit': 'K'},
                'geop' : {'title': 'Geopotential height (zonal mean)',
                          'levels': np.arange(50000,90000,500),
                          'unit': 'm'}}

    if meaning==True:
        tcube = np.mean(cube, axis=0)
    else:
        tcube = cube[time_slice,:,:]

    zcube= np.mean(tcube, axis=-1)
    fig, ax = plt.subplots(figsize=(6,6))
    plt.contourf(plobject.lats[latmin:latmax], plobject.heights[hmin:hmax], 
                 zcube, 
#                 levels=cubedict[key]['levels'],
                 cmap='hot')
    plt.title(cubedict[key]['title'])
    plt.xlabel('Latitude [deg]')
    plt.ylabel('Pressure [mbar]')
#    ax.set_yscale('log')
#    plt.gca().invert_yaxis()
    cbar = plt.colorbar()
    cbar.set_label(cubedict[key]['unit'])
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def alt_lon(plobject, key='eddy temp', time_slice=-1,
            hmin=28, hmax=45, lat=85,
            save=False, savename='alt_lon.png', saveformat='png'):

    """ Create altitude-longitude plot of the cube identified
    by the input key for the selected input time slice and
    model level numbers."""

    if key=='temp':
        cube = plobject.data['temp'][time_slice,hmin:hmax,lat,:]
        cols = 'Reds'  
        levels = np.linspace(200,240,40)
        cunit = 'K'   
    elif key=='eddy temp':
        air_temp = plobject.data['temp'][time_slice,hmin:hmax,lat,:]
        zm_temp = np.mean(air_temp, axis=-1)
        cube = air_temp - zm_temp[:,np.newaxis]
        cols = 'Reds'
        levels = np.linspace(-8,8,8)
        cunit = 'K'
    elif key=='div':       
        cube = plobject.data['div'][time_slice,hmin:hmax,lat,:]*1e6
        cols = 'seismic'
        levels = np.linspace(-1000,1000,20)
        cunit = '$10^{-6}$ s-1'
    elif key=='eddy zeta':
        rel_vort = plobject.data['zeta'][time_slice,hmin:hmax,lat,:]
        zm_zeta = np.mean(rel_vort, axis=-1)
        cube = (rel_vort - zm_zeta[:,np.newaxis])*1e6
        cols = 'coolwarm'
        levels = np.linspace(-240,240,20)
        cunit = '$10^{-6}$ s-1'
    elif key=='eddy wind':
        zonal_wind = plobject.data['vitu'][time_slice,hmin:hmax,lat,:]
        zm_wind = np.mean(zonal_wind, axis=-1)
        cube = zonal_wind - zm_wind[:,np.newaxis]
        cols = 'coolwarm'
        levels = np.linspace(-80,80,20)
        cunit = 'm/s'
    else:
        print('Key argument is not valid. Possible keys \
              are temp, eddy temp, div, eddy zeta.')
        
    fig, ax = plt.subplots(figsize=(6,6))
    plt.contourf(plobject.lons, plobject.heights[hmin:hmax], 
                 cube, 
    #             levels=levels,
                 cmap=cols)
    plt.title(f'{key}, lat={plobject.lats[lat]}')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Pressure [mbar]')
#    ax.set_yscale('log')
#    plt.gca().invert_yaxis()
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def hovmoeller(plobject, key='eddy temp', trange=(400,500),
               hmin=28, hmax=45, lat=88, lon=90,
               save=False, savename='hovmoeller.png', saveformat='png'):

    """ Hovmoeller plot of the input key field variable for
    the time and height ranges specified."""

    if key=='temp':
        cube = plobject.data['temp'][trange[0]:trange[1],hmin:hmax,lat,lon]
        cols = 'Reds'  
        levels = np.linspace(235,245,10)
        cunit = 'K'   
    elif key=='eddy temp':
        air_temp = plobject.data['temp'][trange[0]:trange[1],hmin:hmax,lat,:]
        zm_temp = np.mean(air_temp, axis=-1)
        cube = air_temp - zm_temp[:,:,np.newaxis]
        cube = cube[:,:,lon]
        cols = 'coolwarm'
        levels = np.linspace(-8,8,8)
        cunit = 'K'
    elif key=='temp anomaly':
        air_temp = plobject.data['temp'][trange[0]:trange[1],hmin:hmax,lat,:]
        zm_temp = np.mean(air_temp, axis=0)
        cube = air_temp - zm_temp[np.newaxis,:,:]
        cube = cube[:,:,lon]
        cols = 'coolwarm'
        levels = np.linspace(-8,8,10)
        cunit = 'K'
    elif key=='div':       
        cube = plobject.data['div'][trange[0]:trange[1],hmin:hmax,lat,lon]*1e6
        cols = 'seismic'
        levels = np.linspace(-1000,1000,20)
        cunit = '$10^{-6}$ s-1'
    elif key=='eddy zeta':
        rel_vort = plobject.data['zeta'][trange[0]:trange[1],hmin:hmax,lat,:]
        zm_zeta = np.mean(rel_vort, axis=-1)
        cube = (rel_vort - zm_zeta[:,:,np.newaxis])*1e6
        cube = cube[:,:,lon]
        cols = 'coolwarm'
        levels = np.linspace(-240,240,20)
        cunit = '$10^{-6}$ s-1'
    elif key=='eddy wind':
        zonal_wind = plobject.data['vitu'][trange[0]:trange[1],hmin:hmax,lat,:]
        zm_wind = np.mean(zonal_wind, axis=-1)
        cube = zonal_wind - zm_wind[:,:,np.newaxis]
        cube = cube[:,:,lon]
        cols = 'coolwarm'
        levels = np.linspace(-80,80,20)
        cunit = 'm/s'
    else:
        print('Key argument is not valid. Possible keys \
              are temp, eddy temp, div, eddy zeta.')
        
    time_axis = np.arange(0,len(plobject.data['time_counter'][trange[0]:trange[1]]))   
    fig, ax = plt.subplots(figsize=(6,6))
    plt.contourf(time_axis, plobject.heights[hmin:hmax], 
                 cube.T, 
 #                levels=levels,
                 norm=TwoSlopeNorm(0),
                 cmap=cols)
    plt.title(f'{key}, lat={plobject.lats[lat]}, lon={plobject.lons[lon]}')
    plt.xlabel('Time')
    plt.ylabel('Height [km]')
#    ax.set_yscale('log')
#    plt.gca().invert_yaxis()
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



# %%
