# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.integrate import cumtrapz

# %%
def zmzw(plobject, meaning=True, time_slice=-1, 
         save=False, saveformat='png', savename='zmzw.png'):

    """ Input: numpy array for zonal wind 
        Output: plot of zonal mean zonal wind
        
        meaning (default True) calculates the time mean
        time_slice (default -1) selects time if meaning=False """

    zonal = plobject.data['vitu']
    if meaning==True:
        zonal = np.mean(zonal, axis=0)
    else:
        zonal = zonal[time_slice,:,:,:]
    zmean = np.mean(zonal, axis=-1) 
    
    plt.contourf(plobject.lats, plobject.heights, -zmean, 
                 cmap='RdBu', norm=TwoSlopeNorm(0))
    plt.title('Zonal mean zonal wind')
    plt.xlabel('Latitude [deg]')
    plt.ylabel('Height [km]')
    cbar = plt.colorbar()
    cbar.ax.set_title('m/s')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def zmmw(plobject, meaning=True, time_slice=-1, 
         save=False, saveformat='png', savename='zmzw.png'):

    """ Input: numpy array for meridional wind 
        Output: plot of zonal mean meridional wind
        
        meaning (default True) calculates the time mean
        time_slice (default -1) selects time if meaning=False """

    zonal = plobject.data['vitv']
    if meaning==True:
        zonal = np.mean(zonal, axis=0)
    else:
        zonal = zonal[time_slice,:,:,:]
    zmean = np.mean(zonal, axis=-1) 
    
    plt.contourf(plobject.lats, plobject.heights[:-5], zmean[:-5,:], 
                 cmap='RdBu_r', levels=np.arange(-10,10,1), norm=TwoSlopeNorm(0))
    plt.title('Zonal mean meridional wind')
    plt.xlabel('Latitude [deg]')
    plt.ylabel('Height [km]')
    cbar = plt.colorbar()
    cbar.ax.set_title('m/s')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

#  %%
def zmzw_snaps(plobject, time_range=(0,100,10), 
               save=False, saveformat='png'):

    """ Input: numpy array for zonal wind 
        Output: plot of zonal mean zonal wind
        
        meaning (default True) calculates the time mean
        time_slice (default -1) selects time if meaning=False """

    zonal = plobject.data['vitu']
    zmean = np.mean(zonal, axis=-1)

    for time_slice in range(time_range[0],time_range[1], time_range[2]):
        print(time_slice)
        savename = 'zmzw_' + str(time_slice) + '.' + saveformat
    
        plt.contourf(plobject.lats, plobject.heights, -zmean[time_slice,:,:], 
                     levels=np.arange(-160, 40, 20), cmap='RdBu', 
                     norm=TwoSlopeNorm(0))
        plt.title('Zonal mean zonal wind')
        plt.xlabel('Latitude')
        plt.ylabel('Height [km]')
        cbar = plt.colorbar()
        cbar.ax.set_title('m/s')

        if save==True:
            plt.savefig(savename, format=saveformat, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# %%
def u_series(plobject, time_range=(0,-1), meaning=True, lat=16, lon=24, lev=40,
             save=False, savename='u_series.png', saveformat='png'):

    u_wind = plobject.data['vitu']
    if meaning==True:
        u_wind = np.mean(u_wind[time_range[0]:time_range[-1],lev,lat,:], axis=-1)
        titleterm = f'Zonal mean zonal wind at h={int(plobject.heights[lev])} km, ' \
                    f'lat={int(plobject.lats[lat])}'
    else:
        u_wind = u_wind[time_range[0]:time_range[-1],lev,lat,lon]
        titleterm = f'Zonal wind at h={int(plobject.heights[lev])} km, ' \
                    f'lat={int(plobject.lats[lat])}, ' \
                    f'lon={int(plobject.lons[lon])}'

    plt.plot(-u_wind)
    plt.title(f'{titleterm}')
    plt.ylabel('Wind speed [m/s]')
    plt.xlabel('Time [days?]')
    plt.show()

# %%
def wind_vectors(plobject, meaning=True, time_slice=-1, n=2, 
                 qscale=2, level=40, wtype='Vertical'):
    
    """ Plot the horizontal and vertical wind on a model level in one figure."""

    u = plobject.data['vitu'][:,level,:,:]
    v = plobject.data['vitv'][:,level,:,:]

    if wtype=='Pressure':
        w = plobject.data['vitw'][:,level,:,:]
        unit = 'Pa/s'

    elif wtype=='Vertical':
        omega = plobject.data['vitw'][:,level,:,:]
        temp = plobject.data['temp'][:,level,:,:]
        pres = plobject.data['pres'][:,level,:,:]
        w = -(omega*temp*plobject.RCO2)/(pres*plobject.g)
        unit = 'm/s'
    else:
        print('Arg wtype must be either Pressure or Vertical')

    if meaning==True:
        u = np.mean(u, axis=0)
        v = np.mean(v, axis=0)
        w = np.mean(w, axis=0)
    else:
        u = u[time_slice,:,:]
        v = v[time_slice,:,:]
        w = w[time_slice,:,:]
    print(w[15,7], u[15,7])
    print(v[15,7])

#    X, Y = np.meshgrid(np.arange(0,len(plobject.lons)), np.arange(0,len(plobject.lats)))
    X, Y = np.meshgrid(plobject.lons, plobject.lats)
    fig, ax = plt.subplots(figsize=(8,5))
    wplot = ax.contourf(plobject.lons, plobject.lats, w, 
                      cmap='coolwarm', norm=TwoSlopeNorm(0))
    cbar = plt.colorbar(wplot, orientation='vertical', fraction=0.05)
    cbar.set_label(f'Vertical wind, {unit}', loc='center')
    q1 = ax.quiver(X[::n, ::n], Y[::n, ::n], -u[::n, ::n],
                   v[::n, ::n], angles='xy', scale_units='xy', scale=qscale)
    ax.quiverkey(q1, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Winds of Venus, h={int(plobject.heights[level])} km')
    plt.show()

# %%
def psi_m(plobject, meaning=True, trange=(-230,-1), time_slice=-1):

    """ Plot the mean meridional mass streamfunction. """

    v = plobject.data['vitv']
    pres = plobject.data['pres']
    if meaning==True:
        v = np.mean(v[trange[0]:trange[1],:,:,:], axis=0)
        pres = np.mean(pres[trange[0]:trange[1],:,:,:], axis=0)
    else:
        v = v[time_slice,:,:,:]
        pres = pres[time_slice,:,:,:]
    zm_v = np.mean(v, axis=-1)
    # That's the zonal mean of the northward wind calculated
    zm_pres = np.mean(pres, axis=-1)
    # Zonal mean of pressure

    dp = np.gradient(zm_pres, axis=0)
    integrand = np.flip(zm_v, axis=0)*np.flip(dp, axis=0) # dp x v multiplied from top to bottom
#    stf = -np.cumsum(integrand, axis=0) # cumulative sum from top to bottom
    stf = -cumtrapz(integrand, axis=0)
    stf_constant = (2*np.pi*plobject.radius*1e3)*(np.cos(plobject.lats*(np.pi/180)))/(plobject.g)
    stf = stf_constant*np.flip(stf, axis=0)*1e-10

    fig, ax = plt.subplots(figsize=(6,6))
    cs = plt.contour(plobject.lats, plobject.heights[:-1], stf, colors='black',
                     levels=40)
    ax.clabel(cs, cs.levels, inline=True)
    plt.title('Mean meridional mass streamfunction, $10^{10}$ kg/s')
    plt.xlabel('Latitude [deg]')
    plt.ylabel('Height [km]')
    plt.show()

# %%
def wmap(plobject, meaning=True, lev=30, time_slice=-1, wtype='Vertical'):
    """ Plot lon-lat map of vertical or pressure velocity
    
    if wtype Pressure, plot pressure velocity in Pa/s (default)
    if wtype Vertical, plot vertical velocity in m/s"""

    if wtype=='Pressure':
        w = plobject.data['vitw'][:,lev,:,:]
        unit = 'Pa/s'
    elif wtype=='Vertical':
        omega = plobject.data['vitw'][:,lev,:,:]
        temp = plobject.data['temp'][:,lev,:,:]
        pres = plobject.data['pres'][:,lev,:,:]
        w = -(omega*temp*plobject.RCO2)/(pres*plobject.g)
        unit = 'm/s'
    else:
        print('Arg wtype must be either Pressure or Vertical')

    if meaning==True:
        w = np.mean(w, axis=0)
        titleterm = 'long-term mean'
    else:
        w = w[time_slice,:,:]
        titleterm = f't={time_slice}'

    fig, ax = plt.subplots(figsize=(8,6))
    wm = plt.contourf(plobject.lons, plobject.lats, w,
#                      levels=np.arange(-0.04,0.041,0.005), 
                      cmap='coolwarm', norm=TwoSlopeNorm(0))
    plt.title(f'{wtype} velocity, h={plobject.heights[lev]} km, {titleterm}')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    cbar = plt.colorbar()
    cbar.set_label(f'{unit}')
    plt.show()


# %%
def lonlatslice(plobject, cubename, lev, time_slice=-1,
                save=False, saveformat='png', savename='lonlatslice.png'):

    data = plobject.data[cubename][time_slice,lev,:,:]

    cubedict = {'duvdf': {'ptitle':'Boundary layer du', 'punit':'m/s2'},
                'duajs': {'ptitle':'Dry convection du', 'punit': 'm/s2'},
                'dudyn': {'ptitle':'Dynamics du', 'punit': 'm/s2'},
                'eta': {'ptitle':'Absolute vorticity', 'punit':'s$^{-1}$'},
                'zeta': {'ptitle':'Relative vorticity', 'punit':'s$^{-1}$'}}  

    fig, ax = plt.subplots(figsize=(8,6))
    CS = plt.contourf(plobject.lons, plobject.lats, data,
                      cmap='coolwarm', norm=TwoSlopeNorm(0))
    plt.title(cubedict[cubename]['ptitle'] + f', h={plobject.heights[lev]} km')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')

    cbar = plt.colorbar(CS, orientation='vertical', fraction=0.05)
    cbar.set_label(cubedict[cubename]['punit'], loc='center')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# %%
def vprofile(plobject, key, coords, ptitle, xlab, unit,
             hmin, hmax,
             zmean=False, meaning=False, time_slice=-1, 
             convert2yr=True,
             save=False, saveformat='png', savename='lonlatslice.png'):

    """ Plot vertical profiles at selected locations
        Can specify time mean and/or zonal mean
        Coords must be a list of latxlon tuples
        Assumes a 4-D cube                      """

    if meaning==True:
        cube = np.mean(plobject.data[key], axis=0)
    else:
        cube = plobject.data[key][time_slice,:,:,:]

    if zmean==True:
        cube = np.mean(cube, axis=-1)

    if key=='age' and convert2yr==True:
        cube = cube/(60*60*24*360)
        unit = 'years'
    elif key=='age' and convert2yr==False:
        unit = 'seconds' 

    fig, ax = plt.subplots(figsize=(6,8))

    for coord in coords:
        lat_lab, lon_lab = plobject.lats[coord[0]], plobject.lons[coord[1]]
        plt.plot(cube[hmin:hmax,coord[0],coord[1]], plobject.heights[hmin:hmax], 
                 label=f'{lat_lab}$^\circ$ lat, {lon_lab}$^\circ$ lon')
    plt.title(f'Vertical profile of {ptitle}')
    plt.xlabel(f'{xlab} [{unit}]')
    plt.ylabel('Height [km]')
    plt.xlim((0,15))
    plt.legend()
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
