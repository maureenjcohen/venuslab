# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# %%
def zmzw(plobject, meaning=True, time_slice=-1, 
         save=False, saveformat='png', savename='zmzw.png'):

    """ Input: numpy array for zonal wind 
        Output: plot of zonal mean zonal wind
        
        meaning (default True) calculates the time mean
        time_slice (default -1) selects time if meaning=False """

    zonal = -np.copy(plobject.data['vitu'])
    if meaning==True:
        zonal = np.mean(zonal, axis=0)
    else:
        zonal = zonal[time_slice,:,:,:]
    zmean = np.mean(zonal, axis=2) 
    
    plt.contourf(plobject.lats, plobject.heights, zmean, 
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

#  %%
def zmzw_snaps(plobject, time_range=(0,2), 
               save=False, saveformat='png'):

    """ Input: numpy array for zonal wind 
        Output: plot of zonal mean zonal wind
        
        meaning (default True) calculates the time mean
        time_slice (default -1) selects time if meaning=False """

    zonal = -np.copy(plobject.data['vitu'])
    zmean = np.mean(zonal, axis=-1)

    for time_slice in range(time_range[0],time_range[1]):
        print(time_slice)
        savename = 'zmzw_' + str(time_slice) + '.' + saveformat
    
        plt.contourf(plobject.lats, plobject.heights, zmean[time_slice,:,:], 
                     levels=np.arange(-100, 101, 20), cmap='RdBu', 
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

    u_wind = -np.copy(plobject.data['vitu'])
    if meaning==True:
        u_wind = np.mean(u_wind[time_range[0]:time_range[-1],lev,lat,:], axis=-1)
        titleterm = f'Zonal mean zonal wind at h={int(plobject.heights[lev])} km, ' \
                    f'lat={int(plobject.lats[lat])}'
    else:
        u_wind = u_wind[time_range[0]:time_range[-1],lev,lat,lon]
        titleterm = f'Zonal wind at h={int(plobject.heights[lev])} km, ' \
                    f'lat={int(plobject.lats[lat])}, ' \
                    f'lon={int(plobject.lons[lon])}'

    plt.plot(u_wind)
    plt.title(f'{titleterm}')
    plt.ylabel('Wind speed [m/s]')
    plt.xlabel('Time [days?]')
    plt.show()

# %%
def wind_vectors(plobject, meaning=True, time_slice=-1, n=2, 
                 qscale=2, level=40):

    u = -np.copy(plobject.data['vitu'])
    v = np.copy(plobject.data['vitv'])
    w = np.copy(plobject.data['vitw'])

    if meaning==True:
        u = np.mean(u, axis=0)
        v = np.mean(v, axis=0)
        w = np.mean(w, axis=0)
    else:
        u = u[time_slice,:,:,:]
        v = v[time_slice,:,:,:]
        w = w[time_slice,:,:,:]

 #   X, Y = np.meshgrid(np.arange(0,len(plobject.lons)), np.arange(0,len(plobject.lats)))
    X, Y = np.meshgrid(plobject.lons, plobject.lats)
    fig, ax = plt.subplots(figsize=(8,5))
    wplot = ax.contourf(plobject.lons, plobject.lats, w[level,:,:], 
                        cmap='coolwarm', norm=TwoSlopeNorm(0))
    cbar = plt.colorbar(wplot, orientation='vertical', fraction=0.05)
    cbar.set_label('Vertical wind, Pa/s', loc='center')
    q1 = ax.quiver(X[::n, ::n], Y[::n, ::n], u[level, ::n, ::n],
                   -v[level, ::n, ::n], angles='xy', scale_units='xy', scale=qscale)
    ax.quiverkey(q1, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Winds of Venus, h={int(plobject.heights[level])} km')
    plt.show()

