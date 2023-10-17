# %%
import iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def zmzw(cube, meaning=True, time_slice=-1, save=False, saveformat='png',
        savename='zmzw.png'):

    """ Input: iris cube for zonal wind 
        Output: plot of zonal mean zonal wind
        
        meaning (default True) calculates the time mean
        time_slice (default -1) selects time if meaning=False """

    zonal = cube.copy()
    if meaning==True:
        zonal = zonal.collapsed('time', iris.analysis.MEAN)
    else:
        zonal = zonal[time_slice,:,:,:]
    zmean = zonal.collapsed('longitude', iris.analysis.MEAN)
    
    plt.contourf(zmean.data, cmap='RdBu_r', norm=TwoSlopeNorm(0))
    plt.title('Zonal mean zonal wind')
    plt.xlabel('Latitude [deg]')
    plt.ylabel('Pressure [mbar]')

    plt.xticks((0,8,16,24,32), ('90S', '45S', '0', '45N', '90N'))
    plt.yticks((10,20,30,40,50), 
            np.round(zmean.coord('model_level_number').points[::10]*1e-5, 0))
    #plt.gca().invert_yaxis()
    plt.colorbar()
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

#  %%
def zmzw_snaps(cube, time_range=(0,2), save=False, saveformat='png'):

    """ Input: iris cube for zonal wind 
        Output: plot of zonal mean zonal wind
        
        meaning (default True) calculates the time mean
        time_slice (default -1) selects time if meaning=False """

    zonal = cube.copy()
    zmean = zonal.collapsed('longitude', iris.analysis.MEAN)
    zmean = zmean.data

    for time_slice in range(time_range[0],time_range[1]):
        print(time_slice)
        savename = 'zmzw_' + str(time_slice) + '.' + saveformat
    
        plt.contourf(zmean[time_slice,:,:], cmap='RdBu_r', norm=TwoSlopeNorm(0))
        plt.title('Zonal mean zonal wind')
        plt.xlabel('Latitude')
        plt.ylabel('Model level')
        if save==True:
            plt.savefig(savename, format=saveformat, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# %%
def u_series(cube, time_range=(0,-1), meaning=True, lat=16, lon=24, lev=40,
             save=False, savename='u_series.png', saveformat='png'):

    u_wind = cube.copy()
    if meaning==True:
        u_wind = u_wind[time_range[0]:time_range[-1],lev,lat,:].collapsed('longitude', iris.analysis.MEAN)
        titleterm = f'Zonal mean zonal wind at level {lev}, lat {lat}'
    else:
        u_wind = u_wind[time_range[0]:time_range[-1],lev,lat,lon]
        titleterm = f'Zonal wind at level {lev}, lat {lat}, lon {lon}'
    
    u_data = u_wind.data

    plt.plot(u_data)
    plt.title(f'{titleterm}')
    plt.ylabel('Wind speed [m/s]')
    plt.xlabel('Time [days?]')
    plt.show()