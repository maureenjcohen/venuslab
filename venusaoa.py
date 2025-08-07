# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# %%
def smoothed_age(plobject, coords=(-0., -0.), linelev=30, 
                 trange=(0,1876), plot=True):
    
    """ Plot age of air time series and smoothed time series on """
    
    line_lat = np.where(plobject.lats==coords[0])[0][0]
    line_lon = np.where(plobject.lons==coords[1])[0][0]
    print(line_lat, line_lon)
    ageo = plobject.data['age'][trange[0]:trange[1],linelev,line_lat,line_lon]
    ageo = ageo/(60*60*24*360)
    filtered = savgol_filter(ageo, 500, 1) # Savitzy-Golay filter
    filtered_grad = savgol_filter(np.gradient(filtered), 500, 1) # Do it again to the gradient
    filtered_second = savgol_filter(np.gradient(filtered_grad), 500, 1)
    time_axis = np.arange(0, ageo.shape[0])*plobject.tinterval/(60*60*24*360)

    if plot==True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
        fig.suptitle(f'Age of air convergence at {plobject.heights[linelev]} km',
                     fontsize=14)

        ax1.plot(time_axis, ageo, color='b', label='Age of air')
        ax1.plot(time_axis, filtered, color='r', label='Smoothed')
        ax1.set_title('Age of air')
        ax1.legend()
        ax1.set_xlabel('Simulation time / Earth years')
        ax1.set_ylabel('Age of air / Earth years')

        ax2.plot(time_axis, filtered_grad, color='g')
        ax2.set_title('Gradient of age of air')
        ax2.set_xlabel('Simulation time / Earth years')
        ax2.set_ylabel('Change per year')

        ax3.plot(time_axis, filtered_second, color='m')
        ax3.set_title('Second gradient of age of air')
        ax3.set_xlabel('Simulation time / Earth years')
        ax3.set_ylabel('Change in gradient per year')

        plt.show()

    return filtered, filtered_grad   

# %%
def ageline_fit(x, a, tau):
    return a*(1 - (np.exp(-x/tau)))

# %%
def plot_fits(plobject, coords=(-0., -0.), linelev=30, 
              trange=(0,1876)):
    
    smoothed, smoothed_grad = smoothed_age(plobject=plobject, coords=coords,
                                           linelev=linelev, trange=trange,
                                           plot=False)
    time_axis = np.arange(0, smoothed.shape[0])*(plobject.tinterval/(60*60*24*360))
    popt, pcov = curve_fit(ageline_fit, time_axis, smoothed)

    plt.plot(time_axis, smoothed, color='b', label='Smoothed data')
    plt.plot(time_axis, ageline_fit(time_axis, *popt), color='r',
             label='Fit: a=%5.3f, b=%5.3f' % tuple(popt))
    plt.legend()
    plt.xlabel('Simulation time / Earth years')
    plt.ylabel('Age of air / Earth years')
    plt.title('%s km data fit to $a(1 - e^{-bt})$' % plobject.heights[linelev])
    plt.show()

# %%
def ageline(plobject, coords=(-0., -0.), sourcelev=22, linelev=30, trange=(4000,4499),
            convert2yr=False, fractional=False,
            save=False, saveformat='png', savename='age_line.png'):

    line_lat = np.where(plobject.lats==coords[0])[0][0]
    line_lon = np.where(plobject.lons==coords[1])[0][0]
    print(line_lat, line_lon)
    ageo = plobject.data['age'][trange[0]:trange[1],:,line_lat,line_lon]

    if convert2yr==True:
        ageo = ageo/(60*60*24*360)
        cunit = 'years'
        interval = plobject.tinterval/(60*60*24*360)
    else:
        cunit = 'seconds'
        interval = plobject.tinterval

    if fractional==True:
        sim_times = np.arange(trange[0],trange[1])*interval
        ageo = ageo/sim_times[:,np.newaxis]

    fig, ax = plt.subplots(figsize=(6,6))
    plt.plot(ageo[:,sourcelev], color='b', label=f'h={plobject.heights[sourcelev]} km')
    plt.plot(ageo[:,linelev], color='r', label=f'h={plobject.heights[linelev]} km')
    plt.title(f'Age of air at h={plobject.heights[linelev]} km, \
              lat={plobject.lats[line_lat]}, \
              lon={plobject.lons[line_lon]}')
    plt.xlabel('Simulation time')
    plt.ylabel(f'Age [{cunit}]')
    plt.legend(loc='best')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return ageo

# %%
def tracerline(plobject, coords=(-0., -0.), sourcelev=22, line_lev=30, trange=(4000,4499),
            save=False, saveformat='png', savename='age_line.png'):

    line_lat = np.where(plobject.lats==coords[0])[0][0]
    line_lon = np.where(plobject.lons==coords[1])[0][0]
    print(line_lat, line_lon)
    ageo = plobject.data['aoa'][trange[0]:trange[1],:,line_lat,line_lon]

    fig, ax = plt.subplots(figsize=(6,6))
    plt.plot(ageo[:,sourcelev], color='b', label=f'h={plobject.heights[sourcelev]} km')
    plt.plot(ageo[:,line_lev], color='r', label=f'h={plobject.heights[line_lev]} km')
    plt.title(f'Tracer concentration at h={plobject.heights[line_lev]} km, \
              lat={plobject.lats[line_lat]}, \
              lon={plobject.lons[line_lon]}')
    plt.xlabel('Simulation time')
    plt.ylabel('Mixing ratio [kg/kg]')
    plt.legend(loc='best')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# %%
def zmage(plobject, hmin=0, hmax=20, time_slice=-2, convert2yr=True,
          cmin=11.64, cmax=11.91,
         save=False, saveformat='png', savename='zmage.png'):

    """ Input: numpy array for age of air 
        Output: plot of zonal mean age of air
        
        time_slice (default -1) selects time """

    ageo = plobject.data['age']
    ageo = ageo[time_slice,:,:,:]
    zmageo = np.mean(ageo, axis=-1)
    levels = np.linspace(cmin, cmax, 40)

    if convert2yr==True:
        zmageo = zmageo/(60*60*24*360)
        cunit = 'years'
    else:
        cunit = 'seconds' 
 
    zmslice = zmageo[hmin:hmax,:]
 #   levels = np.linspace(np.min(zmslice),np.max(zmslice),40)
    
    fig = plt.figure(figsize=(6, 6))
    plt.contourf(plobject.lats, plobject.heights[hmin:hmax], 
                 zmslice, 
#                 levels=levels,
                 cmap='cividis')
    plt.title('Age of air (zonal mean)')
    plt.xlabel('Latitude [deg]')
    plt.ylabel('Height [km]')
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def age_map(plobject, lev=16, time_slice=-1, convert2yr=True,
            save=False, savepath='/exomars/projects/mc5526/VPCM_volcanic_plumes/scratch_plots/',
            saveformat='png', savename='age_map.png'):
        
    """ Input: numpy array for age of air 
        Output: lon-lat plot of age of air at given model level
        
        time_slice (default -1) selects time """

    ageo = plobject.data['age']
    age = ageo[time_slice,lev,:,:]

    if convert2yr==True:
        ageo = ageo/(60*60*24*360)
        age = age/(60*60*24*360)
        cunit = 'years'
    else:
        cunit = 'seconds' 

    #levels = np.linspace(ageo.quantile(0.01),ageo.quantile(0.99),40)

    plt.contourf(plobject.lons, plobject.lats, 
                 age, vmin=ageo.quantile(0.01),
                 vmax=ageo.quantile(0.99), cmap='cividis')
    plt.title(f'Age of air, h={plobject.heights[lev]} km')
    plt.xlabel('Longitude /deg')
    plt.ylabel('Latitude /deg')
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')
    if save==True:
        plt.savefig(savepath+savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
# %%
