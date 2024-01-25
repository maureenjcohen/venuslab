# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# %%
def ageline(plobject, coords=(-0., -0.), sourcelev=22, line_lev=30, trange=(4000,4499),
            convert2yr=True,
            save=False, saveformat='png', savename='age_line.png'):

    line_lat = np.where(plobject.lats==coords[0])[0][0]
    line_lon = np.where(plobject.lons==coords[1])[0][0]
    print(line_lat, line_lon)
    ageo = plobject.data['age'][trange[0]:trange[1],:,line_lat,line_lon]

    if convert2yr==True:
        ageo = ageo/(60*60*24*360)
        cunit = 'years'
    else:
        cunit = 'seconds' 

    fig, ax = plt.subplots(figsize=(6,6))
    plt.plot(ageo[:,sourcelev], color='b', label=f'h={plobject.heights[sourcelev]} km')
    plt.plot(ageo[:,line_lev], color='r', label=f'h={plobject.heights[line_lev]} km')
    plt.title(f'Age of air at h={plobject.heights[line_lev]} km, \
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
def zmage(plobject, hmin=0, hmax=20, time_slice=-1, convert2yr=True,
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
    
    plt.contourf(plobject.lats, plobject.heights[hmin:hmax], 
                 zmslice, 
                 levels=levels,
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
            save=False, saveformat='png', savename='age_map.png'):
        
    """ Input: numpy array for age of air 
        Output: lon-lat plot of age of air at given model level
        
        time_slice (default -1) selects time """

    ageo = plobject.data['age']
    ageo = ageo[time_slice,lev,:,:]

    if convert2yr==True:
        ageo = ageo/(60*60*24*360)
        cunit = 'years'
    else:
        cunit = 'seconds' 

    levels = np.linspace(np.min(ageo),np.max(ageo),40)

    plt.contourf(plobject.lons, plobject.lats, 
                 ageo, levels=levels, cmap='cividis')
    plt.title(f'Age of air, h={plobject.heights[lev]} km')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()