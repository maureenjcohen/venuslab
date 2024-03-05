# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# %%
def sw_flux(plobject, meaning=True, time_slice=-1, lev=-1):

    """ 2D plot of shortwave radiation at top of atmosphere."""
    sw = np.copy(plobject.data['tops'])
    if meaning==True:
        sw = np.mean(sw, axis=0)
        titleterm = 'long-term mean'
    else:
        sw = sw[time_slice,:,:]
        titleterm = f'at {time_slice}'

    fig, ax = plt.subplots(figsize=(8,4))
    plt.contourf(plobject.lons, plobject.lats, sw, cmap='hot')
    plt.title(f'Solar rad. at TOA, {titleterm}')
    plt.ylabel('Latitude [deg]')
    plt.xlabel('Longitude [deg]')
    cbar = plt.colorbar()
    cbar.set_label('W/m$^2$', loc='center')
    plt.show()

# %%
def surface_temp(plobject, meaning=True, time_slice=-1):

    """2D plot of surface temperature. """
    st = np.copy(plobject.data['tsol'])
    if meaning==True:
        st = np.mean(st, axis=0)
        titleterm = 'long-term mean'
    else:
        st = st[time_slice,:,:]
        titleterm = f'at {time_slice}'

    fig, ax = plt.subplots(figsize=(8,4))
    plt.contourf(plobject.lons, plobject.lats, st, cmap='hot')
    plt.title(f'Surface temperature, {titleterm}')
    plt.ylabel('Latitude [deg]')
    plt.xlabel('Longitude [deg]')
    cbar = plt.colorbar()
    cbar.set_label('K', loc='center')
    plt.show()
    
# %%
def heating_rates(plobject, tmean=True, time_slice=-1,
                  select='all', smean=True, coord=(0,0)):
    
    """ Vertical profiles of heating rates"""
    if tmean==True:
        dyn_dt = np.mean(plobject.data['dtdyn'][:], axis=0)
        pbl_dt = np.mean(plobject.data['dtvdf'][:], axis=0)
        dry_dt = np.mean(plobject.data['dtajs'][:], axis=0)
        sw_dt = np.mean(plobject.data['dtswr'][:], axis=0)
        lw_dt = np.mean(plobject.data['dtlwr'][:], axis=0)
        total = dyn_dt + pbl_dt + dry_dt + sw_dt + \
            lw_dt
    else:
        dyn_dt = plobject.data['dtdyn'][:][time_slice,:,:,:]
        pbl_dt = plobject.data['dtvdf'][:][time_slice,:,:,:]
        dry_dt = plobject.data['dtajs'][:][time_slice,:,:,:]
        sw_dt = plobject.data['dtswr'][:][time_slice,:,:,:]
        lw_dt = plobject.data['dtlwr'][:][time_slice,:,:,:]
        total = dyn_dt + pbl_dt + dry_dt + sw_dt + \
            lw_dt

    if smean==True:
#        dyn_dt = np.sum(dyn_dt, axis=(1,2))*(plobject.data['aire'][:,None])/plobject.area
        total = np.sum(total*plobject.data['aire'], axis=(1,2))/plobject.area
    print(total)
    fig, ax = plt.subplots(figsize=(4,6))
    plt.plot(total, plobject.heights)
    plt.title('Total heating rate')
    plt.ylabel('Height [km]')
    plt.xlabel('Heating rate [K/s]')
    plt.show()

# %%
def static_stability(plobject, coords=[(0,48), (48,48), (95,48)],
                     hmin=0, hmax=-10, time_mean=True, time_slice=-1):
    
    """ Plot vertical profile of static stability"""
    if not hasattr(plobject, 'rho'):
        plobject.calc_rho()
    # Check if Planet object already has potential
    # temperature cube
    
    if time_mean==True:
        temp = np.mean(plobject.data['temp'][:], axis=0)
    else:
        temp = plobject.data['temp'][time_slice,:,:,:]
    # Get rid of time axis through meaning or selection

    ## First get global area-weighted mean profile
    global_prof = np.sum(temp*plobject.data['aire'][:], axis=(1,2))/plobject.area
    # Calculate area weighted means on each model level
    dtdz = np.gradient(global_prof)/np.gradient(np.array(plobject.heights))
    # Temperature lapse rate, dT/dz
    global_stab = (dtdz - (-8.87))
    # Stability parameter: temperature lapse rate minus 
    # dry adiabatic lapse rate for Venus

    ## Now get profiles at individual gridboxes
    gridbox_dtdz = np.gradient(temp)/np.gradient(np.array(plobject.heights))
    gridbox_stab = (gridbox_dtdz - (-8.87))

    fig, ax = plt.subplots(figsize=(6,4))
    for coord in coords:
        lat_lab, lon_lab = plobject.lats[coord[0]], plobject.lons[coord[1]]
        plt.plot(gridbox_stab[hmin:hmax,coord[0],coord[1]], plobject.heights[hmin:hmax], 
                 label=f'{lat_lab}$^\circ$ lat, {lon_lab}$^\circ$ lon')
    plt.plot(global_stab[hmin:hmax], plobject.heights[hmin:hmax], color='k')
    plt.plot(np.zeros(40), plobject.heights[hmin:hmax], color='k', linestyle='--')
    plt.title('Static stability of the atmosphere')
    plt.ylabel('Height [km]')
    plt.xlabel('Static stability [K/km]')
#    plt.xlim((-2,10))
    plt.show()
# %%
