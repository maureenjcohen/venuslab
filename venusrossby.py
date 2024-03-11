# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def omega_profile(plobject, hrange=(0,-1), trange=(0,-1),
                  plot=False, save=False, 
                  saveformat='png', savename='uprofile.png'):

    """ Global area-weighted profile of rotation rate of
    the atmosphere (effective omega)        """

    zwind = np.mean(plobject.data['vitu'][trange[0]:trange[1],hrange[0]:hrange[1],:,:],0)
    grid_areas = plobject.data['aire'][:]
    zmean = np.sum(zwind*grid_areas[np.newaxis,:,:], axis=(1,2))/(plobject.area)
    # Area-weighted spatial and time mean of zonal wind

    circumf = (2*np.pi*(plobject.radius + plobject.heights[hrange[0]:hrange[1]])*1000)
    period = (circumf/np.abs(zmean))
    omega = (2*np.pi)/period

    if plot==True:
        fig, ax = plt.subplots(figsize=(6,8))
        plt.plot(omega,plobject.heights[hrange[0]:hrange[1]])
        plt.title('Rotation rate of atmosphere')
        plt.ylabel('Height [km]')
        plt.xlabel('Rotation rate [rad/s]')
        plt.xscale('log')
        if save==True:
            plt.savefig(savename, format=saveformat, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return zmean, omega

# %%
def bv_freq(plobject, hrange=(0,-1), trange=(0,-1),
            plot=False, save=False, 
            saveformat='png', savename='bvprofile.png'):
    
    """ Global-area weighted profile of Brunt-Vaisala frequency"""

    if not hasattr(plobject, 'theta'):
        plobject.calc_theta() # Calculate potential temperature
    # Method uses formula from LMDZ Venus model, not standard one

    th_mean = np.mean(plobject.theta[trange[0]:trange[1],hrange[0]:hrange[1],:,:], axis=0)
    grid_areas = plobject.data['aire'][:]
    th_prof = np.sum(th_mean*grid_areas[np.newaxis,:,:], axis=(1,2))/(plobject.area)
    th_dz = np.gradient(th_prof)/np.gradient(plobject.heights[hrange[0]:hrange[1]]*1000)
    root_term = plobject.g*th_dz/th_prof
    freq = np.sqrt(root_term) 

    if plot==True:
        fig, ax = plt.subplots(figsize=(6,8))
        plt.plot(freq,plobject.heights[hrange[0]:hrange[1]])
        plt.title('Brunt-Vaisala frequency')
        plt.ylabel('Height [km]')
        plt.xlabel('Frequency [s-1]')
        if save==True:
            plt.savefig(savename, format=saveformat, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return freq

# %%
def coriolis(plobject, lat=86, hrange=(0,-1), trange=(0,-1)):

    """ Calculate f=2*Omega*sin(lat) """

    wind, omeg = omega_profile(plobject, hrange, trange,
                               plot=False, save=False)
    # Global area-weighted vertical profile of rotation rate
    lat_rad = np.deg2rad(plobject.lats[lat])
    f = 2*omeg*np.sin(lat_rad)

    return f

# %%
def scaleheight(plobject, hrange=(0,-1), trange=(0,-1)):

    """ Approximate velocity of a gravity wave      """

    mean_temp = np.mean(plobject.data['temp'][trange[0]:trange[1],hrange[0]:hrange[1],:,:], axis=0)
    grid_areas = plobject.data['aire'][:]
    global_mean = np.sum(mean_temp*grid_areas[np.newaxis,:,:], axis=(1,2))/(plobject.area)
    numerator = (1.38e-23)*global_mean
    denom = 44.01*(1.67e-27)*plobject.g
    H = numerator/denom

    return H
  

# %%
def extratropical(plobject, lat=86, hrange=(0,-1), trange=(0,-1)):
    
    """ Calculate vertical profile of extratropical 
        Rossby radius of deformation                    
        see Carone et al. 2015                          """
    
    bv = bv_freq(plobject, hrange=hrange, trange=trange, plot=False, save=False)
    f = coriolis(plobject, lat, hrange, trange)
    H = scaleheight(plobject, hrange, trange)
    extra_r = bv*H/f
    
    return extra_r

# %%
def tropical(plobject, hrange=(0,-1), trange=(0,-1)):

    """ Calculate vertical profile of tropical Rossby
        radius of deformation
        see Carone et al. 2015                  """
    
    bv = bv_freq(plobject, hrange, trange, plot=False, save=False)
    zwind, omeg = omega_profile(plobject, hrange, trange,
                               plot=False, save=False)
    beta = 2*omeg/(plobject.radius*1000) 
    # Value for equator
    H = scaleheight(plobject, hrange, trange)
    trop_r = np.sqrt(bv*H/(2*beta))

    return trop_r

# %%
def plot_profiles(plobject, lat=86, hrange=(0,-1), trange=(0,-1),
                  save=False, 
                  saveformat='png', savename='radii_profiles.png'):

    """ Plot selected profiles calculated in the functions above"""

    L_r = extratropical(plobject, lat, hrange, trange)
    L_r = L_r/(plobject.radius*1000)
    lambda_r = tropical(plobject, hrange, trange)
    lambda_r = lambda_r/(plobject.radius*1000)

    fig, ax = plt.subplots(figsize=(6,8))
    plt.plot(L_r, plobject.heights[hrange[0]:hrange[1]], 
            color='b', label='Extratropical Rossby wave')
    plt.plot(lambda_r, plobject.heights[hrange[0]:hrange[1]], 
            color='r', label='Tropical Rossby wave')
    plt.plot(np.ones_like(lambda_r), 
             plobject.heights[hrange[0]:hrange[1]],
             color='k', linestyle='dashed',
             label='Wavenumber=1')
    plt.title('Meridional Rossby wavenumber')
    plt.ylabel('Height [km]')
    plt.xlabel('Rossby wavenumber')
    plt.xscale('log')
    plt.legend()
    if save==True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

