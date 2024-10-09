# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# %%
def omega_profile(plobject, hrange=(0,-1), trange=(0,-1),
                  plot=False, save=False, gmean=True, lat=80,
                  saveformat='png', savename='uprofile.png'):

    """ Global area-weighted profile of rotation rate of
    the atmosphere (effective omega)        """

    zwind = np.mean(plobject.data['vitu'][trange[0]:trange[1],hrange[0]:hrange[1],:,:],0)
    if gmean==True:
        grid_areas = plobject.data['aire'][:]
        zmean = np.sum(zwind*grid_areas, axis=(1,2))/(plobject.area)
    # Area-weighted spatial and time mean of zonal wind
    else:
        zmean = np.mean(zwind,axis=-1)
        zmean = zmean[:,lat]

    circumf = (2*np.pi*(plobject.radius + plobject.heights[hrange[0]:hrange[1]])*1000)
    period = (circumf/np.abs(zmean))
    omega = (2*np.pi)/period
    period_days = period/(60*60*24)

    if plot==True:
        fig, ax = plt.subplots(figsize=(6,8))
        plt.plot(period_days,plobject.heights[hrange[0]:hrange[1]])
        plt.title('Rotation period of atmosphere')
        plt.ylabel('Height / km')
        plt.xlabel('Period / Earth days')
        plt.xscale('log')
        if save==True:
            plt.savefig(savename, format=saveformat, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return zmean, omega, period_days

# %%
def bv_freq(plobject, hrange=(0,-1), trange=(0,-1),
            plot=False, save=False, gmean=True, lat=80,
            saveformat='png', savename='bvprofile.png'):
    
    """ Global-area weighted profile of Brunt-Vaisala frequency"""

    if not hasattr(plobject, 'theta'):
        plobject.calc_theta() # Calculate potential temperature
    # Method uses formula from LMDZ Venus model, not standard one

    th_mean = np.mean(plobject.theta[trange[0]:trange[1],hrange[0]:hrange[1],:,:], axis=0)
    if gmean==True:
        grid_areas = plobject.data['aire'][:]
        th_prof = np.sum(th_mean*grid_areas, axis=(1,2))/(plobject.area)
        th_dz = np.gradient(th_prof)/np.gradient(plobject.heights[hrange[0]:hrange[1]]*1000)
    else:
        th_prof = np.mean(th_mean, axis=-1)
        th_prof = th_prof[:,lat]
        th_dz = np.gradient(th_prof)/np.gradient(plobject.heights[hrange[0]:hrange[1]]*1000)
    
    root_term = plobject.g*th_dz/th_prof
    freq = np.sqrt(root_term) 

    if plot==True:
        fig, ax = plt.subplots(figsize=(6,8))
        plt.plot(freq,plobject.heights[hrange[0]:hrange[1]])
        plt.title('Brunt-Vaisala frequency')
        plt.ylabel('Height / km')
        plt.xlabel('Frequency / s-1')
        if save==True:
            plt.savefig(savename, format=saveformat, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return freq

# %%
def construct_bv(plobject, bv_freq):
    """ Make a simplified BV frequency profile from surface to 65 km"""

    below_40 = [index for index, value in enumerate(plobject.heights) if value <= 40.]
    above_40 = [index for index, value in enumerate(plobject.heights) if value > 40. and value <= 65.]

    mean_lower_bv = np.nanmean(bv_freq[below_40[0]:below_40[-1]])
    # Constant value of BV for below 40 km
    bv_65 = bv_freq[above_40[-1]].values
    # Get BV value at or close to 65 km
    lower_y = plobject.heights[above_40[0]]
    upper_y = plobject.heights[above_40[-1]]
    # Get y range for linear fit
    m = (upper_y - lower_y)/(bv_65 - mean_lower_bv) 
 
    # Slope of line
    x = (plobject.heights[above_40[0]:] - lower_y)/m + mean_lower_bv

    bv_profile = np.ones_like(plobject.heights)
    bv_profile[above_40[0]:] = bv_freq[above_40[0]-1:]
    bv_profile[below_40[0]:below_40[-1]+1] = mean_lower_bv

    return bv_profile


# %%
def coriolis(plobject, gmean=True, lat=80, hrange=(0,-1), trange=(0,-1)):

    """ Calculate f=2*Omega*sin(lat) """

    wind, omeg, period_days = omega_profile(plobject, hrange, trange,
                               plot=False, gmean=gmean, lat=lat, save=False)
    # Global area-weighted vertical profile of rotation rate
    lat_rad = np.deg2rad(plobject.lats[lat])
    f = 2*omeg*np.sin(lat_rad)

    return f

# %%
def scaleheight(plobject, hrange=(0,-1), trange=(0,-1),
                gmean=True, lat=80):

    """ Scale height of atmosphere as a function of altitude  
        kT/mg where k is Boltzmann's constant, T is temp, m is average mass of atoms (CO2) in kg,
        and g is gravity
        This is equivalent to RT/Mg, where R is universal gas constant (8.314 Nm/molK), T
        is temp, M is molar mass (kg/mol), and g is gravity"""

    mean_temp = np.mean(plobject.data['temp'][trange[0]:trange[1],hrange[0]:hrange[1],:,:], axis=0)
    if gmean==True:
        grid_areas = plobject.data['aire'][:]
        global_mean = np.sum(mean_temp*grid_areas, axis=(1,2))/(plobject.area)
    else:
        global_mean = np.mean(mean_temp, axis=-1)
        global_mean = global_mean[:,lat]
    numerator = (1.38e-23)*global_mean
    denom = 44.01*(1.67e-27)*plobject.g
    H = numerator/denom

    return H
  

# %%
def extratropical(plobject, gmean=True, lat=80, hrange=(0,-1), trange=(0,-1)):
    
    """ Calculate vertical profile of extratropical 
        Rossby radius of deformation                    
        see Carone et al. 2015                          """
    bv_real = bv_freq(plobject, hrange=hrange, trange=trange, gmean=gmean, lat=lat, plot=False, save=False)
    bv = construct_bv(plobject, bv_real)
    f = coriolis(plobject, gmean=gmean, lat=lat, hrange=hrange, trange=trange)
    H = scaleheight(plobject, gmean=gmean, lat=lat, hrange=hrange, trange=trange)
    extra_r = bv[hrange[0]:hrange[-1]]*H/f
    
    return extra_r

# %%
def tropical(plobject, gmean=True, lat=80, hrange=(0,-1), trange=(0,-1)):

    """ Calculate vertical profile of tropical Rossby
        radius of deformation
        see Carone et al. 2015                  """

    bv_real = bv_freq(plobject, hrange, trange, gmean=gmean, lat=lat, plot=False, save=False)
    bv = construct_bv(plobject, bv_real)
    zwind, omeg, period_days = omega_profile(plobject, hrange, trange,
                               plot=False, gmean=gmean, lat=lat, save=False)
    beta = 2*omeg/(plobject.radius*1000) 
    # Value for equator
    H = scaleheight(plobject, lat=lat, gmean=gmean, hrange=hrange, trange=trange)
    trop_r = np.sqrt(bv[hrange[0]:hrange[-1]]*H/(2*beta))

    return trop_r

# %%
def plot_profiles(plobject, gmean=True, lat=64, hrange=(0,-1), trange=(0,-1),
                  save=False,
                  saveformat='png', savename='radii_profiles.png'):

    """ Plot selected profiles calculated in the functions above"""

    L_r = extratropical(plobject, gmean=gmean, lat=lat, hrange=hrange, trange=trange)
    L_r = L_r/(plobject.radius*1000)
    lambda_r = tropical(plobject, gmean=gmean, lat=lat, hrange=hrange, trange=trange)
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


# %%
def eddy_geop(plobject, time_slice=-100, lev=22):
    """ Plot eddy geopotential height for input model level"""

    geop = plobject.data['geop'][time_slice,lev,:,:]
    zonal_geop = np.mean(geop, axis=-1)
    eddy_geop = geop - zonal_geop[:,np.newaxis]

    fig, ax = plt.subplots(figsize=(8,5))
    cf = ax.contourf(plobject.lons, plobject.lats, eddy_geop,
                     cmap='coolwarm', norm=TwoSlopeNorm(0)
                     )
    cbar = plt.colorbar(cf, orientation='vertical')
    ax.set_title('Eddy geopotential height')
    ax.set_xlabel('Longitude / deg')
    ax.set_ylabel('Latitude / deg')
    plt.show()
    
# %%
