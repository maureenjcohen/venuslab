# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from venusdata import *

# %%
#datapath = '/home/maureenjcohen/lmd_data/aoa_surface.nc'
datapath = '/exomars/data/analysis/volume_8/mc5526/aoa_surface.nc'
# %%
venus = Planet(venusdict)
venus.load_file(datapath)
venus.setup()

# %%
def calc_t_c_so2(plobject):
    """ Approximate vertical profile of chemical lifetime of
        SO2 in the Venus atmosphere
        
        For VPCM, pre-cloud/cloud starts at level 22
        Cloud top is set at level 35                """
    
    deep_t_c = 1e13
    # Set deep atmosphere thermochemical lifetime to 10^13 seconds
    # aka hundreds of thousands of years

    top_t_c = 1e5
    # Set cloud top lifetime to 10^5 seconds, around one day

    tc_array = np.ones_like(plobject.heights)
    tc_array[0:23] = tc_array[0:23]*deep_t_c
    tc_array[36:] = tc_array[36:]*top_t_c

    tc_array[23:36] = 1.5e7 # Guess around 6 months chemical lifetime in clouds

#    res_times = np.diff(plobject.data['age'][-1,:,48,48])
#    tc_array[23:36] = res_times[22:35]

#    tc_array[23:36] = np.interp(plobject.heights[23:36], 
#                                np.hstack((plobject.heights[0:23], plobject.heights[36:])),
#                                np.hstack((tc_array[0:23], tc_array[36:])))

    return tc_array

# %%
def calc_t_d(plobject, trange=[0,-1]):
    """ Horizontal advective dynamical timescale 
        Planetary radius divided by global long-term mean zonal wind """
    
    u_mean = np.mean(plobject.data['vitu'][trange[0]:trange[1],...], axis=0) # Time mean of u
    u_bar = np.sum(u_mean*plobject.data['aire'][:], axis=(1,2))/plobject.area 
    # Area-weighted global mean on levels

    t_d = plobject.radius*1e3/np.abs(u_bar)
    # Dynamical transport timescale: characteristic length (radius / horizontal wind)

    return t_d

# %%
def calc_w_bar(plobject, trange=[0,-1]):
    """ Global long-term mean of vertical wind squared on levels """

    if not hasattr(plobject,'w_wind'):
        plobject.calc_w(trange=trange) # Get w in m/s

    w_mean = np.mean(plobject.w_wind[trange[0]:trange[1],...], axis=0) # Time mean of w [m/s]
    w_squared = w_mean**2
    w_bar = np.sum(w_squared*plobject.data['aire'][:], axis=(1,2))/plobject.area
    # Area-weighted global mean on levels 

    return w_bar


# %%
def short_uniform(plobject, outpath, t_c, t_d, w_bar,
                  save=False):
    """ Calculate eddy diffusion coefficient profile for a short-lived
        tracer with uniform chemical equilibrium    
        
    Inputs:    Planet object containing simulation data 
               t_c is the chemical relaxation time-scale
               t_d is dynamical timescale
               w_bar is global long-term mean of squared vertical wind
               
    Outputs:    csv file containing pressure levels, approx. altitudes, and coefficients """

    k_zz = w_bar/((1/t_d) + (1/t_c))

    if save==True:
        output_dict = {'Pressure (Pa)': plobject.plevs,
                    'Altitude (km)': plobject.heights,
                    'Eddy diffusivity (m2 s-1)': k_zz}

        output = pd.DataFrame(data=output_dict)
        output.to_csv(outpath)
        return output

    else:
        return k_zz

# %%
def long(plobject, outpath, t_d, w_bar,
         save=False):
    """ Calculate eddy diffusion coefficient profile for a long-lived tracer
    
    Inputs:  Planet object containing simulation data
             t_d is dynamical timescale
             w_bar is global long-term mean of squared vertical wind
    
    Outputs:    csv file containing pressure levels, approx. altitudes, and coefficients """

    k_zz = w_bar*t_d

    if save==True:
        output_dict = {'Pressure (Pa)': plobject.plevs,
                    'Altitude (km)': plobject.heights,
                    'Eddy diffusivity (m2 s-1)': k_zz}

        output = pd.DataFrame(data=output_dict)
        output.to_csv(outpath)
        return output
    
    else:
        return k_zz

# %%
def make_dataset(plobject, trange=[0,-1]):
    """ Create dataset with eddy diffusivity for long-lived tracer
        and SO2                                     """
    
    init_dict = {'Pressure (Pa)': plobject.plevs,
                 'Altitude (km)': plobject.heights,
                 'Age of air (s)': plobject.data['age'][-1,:,48,48]}
    
    df = pd.DataFrame(data=init_dict)

    t_d = calc_t_d(plobject, trange=trange)
    w_bar = calc_w_bar(plobject, trange=trange)
    t_c_so2 = calc_t_c_so2(plobject)
    kzz_long = long(plobject, outpath=None, t_d=t_d, 
                    w_bar=w_bar, save=False)
    kzz_so2 = short_uniform(plobject, outpath=None, t_c=t_c_so2, 
                            t_d=t_d, w_bar=w_bar, save=False)

    df['Dynamical timescale (s)'] = t_d
    df['K_zz (long) (m2 s-1)'] = kzz_long
    df['Chemical timescale SO2 (s)'] = t_c_so2
    df['K_zz (SO2) (m2 s-1)'] = kzz_so2

    return df

# %%
