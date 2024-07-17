# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from venusdata import *

# %%
datapath = '/home/maureenjcohen/lmd_data/aoa_surface.nc'
#datapath = '/exomars/data/analysis/volume_8/mc5526/aoa_surface.nc'
# %%
sim = Planet(venusdict)
sim.load_file(datapath)
sim.setup()

# %%
def short_uniform(plobject, outpath, t_c, trange=[0,-1]):
    """ Calculate eddy diffusion coefficient profile for a short-lived
        tracer with uniform chemical equilibrium    
        
    Inputs:    Planet object containing simulation data 
               t_c is the chemical relaxation time-scale
               
    Outputs:    csv file containing pressure levels, approx. altitudes, and coefficients """

    if not hasattr(plobject,'w_wind'):
        plobject.calc_w(trange=trange) # Get w in m/s

    w_mean = np.mean(plobject.w_wind[trange[0]:trange[1],...], axis=0) # Time mean of w [m/s]
    w_squared = w_mean**2
    w_bar = np.sum(w_squared*plobject.data['aire'][:], axis=(1,2))/plobject.area
    # Area-weighted global mean on levels 
    print(w_bar)

    u_mean = np.mean(plobject.data['vitu'][trange[0]:trange[1],...], axis=0) # Time mean of u
    u_bar = np.sum(u_mean*plobject.data['aire'][:], axis=(1,2))/plobject.area 
    # Area-weighted global mean on levels
    t_d = plobject.radius*1e3/np.abs(u_bar)
    # Dynamical transport timescale: characteristic length (radius / horizontal wind)
    print(t_d)

    k_zz = w_bar/((1/t_d) + (1/t_c))
    output_dict = {'Pressure (Pa)': plobject.plevs,
                    'Altitude (km)': plobject.heights,
                    'Eddy diffusivity (m2 s-1)': k_zz}

    output = pd.DataFrame(data=output_dict)
    output.to_csv(outpath)

    return output

# %%
def long(plobject, outpath, trange=[0,-1]):
    """ Calculate eddy diffusion coefficient profile for a long-lived tracer
    Inputs:  Planet object containing simulation data
    
    Outputs:    csv file containing pressure levels, approx. altitudes, and coefficients """

    if not hasattr(plobject,'w_wind'):
        plobject.calc_w(trange=trange) # Get w in m/s

    w_mean = np.mean(plobject.w_wind[trange[0]:trange[1],...], axis=0) # Time mean of w [m/s]
    w_squared = w_mean**2
    w_bar = np.sum(w_squared*plobject.data['aire'][:], axis=(1,2))/plobject.area
    # Area-weighted global mean on levels 

    u_mean = np.mean(plobject.data['vitu'][trange[0]:trange[1],...], axis=0) # Time mean of u
    u_bar = np.sum(u_mean*plobject.data['aire'][:], axis=(1,2))/plobject.area 
    # Area-weighted global mean on levels
    t_d = plobject.radius*1e3/np.abs(u_bar)
    # Dynamical transport timescale: characteristic length (radius / horizontal wind)

    k_zz = w_bar*t_d
    output_dict = {'Pressure (Pa)': plobject.plevs,
                    'Altitude (km)': plobject.heights,
                    'Eddy diffusivity (m2 s-1)': k_zz}

    output = pd.DataFrame(data=output_dict)
    output.to_csv(outpath)

    return output

    



# %%
