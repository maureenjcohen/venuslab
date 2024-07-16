# %%
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

from venusdata import *


# %%
sim = Planet(venusdict)
sim.load_file('/exomars/data/analysis/volume_8/mc5526/aoa_surface.nc')
sim.setup()

# %%
def short_uniform(plobject, t_c):
    """ Calculate eddy diffusion coefficient profile for a short-lived
        tracer with uniform chemical equilibrium    
        
    Inputs:    Planet object containing simulation data 
               t_c is the chemical relaxation time-scale"""

    if not hasattr(plobject,'w_wind'):
        plobject.calc_w() # Get w in m/s

    w_mean = np.mean(plobject.w_wind, axis=0) # Time mean of w [m/s]
    w_bar = np.sum(w_mean*plobject.data['aire'][:], axis=(1,2))/plobject.area 
    # Area-weighted global mean on levels

    u_mean = np.mean(plobject.data['vitu'], axis=0) # Time mean of u
    u_bar = np.sum(u_mean*plobject.data['aire'][:], axis=(1,2))/plobject.area 
    # Area-weighted global mean on levels
    t_d = plobject.radius/u_bar
    # Dynamical transport timescale: characteristic length (radius / horizontal wind)

    k_zz = w_bar**2/(t_d + t_c)

    return k_zz
# %%
