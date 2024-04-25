""" Uses functions from venuslab to auto-generate plots for
    'An altitude-dependent circulation regime change in the Venus atmosphere',
     Cohen et al. 2024                                       """

""" Usage from command line: python make_regimes.py
    No inputs required as filepaths are specified in script """

# Input file paths and outputs - only part of script that should be edited
# %%
vpcm_path = '/home/maureenjcohen/lmd_data/aoa_surface.nc'
# Simulation with surface age of air tracer - baseline model state

# Import packages
# %%
from venusdata import *

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import windspharm

# %%
def init_model_data(inpath):
    """ Instantiate Planet object from Venus PCM output data"""

    plobject = Planet(venusdict)
    plobject.load_file(inpath)
    plobject.setup()

    return plobject

# %%
def helm_panels(plobject, time_slice=-2, levs=[8,15,22], qscale=1, 
                n=3, fsize=14):
    """ Figure with 6 sub-figures, showing the wind vectors and Helmholtz decomp
        at 3 different altitude levels                                  """
    
    u = -np.copy([plobject.data['vitu'][time_slice,lev,:,:] for lev in levs])
    v = np.copy([plobject.data['vitv'][time_slice,lev,:,:] for lev in levs])

    winds = windspharm.standard.VectorWind(u, v)
    # Create a VectorWind data object from the x and y wind cubes
    uchi, vchi, upsi, vpsi = winds.helmholtz(truncation=21)

    zonal_upsi = np.mean(upsi, axis=-1)
    zonal_vpsi = np.mean(vpsi, axis=-1)

    eddy_vpsi = vpsi - zonal_vpsi[:,:,np.newaxis]

    X, Y = np.meshgrid(plobject.lons, plobject.lats)
    fig, ax = plt.subplots(len(levs), 2, figsize=(12,12), sharex=True, sharey=True)

    plot_labels = list(map(chr, range(ord('a'), ord('z')+1))) # List containing letters of the alphabet
    for i in range(0,len(levs)):
        alph_i = i*2 # Index x2, equivalent to skipping every other letter of the alphabet
        lev1_winds = ax[i,0].quiver(X[::n, ::n], Y[::n, ::n], -u[i,::n,::n],
                    v[i,::n,::n], angles='xy', scale_units='xy', scale=qscale)
        ax[i,0].quiverkey(lev1_winds, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                    labelpos='E', coordinates='axes')
        ax[i,0].set_ylabel('Latitude / deg', fontsize=fsize)
        if i == len(levs)-1:
            ax[i,0].set_xlabel('Longitude /deg', fontsize=fsize)
        ax[i,0].set_title(f'{plot_labels[alph_i]}) Horizontal wind, h={np.round(plobject.heights[levs[0]],0)} km', fontsize=fsize)

        lev1_helm = ax[i,1].quiver(X[::n,::n], Y[::n,::n], eddy_upsi[i,::n,::n], eddy_vpsi[i,::n,::n],
                    angles='xy', scale_units='xy', scale=qscale)
        ax[i,1].quiverkey(lev1_helm, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                    labelpos='E', coordinates='axes')
        ax[i,1].set_title(f'{plot_labels[alph_i+1]}) Eddy rotational component, h={np.round(plobject.heights[levs[0]],0)} km', fontsize=fsize)
        if i == len(levs)-1:
            ax[i,1].set_xlabel('Longitude /deg', fontsize=fsize)

    plt.show()
# %%
