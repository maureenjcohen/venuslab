""" Uses functions from venuslab to auto-generate plots for
    'Three worlds in one: Venus as a natural laboratory' etc.,
     Cohen et al. 2024                                       """

""" Usage from command line: python make_regimes.py
    No inputs required as filepaths are specified in script """

# Input file paths and outputs - only part of script that should be edited
# %%
surfacepath = '/exomars/data/analysis/volume_8/mc5526/aoa_surface.nc'
surfaceplevs = '/exomars/data/analysis/volume_8/mc5526/aoa_surface_plevs.nc'
# Simulation with surface age of air tracer - baseline model state
cloudpath = '/exomars/data/analysis/volume_9/mc5526/lmd_data/aoa_cloud.nc'
outpath = '/exomars/data/analysis/volume_8/mc5526/make_regimes/'

# Import packages
# %%
from venusdata import *

import numpy as np
import matplotlib.pyplot as plt
#import netCDF4 as nc
import windspharm
from matplotlib.colors import TwoSlopeNorm

# %%
def init_model_data(inpath, modelname, simname):
    """ Instantiate Planet object from Venus PCM output data"""

    plobject = Planet(venusdict, modelname, simname)
    plobject.load_file(inpath)
#    plobject.setup()

    return plobject

# %%
def helm_panels(plobject, time_slice=-99, levs=[12,20,30], qscale=[0.1,1,2], 
                qmultiplier=0.5, n=3, fsize=14, savearg=False, savename='fig1_regimes.png',
                sformat='png'):
    """ Figure with 6 sub-figures, showing the wind vectors and Helmholtz decomp
        at 3 different altitude levels                                  """
    
    u = -np.array([plobject.data['vitu'][time_slice,lev,:,:] for lev in levs])
    v = np.array([plobject.data['vitv'][time_slice,lev,:,:] for lev in levs])
    omega = np.array([plobject.data['vitw'][time_slice,lev,:,:] for lev in levs])
    temp = np.array([plobject.data['temp'][time_slice,lev,:,:] for lev in levs])
    pres = np.array([plobject.data['pres'][time_slice,lev,:,:] for lev in levs])
    w = -np.array((omega*temp*plobject.RCO2)/(pres*plobject.g))
    geop = np.array([plobject.data['geop'][time_slice,lev,:,:] for lev in levs])/plobject.g

    eddy_us = []
    eddy_vs = []

    for lev in range(0,len(levs)-1):
        eddy_u = u[lev,:,:] - np.mean(u[lev,:,:], axis=-1)[:,np.newaxis]
        eddy_v = v[lev,:,:] - np.mean(v[lev,:,:], axis=-1)[:,np.newaxis]
        eddy_us.append(eddy_u)
        eddy_vs.append(eddy_v)
    # For first two panels, just calculate the eddy wind from the wind fields
    # and append to list

    winds = windspharm.standard.VectorWind(np.flip(u[-1,:,:], axis=(0,1)), 
                                           np.flip(v[-1,:,:], axis=(0,1)), 
                                           rsphere=plobject.radius*1000)
    # Create a VectorWind data object from the x and y wind cubes
    uchi, vchi, upsi, vpsi = winds.helmholtz(truncation=21)
    
    zonal_upsi = np.mean(upsi, axis=-1)
    zonal_vpsi = np.mean(vpsi, axis=-1)

    eddy_upsi = np.flip(upsi - zonal_upsi[:,np.newaxis], axis=(0,1))
    eddy_vpsi = np.flip(vpsi - zonal_vpsi[:,np.newaxis], axis=(0,1))
    eddy_us.append(eddy_upsi)
    eddy_vs.append(eddy_vpsi)
    # For the third panel, first do the Helmholtz decomposition, then find
    # the eddy rotational component and append to list

    eddy_geops = []
    for lev in range(0,len(levs)):
        eddy_geop = geop[lev,:,:] - np.mean(geop[lev,:,:], axis=-1)[:,np.newaxis]
        eddy_geops.append(eddy_geop)

    X, Y = np.meshgrid(plobject.lons, plobject.lats)
    fig, ax = plt.subplots(len(levs), 2, figsize=(12,12), sharex=True, sharey=False)

    plot_labels = list(map(chr, range(ord('a'), ord('z')+1))) # List containing letters of the alphabet
    for i in range(0,len(levs)):
        alph_i = i*2 # Index x2, equivalent to skipping every other letter of the alphabet
        cf = ax[i,0].contourf(plobject.lons, plobject.lats, w[i,:,:],
                            levels=np.arange(-0.04,0.08,0.02),
                            cmap='coolwarm', extend='both', norm=TwoSlopeNorm(0))
        lev_winds = ax[i,0].quiver(X[::n, ::n], Y[::n, ::n], u[i,::n,::n],
                    v[i,::n,::n], angles='xy', scale_units='xy', scale=qscale[i])
        ax[i,0].quiverkey(lev_winds, X=0.9, Y=1.05, U=qscale[i]*10, label='%s m/s' %str(qscale[i]*10),
                    labelpos='E', coordinates='axes')
        ax[i,0].set_ylabel('Latitude / deg', fontsize=fsize)
        if i == len(levs)-1:
            ax[i,0].set_xlabel('Longitude / deg', fontsize=fsize)
        ax[i,0].set_title(f'{plot_labels[alph_i]}) Horiz. wind, h={np.round(plobject.heights[levs[i]],0)} km', fontsize=fsize-4)
        cbar = plt.colorbar(cf, ax=ax[i,0])
        cbar.set_label('Vertical wind / m/s', loc='center')
        lat_ticks = ax[i,0].get_yticks()

        if i > 1:
            title_str = 'Eddy rot. comp.'
        else:
            title_str = 'Eddy wind'

        geop_cf = ax[i,1].contourf(plobject.lons, plobject.lats, eddy_geops[i],
                                   alpha=0.5,
                                   cmap='PuOr', extend='both', norm=TwoSlopeNorm(0))
        
        lev1_helm = ax[i,1].quiver(X[::n,::n], Y[::n,::n], eddy_us[i][::n,::n], eddy_vs[i][::n,::n],
                    angles='xy', scale_units='xy', scale=qscale[i]*qmultiplier)
        ax[i,1].quiverkey(lev1_helm, X=0.9, Y=1.05, U=qscale[i]*qmultiplier*10, label='%s m/s' %str(qscale[i]*qmultiplier*10),
                    labelpos='E', coordinates='axes')
        ax[i,1].set_title(f'{plot_labels[alph_i+1]}) {title_str}, h={np.round(plobject.heights[levs[i]],0)} km', fontsize=fsize-4)
        if i == len(levs)-1:
            ax[i,1].set_xlabel('Longitude / deg', fontsize=fsize)
        ax[i,1].set_yticks(lat_ticks[1:-1])
        gbar = plt.colorbar(geop_cf, ax=ax[i,1])
        gbar.set_label('Eddy geop. height / m', loc='center')
    
    plt.subplots_adjust(wspace=0.2)
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def vort_altlat(plobject, lon=48, time_slice=1818, hmin=0, hmax=-1,
                fsize=14, savearg=False, savename='fig4a_relvort.png',
                sformat='png'):
    """ Altitude-longitude plot of eddy relative vorticity
    Good snapshots: surface 1249, 1256, 1265, 1274, 1433, 1733, 1812, 1818, 1836, 1873"""

    u = -np.flip(plobject.data['vitu'][time_slice,hmin:hmax,:,:], axis=(1,2))
    v = np.flip(plobject.data['vitv'][time_slice,hmin:hmax,:,:], axis=(1,2))
    u = np.transpose(u, axes=[1,2,0])
    v = np.transpose(v, axes=[1,2,0])

    winds = windspharm.standard.VectorWind(u, v, rsphere=plobject.radius*1000)
    div = np.flip(np.transpose(winds.divergence(truncation=21), axes=[2,0,1]), axis=(1,2))
    vrt = np.flip(np.transpose(winds.vorticity(truncation=21), axes=[2,0,1]), axis=(1,2))
    eddy_vrt = vrt - np.mean(vrt, axis=-1)[:,:,np.newaxis]

    fig, ax = plt.subplots(figsize=(6,6))
    cf = ax.contourf(plobject.lats, plobject.heights[hmin:hmax], 
                     eddy_vrt[:,:,lon]*1e5,
                     levels=np.arange(-1.6, 1.7, 0.1),
                     extend='both',
                     cmap='coolwarm', norm=TwoSlopeNorm(0))
    ax.set_title(f'a) Eddy relative vorticity, {int(np.round(plobject.lons[lon],0))}$^{{\circ}}$E/W',
                 fontsize=fsize)
    ax.set_xlabel('Latitude / deg', fontsize=fsize)
    ax.set_ylabel('Height / km', fontsize=fsize)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Relative vorticity / $10^{-5}$ s-1', loc='center')
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# %%
if __name__ == "__main__":

    surface = init_model_data(surfacepath, 'vpcm', 'surface')
    cloud = init_model_data(cloudpath, 'vpcm', 'cloud')

    helm_panels(surface, levs=[12,20,30], savearg=False, 
                savename='/exomars/data/analysis/volume_8/mc5526/make_regimes/fig1_regimes_withgeop.png', 
                sformat='png')
    
    vort_altlat(surface, time_slice=1818, savearg=False, 
                savename='/exomars/data/analysis/volume_8/mc5526/make_regimes/fig4a_eddy_vort.png', 
                sformat='png')
