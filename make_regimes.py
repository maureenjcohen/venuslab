""" Uses functions from venuslab to auto-generate plots for
    'An altitude-dependent circulation regime change in the Venus atmosphere',
     Cohen et al. 2024                                       """

""" Usage from command line: python make_regimes.py
    No inputs required as filepaths are specified in script """

# Input file paths and outputs - only part of script that should be edited
# %%
surfacepath = '/exomars/data/analysis/volume_8/mc5526/aoa_surface.nc'
# Simulation with surface age of air tracer - baseline model state

# Cleaned Pioneer Venus data
day_probe = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/cleaned_day_probe.csv'
night_probe = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/cleaned_night_probe.csv'
north_probe = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/cleaned_north_probe.csv'

# %%
from venusdata import *
from venusaoa import *
from venusrossby import *
from venuspioneer import *

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

# %%
def init_model_data(inpath):
    """ Instantiate Planet object from Venus PCM output data"""

    plobject = Planet(venusdict)
    plobject.load_file(inpath)
    plobject.setup()

    return plobject

# %%
def init_probes(inpathlist):
    """ Instantiate Probe objects for 3 Pioneer Venus descent probes"""

    probelist = []
    for inpath in inpathlist:
        probe_name = inpath.split('_')[-2].capitalize()
        # Extract name of Probe - Day, Night, North
        probe = Probe(inpath, probe_name)
        # Create Probe data object with data and name
        probelist.append(probe)

    return probelist

# %%
def compare_profiles(plobject, probelist, hrange=(0,-1), fsize=14,
                     savearg=False, savename='fig2_profiles.png',
                     sformat='png'):
    """ Figure with two sub-plots
        1) Zonal wind profiles from 3 PV probes plus Venus PCM 
        2) Rotation period profiles from 3 PV probes plus Venus PCM 
        
        Venus PCM vertical domain is greater than descent probes, so
        we only plot heights up to (inclusive of) model level 30 """
    
    vpcm_zmean, vpcm_omega, vpcm_period = omega_profile(plobject, hrange=hrange, plot=False, save=False)
    # Calculate global area-weighted mean zonal wind and atmospheric omega and rotation period
    # for Venus PCM output

    colors=['tab:blue','tab:green','tab:orange']
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,6))
    fig.suptitle('Venus PCM vs. Pioneer Venus entry probes', y=1.01, fontsize=fsize+6)
    # Create figure with two sub-plots on the same y-axis
    for ind, probe in enumerate(probelist):
        ax1.plot(probe.data['WEST'].values, probe.data['ALT(KM)'].values, color=colors[ind], label=probe.name)    
    ax1.plot(vpcm_zmean, plobject.heights[:hrange[1]], color='k', linestyle='dashed', label='Venus PCM')
    ax1.set_title('Zonal wind', fontsize=fsize)
    ax1.set_xlabel('Zonal wind / m/s', fontsize=fsize)
    ax1.set_ylabel('Altitude / km', fontsize=fsize)
    ax1.legend()

    for ind, probe in enumerate(probelist):
            circumf = 2*np.pi*((plobject.radius + probe.data['ALT(KM)'].values)*1000)
            period = (circumf/np.abs(probe.data['WEST'].values))
            omega = (2*np.pi)/period
            period_days = period/(60*60*24)
            ax2.plot(period_days, probe.data['ALT(KM)'].values, color=colors[ind], label=probe.name)
    ax2.plot(vpcm_period, plobject.heights[:hrange[1]], color='k', linestyle='dashed', label='Venus PCM')
    ax2.set_title('Rotation period of atmosphere', fontsize=fsize)
    ax2.set_xlabel('Rotation period / Earth days', fontsize=fsize)
    ax2.set_xscale('log')
    ax2.legend()

    plt.subplots_adjust(wspace=0.1)
    
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compare_rossby(plobject, probelist, lat=64, hrange=(0,-1), trange=(0,-1)):
    """ Figure with single sub-plot:
        Rossby radius of deformation as a function of altitude for
        VPCM model output
        Pioneer Venus entry probe wind data                 """
    
    L_vpcm = extratropical(plobject, lat, hrange, trange)
    L_vpcm = L_vpcm/(plobject.radius*1000)
    lambda_vpcm = tropical(plobject, hrange, trange)
    lambda_vpcm = lambda_vpcm/(plobject.radius*1000)

# %%
if __name__ == "__main__":

    vpcm = init_model_data(surfacepath)
    # Instantiate Venus PCM model data object
    pv_probes = init_probes([day_probe, night_probe, north_probe])
    # Instantiate Probe objects for Pioneer Venus data
    # And colllect in a list

    compare_profiles(vpcm, pv_probes, hrange=(0,-1), fsize=14,
                     savearg=True, savename='fig2_profiles.png',
                     sformat='png')