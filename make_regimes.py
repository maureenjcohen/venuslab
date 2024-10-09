""" Uses functions from venuslab to auto-generate plots for
    'An altitude-dependent circulation regime change in the Venus atmosphere',
     Cohen et al. 2024                                       """

""" Usage from command line: python make_regimes.py
    No inputs required as filepaths are specified in script """

# Input file paths and outputs - only part of script that should be edited
# %%
surfacepath = '/exomars/data/analysis/volume_8/mc5526/aoa_surface.nc'
cloudpath = '/exomars/data/analysis/volume_9/mc5526/lmd_data/aoa_cloud.nc'
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

    plobject = Planet(venusdict, 'vpcm', 'cloud')
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
    
    vpcm_zmean60, vpcm_omega60, vpcm_period60 = omega_profile(plobject, hrange=hrange, 
                                                        gmean=False, lat=80,
                                                        plot=False, save=False)
    # Calculate mean zonal wind and atmospheric omega and rotation period at 60 N
    # for Venus PCM output
    vpcm_zmean30, vpcm_omega30, vpcm_period30 = omega_profile(plobject, hrange=hrange, 
                                                        gmean=False, lat=32,
                                                        plot=False, save=False)
    # Calculate mean zonal wind and atmospheric omega and rotation period at 60 N
    # for Venus PCM output

    colors=['tab:blue','tab:green','tab:orange']
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,6))
    fig.suptitle('Venus PCM vs. Pioneer Venus entry probes', y=1.01, fontsize=fsize+6)
    # Create figure with two sub-plots on the same y-axis
    for ind, probe in enumerate(probelist):
        ax1.plot(probe.data['WEST'].values, probe.data['ALT(KM)'].values, color=colors[ind], label=probe.name + ', ' + probe.latstr)    
    ax1.plot(vpcm_zmean30, plobject.heights[:hrange[1]], color='r', linestyle='dashed', label='Venus PCM, $30^{\circ}$S')
    ax1.plot(vpcm_zmean60, plobject.heights[:hrange[1]], color='k', linestyle='dashed', label='Venus PCM, $60^{\circ}$N')
    ax1.set_title('Zonal wind', fontsize=fsize)
    ax1.set_xlabel('Zonal wind / m/s', fontsize=fsize)
    ax1.set_ylabel('Height / km', fontsize=fsize)
    ax1.legend()

    for ind, probe in enumerate(probelist):
            if not hasattr(probe,'period_days'):
                probe.calc_omega()
            period_days = probe.period
            ax2.plot(period_days, probe.data['ALT(KM)'].values, color=colors[ind], label=probe.name + ', ' + probe.latstr)
    ax2.plot(vpcm_period30, plobject.heights[:hrange[1]], color='r', linestyle='dashed', label='Venus PCM, $30^{\circ}$S')
    ax2.plot(vpcm_period60, plobject.heights[:hrange[1]], color='k', linestyle='dashed', label='Venus PCM, $60^{\circ}$N')

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

# %%
def compare_rossby(plobject, probelist, trop_lat=48, extra_lat=80, 
                   trop_gmean=False, extra_gmean=False,
                   hrange=(0,-1), trange=(0,-1),
                   savearg=False, fsize=14,
                   sformat='png', savename='fig3_radii.png'):
    """ Figure with single sub-plot:
        Rossby radius of deformation as a function of altitude for
        VPCM model output
        Pioneer Venus entry probe wind data                 """
    
    L_vpcm = extratropical(plobject, gmean=extra_gmean, lat=extra_lat, hrange=hrange, trange=trange)
    L_vpcm = L_vpcm/(plobject.radius*1000)
    lambda_vpcm = tropical(plobject, gmean=trop_gmean, lat=trop_lat, hrange=hrange, trange=trange)
    lambda_vpcm = lambda_vpcm/(plobject.radius*1000)

    colors=['tab:blue','tab:green','tab:orange']
    fig, ax = plt.subplots(figsize=(6,8))
    for ind, probe in enumerate(probelist):
            probe.calc_rossby_radii()
            L_probe = probe.extra_r/(probe.radius*1000)
            lambda_probe = probe.trop_r/(probe.radius*1000)
            plt.plot(L_probe, probe.data['ALT(KM)'].values, 
                         linestyle='dashed',
                         color=colors[ind], label=probe.name+', ' +probe.latstr+', extratropical')
            plt.plot(lambda_probe, probe.data['ALT(KM)'].values, 
                         color=colors[ind], label=probe.name+', ' +probe.latstr+', tropical')
    
    plt.plot(L_vpcm, plobject.heights[hrange[0]:hrange[1]], 
            color='r', linestyle='dashed', label='VPCM, $60^{\circ}$N, extratropical')
    plt.plot(lambda_vpcm, plobject.heights[hrange[0]:hrange[1]], 
            color='r', label='VPCM, $30^{\circ}$S, tropical')
    plt.plot(np.ones_like(lambda_vpcm), 
             plobject.heights[hrange[0]:hrange[1]],
             color='k', linestyle='dotted',
             label='Wavenumber=1')
    plt.title('Meridional Rossby wavenumber', fontsize=fsize+2)
    plt.ylabel('Height / km', fontsize=fsize)
    plt.xlabel('Rossby wavenumber', fontsize=fsize)
    plt.xscale('log')
    plt.legend()
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def allwaves(plobject, trange=(1300,1500), fsize=14,
             hmin=12, hmax=40, lat=88, lon=48,
             savearg=False, savename='fig4b_allwaves.png', sformat='png'):
     
    """ Hovmoeller plot of temperature anomaly as a function of height 
    To show high-frequency waves, requires aoa_cloud.nc as input! """

    air_temp = plobject.data['temp'][trange[0]:trange[1],hmin:hmax,lat,:]
    zm_temp = np.mean(air_temp, axis=-1)
    cube = air_temp - zm_temp[:,:,np.newaxis]
    time_axis = np.arange(0,len(plobject.data['time_counter'][trange[0]:trange[1]]))   
    time_axis = time_axis*(117/100)

    fig, ax = plt.subplots(figsize=(6,6))
    plt.contourf(time_axis, plobject.heights[hmin:hmax], 
                 cube[:,:,lon].T, 
                 levels=np.arange(-18,19,2),
                 extend='both',
                 norm=TwoSlopeNorm(0),
                 cmap='coolwarm')
    plt.title(f'b) Temperature anomaly, 0$^{{\circ}}$E/W x {int(np.round(plobject.lats[lat],0))}$^{{\circ}}$N',
              fontsize=fsize)
    plt.xlabel('Time / Earth days', fontsize=fsize)
    plt.ylabel('Height / km', fontsize=fsize)
    cbar = plt.colorbar()
    cbar.set_label('K')
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
    
    compare_rossby(vpcm, pv_probes, trop_lat=32, extra_lat=80, 
                   trop_gmean=False, extra_gmean=False,
                   hrange=(0,-1), trange=(0,-1), fsize=14, savearg=False,
                   savename='fig3_rossby.png', sformat='png')
    
    allwaves(vpcm, savearg=False, savename='fig4b_temp_anomaly.png',
             sformat='png')