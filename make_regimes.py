""" Uses functions from venuslab to auto-generate plots for
    'An altitude-dependent circulation regime change in the Venus atmosphere',
     Cohen et al. 2024                                       """

""" Usage from command line: python make_regimes.py
    No inputs required as filepaths are specified in script """

# Input file paths and outputs - only part of script that should be edited
# %%
surfacepath = '/exomars/data/analysis/volume_8/mc5526/aoa_surface.nc'
cloudpath = '/exomars/data/analysis/volume_9/mc5526/lmd_data/aoa_cloud.nc'
outpath = '/exomars/data/analysis/volume_8/mc5526/make_regimes/'
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
def init_model_data(inpath, modelname, simname):
    """ Instantiate Planet object from Venus PCM output data"""

    plobject = Planet(venusdict, modelname, simname)
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
def compare_profiles(plobject, probelist, fsize=14,
                     savearg=False, savename='fig2_profiles.png',
                     sformat='png'):
    """ Figure with two sub-plots
        1) Zonal wind profiles from 3 PV probes plus Venus PCM 
        2) Rotation period profiles from 3 PV probes plus Venus PCM 
        
        Venus PCM vertical domain is greater than descent probes, so
        we only plot heights up to (inclusive of) model level 30 """
    
    vpcm_zmean60, vpcm_omega60, vpcm_period60 = omega_profile(plobject, 
                                                        gmean=False, lat=80,
                                                        plot=False, save=False)
    # Calculate mean zonal wind and atmospheric omega and rotation period at 60 N
    # for Venus PCM output
    vpcm_zmean30, vpcm_omega30, vpcm_period30 = omega_profile(plobject, 
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
    ax1.plot(vpcm_zmean30, plobject.heights, color='r', linestyle='dashed', label='Venus PCM, $30^{\circ}$S')
    ax1.plot(vpcm_zmean60, plobject.heights, color='k', linestyle='dashed', label='Venus PCM, $60^{\circ}$N')
    ax1.set_title('a) Zonal wind', fontsize=fsize)
    ax1.set_xlabel('Zonal wind / m/s', fontsize=fsize)
    ax1.set_ylabel('Height / km', fontsize=fsize)
    ax1.legend()

    for ind, probe in enumerate(probelist):
            if not hasattr(probe,'period_days'):
                probe.calc_omega()
            period_days = probe.period
            ax2.plot(period_days, probe.data['ALT(KM)'].values, color=colors[ind], label=probe.name + ', ' + probe.latstr)
    ax2.plot(vpcm_period30, plobject.heights, color='r', linestyle='dashed', label='Venus PCM, $30^{\circ}$S')
    ax2.plot(vpcm_period60, plobject.heights, color='k', linestyle='dashed', label='Venus PCM, $60^{\circ}$N')

    ax2.set_title('b) Rotation period of atmosphere', fontsize=fsize)
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
def compare_rossby(plobject, probelist, trop_lat=32, extra_lat=80, 
                   trop_gmean=False, extra_gmean=False,
                   trange=(0,-1),
                   savearg=False, fsize=14,
                   sformat='png', savename='fig3_radii.png'):
    """ Figure with single sub-plot:
        Rossby radius of deformation as a function of altitude for
        VPCM model output
        Pioneer Venus entry probe wind data                 """
    
    L_vpcm = extratropical(plobject, gmean=extra_gmean, lat=extra_lat, trange=trange, constructed=True)
    L_vpcm = L_vpcm/(plobject.radius*1000)
    lambda_vpcm = tropical(plobject, gmean=trop_gmean, lat=trop_lat, trange=trange, constructed=True)
    lambda_vpcm = lambda_vpcm/(plobject.radius*1000)

    colors=['tab:blue','tab:green','tab:orange']
    fig, ax = plt.subplots(figsize=(6,8))
    for ind, probe in enumerate(probelist):
            probe.calc_rossby_radii(constructed=True)
            L_probe = probe.extra_r_constructed/(probe.radius*1000)
            lambda_probe = probe.trop_r_constructed/(probe.radius*1000)
            plt.plot(L_probe, probe.data['ALT(KM)'].values, 
                         linestyle='dashed',
                         color=colors[ind], label=probe.name+', ' +probe.latstr+', extratropical')
            plt.plot(lambda_probe, probe.data['ALT(KM)'].values, 
                         color=colors[ind], label=probe.name+', ' +probe.latstr+', tropical')
    
    plt.plot(L_vpcm, plobject.heights, 
            color='r', linestyle='dashed', label='VPCM, $60^{\circ}$N, extratropical')
    plt.plot(lambda_vpcm, plobject.heights, 
            color='r', label='VPCM, $30^{\circ}$S, tropical')
    plt.plot(np.ones_like(lambda_vpcm), 
             plobject.heights,
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
    cube = air_temp - zm_temp
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
def sensitivity_bv(plobject, probelist, fsize=14, savearg=False, 
                   savename='appendix_fig1_bv.png', sformat='png'):
    
    """ 2x2 plot showing different BV profiles and their impact on Rossby wavenumber """
    vpcm_30S = bv_freq(plobject, gmean=False, lat=32)
    vpcm_60N = bv_freq(plobject, gmean=False, lat=80)
    vpcm_30S_constructed = construct_bv(plobject, vpcm_30S)
    vpcm_60N_constructed = construct_bv(plobject, vpcm_60N)

    for probe in probelist:
        probe.calc_bv_freq()
        probe.construct_bv()

    L_vpcm = extratropical(plobject, gmean=False, lat=80, constructed=False)/(plobject.radius*1000)
    lambda_vpcm = tropical(plobject, gmean=False, lat=32, constructed=False)/(plobject.radius*1000)
    L_vpcm_constructed = extratropical(plobject, gmean=False, lat=80, constructed=True)/(plobject.radius*1000)
    lambda_vpcm_constructed = tropical(plobject, gmean=False, lat=32, constructed=True)/(plobject.radius*1000)

    fig, ax = plt.subplots(2, 2, sharey=True, figsize=(12,12))
    colors=['tab:blue','tab:green','tab:orange']
    fig.suptitle(r'Sensitivity of Rossby wavenumber to Brunt-Väisälä frequency',
                 y=0.95, fontsize=fsize+6)
    ax[0,0].set_title('a) Calculated BV frequency profile', fontsize=fsize)
    ax[0,0].plot(vpcm_30S, plobject.heights, color='k', label='VPCM, $30^{\circ}$S')
    ax[0,0].plot(vpcm_60N, plobject.heights, color='r', label='VPCM, $60^{\circ}$N')
    for ind, probe in enumerate(probelist):
            ax[0,0].plot(probe.bv, probe.data['ALT(KM)'].values, 
                    linestyle='dashed',
                    color=colors[ind], label=probe.name+', ' +probe.latstr)
    ax[0,0].legend()
    ax[0,0].set_xlabel('Frequency / s$^{-1}$', fontsize=fsize)
    ax[0,0].set_ylabel('Height / km', fontsize=fsize)

    ax[0,1].set_title('b) Constructed BV frequency profile', fontsize=fsize)
    ax[0,1].plot(vpcm_30S_constructed, plobject.heights, color='k', label='VPCM, $30^{\circ}$S')
    ax[0,1].plot(vpcm_60N_constructed, plobject.heights, color='r', label='VPCM, $60^{\circ}$N')
    for ind, probe in enumerate(probelist):
            ax[0,1].plot(probe.bv_profile, probe.data['ALT(KM)'].values, 
                    linestyle='dashed',
                    color=colors[ind], label=probe.name+', ' +probe.latstr)
    ax[0,1].legend()
    ax[0,1].set_xlabel('Frequency / s$^{-1}$', fontsize=fsize)
    ax[0,1].set_ylabel('Height / km', fontsize=fsize)

    for ind, probe in enumerate(probelist):
            probe.calc_rossby_radii(constructed=False)
            L_probe = probe.extra_r/(probe.radius*1000)
            lambda_probe = probe.trop_r/(probe.radius*1000)
            ax[1,0].plot(L_probe, probe.data['ALT(KM)'].values, 
                         linestyle='dashed',
                         color=colors[ind], label=probe.name+', ' +probe.latstr+', extratropical')
            ax[1,0].plot(lambda_probe, probe.data['ALT(KM)'].values, 
                         color=colors[ind], label=probe.name+', ' +probe.latstr+', tropical')
    ax[1,0].set_title('c) Meridional Rossby wavenumber based on a)')
    ax[1,0].plot(L_vpcm, plobject.heights, 
            color='r', linestyle='dashed', label='VPCM, $60^{\circ}$N, extratropical')
    ax[1,0].plot(lambda_vpcm, plobject.heights, 
            color='r', label='VPCM, $30^{\circ}$S, tropical')
    ax[1,0].plot(np.ones_like(lambda_vpcm), 
             plobject.heights,
             color='k', linestyle='dotted',
             label='Wavenumber=1')
    ax[1,0].set_ylabel('Height / km', fontsize=fsize)
    ax[1,0].set_xlabel('Rossby wavenumber', fontsize=fsize)
    ax[1,0].set_xscale('log')
    ax[1,0].legend()

    for ind, probe in enumerate(probelist):
            probe.calc_rossby_radii(constructed=True)
            L_probe = probe.extra_r_constructed/(probe.radius*1000)
            lambda_probe = probe.trop_r_constructed/(probe.radius*1000)
            ax[1,1].plot(L_probe, probe.data['ALT(KM)'].values, 
                         linestyle='dashed',
                         color=colors[ind], label=probe.name+', ' +probe.latstr+', extratropical')
            ax[1,1].plot(lambda_probe, probe.data['ALT(KM)'].values, 
                         color=colors[ind], label=probe.name+', ' +probe.latstr+', tropical')
    ax[1,1].set_title('d) Meridional Rossby wavenumber based on b)')
    ax[1,1].plot(L_vpcm_constructed, plobject.heights, 
            color='r', linestyle='dashed', label='VPCM, $60^{\circ}$N, extratropical')
    ax[1,1].plot(lambda_vpcm_constructed, plobject.heights, 
            color='r', label='VPCM, $30^{\circ}$S, tropical')
    ax[1,1].plot(np.ones_like(lambda_vpcm), 
             plobject.heights,
             color='k', linestyle='dotted',
             label='Wavenumber=1')
    ax[1,1].set_ylabel('Height / km', fontsize=fsize)
    ax[1,1].set_xlabel('Rossby wavenumber', fontsize=fsize)
    ax[1,1].set_xscale('log')
    ax[1,1].legend()
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def sensitivity_lat(plobject, fsize=14, savearg=False, 
                   savename='appendix_fig2_lat.png', sformat='png'):
     
    L_vpcm_0N = extratropical(plobject, gmean=False, lat=48, constructed=True)/(plobject.radius*1000)
    L_vpcm_15N = extratropical(plobject, gmean=False, lat=56, constructed=True)/(plobject.radius*1000)
    L_vpcm_30N = extratropical(plobject, gmean=False, lat=64, constructed=True)/(plobject.radius*1000)
    L_vpcm_45N = extratropical(plobject, gmean=False, lat=72, constructed=True)/(plobject.radius*1000)
    L_vpcm_60N = extratropical(plobject, gmean=False, lat=80, constructed=True)/(plobject.radius*1000)
    L_vpcm_75N = extratropical(plobject, gmean=False, lat=88, constructed=True)/(plobject.radius*1000)
    L_vpcm_84N = extratropical(plobject, gmean=False, lat=93, constructed=True)/(plobject.radius*1000)
    L_vpcm_glob = extratropical(plobject, gmean=True, constructed=True)/(plobject.radius*1000)
    
    lambda_vpcm_0N = tropical(plobject, gmean=False, lat=48, constructed=True)/(plobject.radius*1000)
    lambda_vpcm_15N = tropical(plobject, gmean=False, lat=56, constructed=True)/(plobject.radius*1000)
    lambda_vpcm_30N = tropical(plobject, gmean=False, lat=64, constructed=True)/(plobject.radius*1000)
    lambda_vpcm_45N = tropical(plobject, gmean=False, lat=72, constructed=True)/(plobject.radius*1000)
    lambda_vpcm_60N = tropical(plobject, gmean=False, lat=80, constructed=True)/(plobject.radius*1000)
    lambda_vpcm_75N = tropical(plobject, gmean=False, lat=88, constructed=True)/(plobject.radius*1000)
    lambda_vpcm_84N = tropical(plobject, gmean=False, lat=93, constructed=True)/(plobject.radius*1000)
    lambda_vpcm_glob = tropical(plobject, gmean=True, constructed=True)/(plobject.radius*1000)

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12,6))
    fig.suptitle(r'Sensitivity of Rossby wavenumber to latitude',
                 y=1.01, fontsize=fsize+6)
    ax[1].set_title('b) Extratropical')
    ax[1].plot(L_vpcm_0N, plobject.heights, color='tab:blue', label='Equator')
    ax[1].plot(L_vpcm_15N, plobject.heights, color='tab:pink', label='$15^{\circ}$N')
    ax[1].plot(L_vpcm_30N, plobject.heights, color='tab:orange', label='$30^{\circ}$N')
    ax[1].plot(L_vpcm_45N, plobject.heights, color='tab:green', label='$45^{\circ}$N')
    ax[1].plot(L_vpcm_60N, plobject.heights, color='tab:red', linewidth=1.5, label='$60^{\circ}$N')
    ax[1].plot(L_vpcm_75N, plobject.heights, color='tab:gray', label='$75^{\circ}$N')
    ax[1].plot(L_vpcm_84N, plobject.heights, color='tab:purple', label='$84^{\circ}$N')
    ax[1].plot(L_vpcm_glob, plobject.heights, color='k', linestyle='dashed', linewidth=1.5, label='Global mean')   
    ax[1].plot(np.ones_like(L_vpcm_60N), plobject.heights,
             color='k', linestyle='dotted',
             label='Wavenumber=1')
    ax[1].set_ylabel('Height / km', fontsize=fsize)
    ax[1].set_xlabel('Rossby wavenumber', fontsize=fsize)
    ax[1].set_xscale('log')
    ax[1].legend()

    ax[0].set_title('a) Tropical')
    ax[0].plot(lambda_vpcm_0N, plobject.heights, color='tab:blue', label='Equator')
    ax[0].plot(lambda_vpcm_15N, plobject.heights, color='tab:pink', label='$15^{\circ}$N')
    ax[0].plot(lambda_vpcm_30N, plobject.heights, color='tab:orange', label='$30^{\circ}$N')
    ax[0].plot(lambda_vpcm_45N, plobject.heights, color='tab:green', label='$45^{\circ}$N')
    ax[0].plot(lambda_vpcm_60N, plobject.heights, color='tab:red', linewidth=1.5, label='$60^{\circ}$N')
    ax[0].plot(lambda_vpcm_75N, plobject.heights, color='tab:gray', label='$75^{\circ}$N')
    ax[0].plot(lambda_vpcm_84N, plobject.heights, color='tab:purple', label='$84^{\circ}$N')
    ax[0].plot(lambda_vpcm_glob, plobject.heights, color='k', linestyle='dashed', linewidth=1.5, label='Global mean')   
    ax[0].plot(np.ones_like(lambda_vpcm_60N), plobject.heights,
             color='k', linestyle='dotted',
             label='Wavenumber=1')
    ax[0].set_ylabel('Height / km', fontsize=fsize)
    ax[0].set_xlabel('Rossby wavenumber', fontsize=fsize)
    ax[0].set_xscale('log')
    ax[0].legend()
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    


# %%
if __name__ == "__main__":

    surface = init_model_data(surfacepath, 'vpcm', 'surface')
    cloud = init_model_data(cloudpath, 'vpcm', 'cloud')
    # Instantiate Venus PCM model data object
    pv_probes = init_probes([day_probe, night_probe, north_probe])
    # Instantiate Probe objects for Pioneer Venus data
    # And colllect in a list

    compare_profiles(surface, pv_probes, fsize=14,
                     savearg=False, savename='fig2_profiles.png',
                     sformat='png')
    
    compare_rossby(surface, pv_probes, trop_lat=32, extra_lat=80, 
                   trop_gmean=False, extra_gmean=False,
                   trange=(0,-1), fsize=14, savearg=False,
                   savename='fig3_radii.png', sformat='png')
    
    allwaves(cloud, savearg=False, hmin=0, hmax=-1, savename='fig4b_temp_anomaly.png',
             sformat='png')
    
    ## Appendix figures: sensitivity tests
    
    sensitivity_bv(surface, pv_probes, fsize=14, savearg=False, 
                   savename='appendix_fig1_bv.png', sformat='png')
    
    sensitivity_lat(surface, fsize=14, savearg=False, 
                   savename='appendix_fig2_lat.png', sformat='png')