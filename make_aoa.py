""" Uses functions from venuslab to auto-generate plots for
    'Tracer transport in the Venus atmosphere revealed by mean age of air',
     Cohen et al. 2024                                       """

""" Usage from command line: python make_aoa.py
    No inputs required as filepaths are specified in script """

# Input file paths and outputs - only part of script that should be edited
# %%
surfacepath = '/exomars/data/analysis/volume_8/mc5526/aoa_surface.nc'
# Simulation with surface age of air tracer
cloudpath = '/exomars/data/analysis/volume_9/mc5526/lmd_data/aoa_cloud.nc'
# Simulation with cloud deck age of air tracer
allpaths = [surfacepath, cloudpath]
# List of paths
balloon1 = '/exomars/data/analysis/volume_8/mc5526/vega_data/vg1bl_rdr.dat'
# Vega balloon 1 data
balloon2 = '/exomars/data/analysis/volume_8/mc5526/vega_data/vg2bl_rdr.dat'
# Vega balloon 2 data
allbpaths = [balloon1, balloon2]
# List of balloon data paths
outpath = '/exomars/data/analysis/volume_8/mc5526/make_aoa/revised_submission/'
# Where to send the plots
saveas = 'eps'
# What format to save the plots - png for viewing, eps for paper

# Import packages
# %%
from venusdata import *
from venusaoa import *
from venusrossby import *
from venusvega import *
from venusdyn import psi_m, zmzw, time_series
from venuspoles import add_circle_boundary, add_cycl_point
from venusspectral import timeseries_transform, bandpass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.integrate import cumtrapz
import netCDF4 as nc
import cartopy.crs as ccrs
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# %%
def init_data(pathlist, names):
    """ Instantiates a Venus Planet object for each 
        dataset in a list of filepaths          """

    plobjects = []
    for path in pathlist:
        ind = pathlist.index(path)
        plobject = Planet(venusdict, 'vpcm', names[ind])
        plobject.load_file(path)
        plobject.setup()
        plobjects.append(plobject)

    return plobjects

# %%
def init_balloons(bpathlist):
    """ Instantiates a Balloon object for each of the
        Vega balloon datasets """
    
    bobjects = []
    for ind, path in enumerate(bpathlist):
        bname = 'Vega balloon ' + str(ind+1)
        bobject = Balloon(path,bname)
        bobject.coords()
        bobjects.append(bobject)

    return bobjects

# %%
def baseline_atmosphere(plobject, trange=(-10,-1), fsize=14, 
                        savearg=False, savename='fig1_baseline.png', 
                        sformat='png'):
    """ Two-plot figure containing zonal mean wind +
        mean meridional mass stream function

        Leverages calculations from venusdyn module"""
    
    zonal_mean_wind = zmzw(plobject, trange=trange, plot=False)
    mass_streamfunction = psi_m(plobject, trange=trange, plot=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,6))
    cf = ax1.contourf(plobject.lats, plobject.heights, 
                 -zonal_mean_wind, extend='both',
                 levels=np.arange(-120,40,10),
                 cmap='RdBu', norm=TwoSlopeNorm(0))
    ax1.plot(plobject.lats, np.ones_like(plobject.lats)*plobject.heights[22], 'g--', linewidth=2.5)
    ax1.plot(plobject.lats, np.ones_like(plobject.lats)*plobject.heights[1], 'm--', linewidth=2.5)
    ax1.set_title('a) Zonal mean zonal wind', fontsize=fsize)
    ax1.set_xlabel('Latitude / deg', fontsize=fsize)
    ax1.set_ylabel('Height / km', fontsize=fsize)
    cbar = plt.colorbar(cf, ax=ax1, orientation='vertical')
    cbar.ax.set_title('m/s')
    lat_ticks = ax1.get_xticks()

    cs = ax2.contour(plobject.lats, plobject.heights[:-1],
                mass_streamfunction, colors='black',
                levels=np.arange(-100,100,10))
    ax2.clabel(cs, cs.levels, inline=True)
    ax2.plot(plobject.lats, np.ones_like(plobject.lats)*plobject.heights[22], 'g--', linewidth=2.5)
    ax2.plot(plobject.lats, np.ones_like(plobject.lats)*plobject.heights[1], 'm--', linewidth=2.5)
    ax2.set_title('b) Mean meridional mass streamfunction / $10^{10}$ kg/s',
                  fontsize=fsize)
    ax2.set_xlabel('Latitude / deg', fontsize=fsize)
    ax2.set_xticks(lat_ticks[1:-1])

    plt.subplots_adjust(wspace=0.03)

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def zonal_mean_aoa(plobjects, time_slice=-2, fsize=14, 
                    savearg=False, savename='fig2_zonalmean_aoa.png', 
                    sformat='png'):
    """ Four-plot figure comparing zonal mean age of air in two simulations
        First row: all altitudes
        Second row: zoom in on altitudes of interest                """

    surface = np.mean(plobjects[0].data['age'][time_slice,:,:,:], axis=-1)/(60*60*24*360)
    cloud = np.mean(plobjects[1].data['age'][time_slice,:,:,:], axis=-1)/(60*60*24*360)
    mass_streamfunction_surface = psi_m(plobjects[0], trange=(0,-2), plot=False)

    fig, ax = plt.subplots(2, 2, figsize=(12,8), 
                           gridspec_kw={'height_ratios': [1,0.5],
                                        'width_ratios':  [1,1]},
                          sharex=True)
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    fig.suptitle('Zonal mean age of air', y=1.05, fontsize=fsize+6)

    surf_full = ax[0,0].contourf(plobjects[0].lats, plobjects[0].heights,
                                 surface, levels=np.arange(0,29,2), 
                                 cmap='cividis', extend='both')
    cs_surf = ax[0,0].contour(plobjects[0].lats, plobjects[0].heights[:-1],
            mass_streamfunction_surface, colors='black', alpha=0.5,
            levels=np.arange(-100,100,10))
    ax[0,0].clabel(cs_surf, cs_surf.levels, inline=True)
    ax[0,0].set_title('a) Surface tracer', fontsize=fsize)
    ax[0,0].set_ylabel('Height / km', fontsize=fsize)
    cbar1 = plt.colorbar(surf_full, ax=ax[0,0], orientation='vertical')
    cbar1.ax.set_title('years')

    cloud_full = ax[0,1].contourf(plobjects[1].lats, plobjects[1].heights,
                                  cloud, levels=np.arange(0,5.5,0.5), 
                                  cmap='cividis', extend='both')
    cs_cloud = ax[0,1].contour(plobjects[1].lats, plobjects[1].heights[:-1],
            mass_streamfunction_surface, colors='black', alpha=0.5,
            levels=np.arange(-100,100,10))
    ax[0,1].clabel(cs_cloud, cs_cloud.levels, inline=True)
    ax[0,1].set_title('b) Cloud deck tracer', fontsize=fsize)
    cbar2 = plt.colorbar(cloud_full, ax=ax[0,1], orientation='vertical')
    cbar2.ax.set_title('years')

    surf_zoom = ax[1,0].contourf(plobjects[0].lats, plobjects[0].heights[13:38],
                                 surface[13:38], levels=np.arange(16,29,1), 
                                 cmap='cividis', extend='both')
    cs_surf_zoom = ax[1,0].contour(plobjects[0].lats, plobjects[0].heights[13:38],
            mass_streamfunction_surface[13:38], colors='black', alpha=0.5,
            levels=np.arange(-100,100,10))
    ax[1,0].clabel(cs_surf_zoom, cs_surf_zoom.levels, inline=True)
    ax[1,0].set_title('c) Detail, 20-80 km', fontsize=fsize)
    ax[1,0].set_ylabel('Height / km', fontsize=fsize)
    ax[1,0].set_xlabel('Latitude / deg', fontsize=fsize)
    cbar3 = plt.colorbar(surf_zoom, ax=ax[1,0], orientation='vertical', aspect=10)
    cbar3.ax.set_title('years')

    cloud_zoom = ax[1,1].contourf(plobjects[1].lats, plobjects[1].heights[25:50],
                                  cloud[25:50], levels=np.arange(0,1.6,0.2), 
                                  cmap='cividis', extend='both')
    cs_surf_cloud = ax[1,1].contour(plobjects[1].lats, plobjects[1].heights[25:50],
            mass_streamfunction_surface[24:50], colors='black', alpha=0.5,
            levels=np.arange(-100,100,10))
    ax[1,1].clabel(cs_surf_cloud, cs_surf_cloud.levels, inline=True)
    ax[1,1].set_title('d) Detail, 50-100 km', fontsize=fsize)
    ax[1,1].set_xlabel('Latitude / deg', fontsize=fsize)
    cbar3 = plt.colorbar(cloud_zoom, ax=ax[1,1], orientation='vertical', aspect=10)
    cbar3.ax.set_title('years')

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def aoa_profiles(plobjects, coords=[0,48,96], 
                 time_slice=-2, fsize=14, 
                 savearg=False, savename='fig3_profiles_aoa.png', 
                 sformat='png'):
    """ Two-plot figure showing vertical profiles of age of air
        at equator, north pole, and south pole in two simulations """
    
    surface = np.mean(plobjects[0].data['age'][time_slice,:,:,:], axis=-1)/(60*60*24*360)
    cloud = np.mean(plobjects[1].data['age'][time_slice,:,:,:], axis=-1)/(60*60*24*360)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,8))
    fig.suptitle('Vertical profile of age of air', y=0.95, fontsize=fsize+4)
    for coord in coords:
        lat_lab = plobjects[0].lats[coord]
        ax1.plot(surface[:,coord], plobjects[0].heights,
                 label=f'{lat_lab}$^\circ$ lat')
    ax1.set_title('a) Surface tracer', fontsize=fsize)
    ax1.set_ylabel('Height / km', fontsize=fsize)
    ax1.set_xlabel('Age / years', fontsize=fsize)
    ax1.grid()
    ax1.legend()

    for coord in coords:
        lat_lab = plobjects[1].lats[coord]
        ax2.plot(cloud[:,coord], plobjects[1].heights,
                 label=f'{lat_lab}$^\circ$ lat')
    ax2.set_title('b) Cloud deck tracer', fontsize=fsize)
    ax2.set_xlabel('Age / years', fontsize=fsize)
    ax2.grid()
    ax2.legend()

    plt.subplots_adjust(wspace=0.1)
    
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def polar_aoa(plobject, trange=np.arange(1856,1862,1), lev=25,
              clevs=np.arange(24.5,27.0,0.2),
              fsize=14, savearg=False, savename='fig4_polar_aoa.png', 
              sformat='png'):
    """ Four-plot figure showing snapshots of the south polar
        age of air across one Venus day                 
        Surface simulation only!                        """
    
    surface = plobject.data['age'][trange[0]:trange[-1]+1,lev,:,:]/(60*60*24*360)
    # Extract a range of data spanning one Venus day
    timedict = {'1':'a) Day 0', '2':'b) Day 6', '3':'c) Day 12', 
                '4':'d) Day 18', '5':'e) Day 24', '6': 'f) Day 30'}
    fig, ax = plt.subplots(3, 2, figsize=(4,6),
                            subplot_kw={'projection': ccrs.Orthographic(0,-90)},
                            gridspec_kw={'wspace':0, 'top':1., 'bottom':0., 'left':0., 'right':1.})
    # Create 2x2 figure with each plot an orthographic proj centered at south pole

    snapshots = []
    for t in [0,1,2,3,4,5]:
        snapshot = surface[t,:,:]
        snapshots.append(snapshot)
    # Extract every 5th time slice
    # Twenty slices is one Venus day, so we are time-stepping by 1/4 day
    counter = 0
    for snap, axitem in zip(snapshots, ax.flatten()):
        counter = counter + 1
        # Iterate through 1/4 day snapshots and axis objects at the same time
        axitem.set_global()
        axitem.gridlines(linewidth=0.5)
        axitem.set_extent([0, 360, -90, -60], crs=ccrs.PlateCarree())
        add_circle_boundary(axitem)
        snap, clon = add_cycl_point(snap, plobject.lons, -1)
        polarsnap = axitem.contourf(clon, plobject.lats,
                                    snap, cmap='cividis',
                                    levels=clevs,
#                                    extend='both', 
                                    transform=ccrs.PlateCarree())
        axitem.set_title(timedict[str(counter)])

    cbar_ax = fig.add_axes([0.15, -0.05, 0.75, 0.02])
    cbar = fig.colorbar(polarsnap, cax=cbar_ax, orientation='horizontal',
                        extend='both')
    cbar.set_label(label='years', fontsize=fsize-2)
    fig.suptitle(f'Age of air at south pole, \n h={np.round(plobject.heights[lev],0)} km', fontsize=fsize+2, y=1.125)
    plt.subplots_adjust(wspace=None, hspace=0.18)

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def aoa_slices(plobject, times=[1856,1858], levs=[15,22],
               fsize=14,
               savearg=False, savename='fig5_aoa_slices.png', 
               sformat='png'):
    """ Figure with four subplots showing two timesnaps for two different
        levels with horizontal wind vectors and vertical wind as contourfill"""
    
    surface = plobject.data['age'][times[0]:times[1]+1,levs[0]:levs[1]+1,:,:]/(60*60*24*360)
    geop_mean = np.mean(plobject.data['geop'], axis=0)
    geop = plobject.data['geop'][times[0]:times[1]+1,levs[0]:levs[1]+1,:,:] - geop_mean

    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)

    cf1 = ax[0,0].contourf(plobject.lons, plobject.lats, surface[0,0,:,:],
                     levels=np.arange(16.5,22.5,0.25), 
                     extend='both', cmap='cividis')
    cs1 = ax[0,0].contour(plobject.lons, plobject.lats, geop[0,0,:,:], colors='black', alpha=0.5)
    ax[0,0].clabel(cs1, cs1.levels, inline=True)
    ax[0,0].set_ylabel('Latitude / deg', fontsize=fsize)
    ax[0,0].set_title(f'a) h={np.round(plobject.heights[levs[0]],0)} km, day 0', fontsize=fsize)
    cbar1 = plt.colorbar(cf1, orientation='vertical', fraction=0.05)
    cbar1.set_label(f'years', loc='center')

    cf2 = ax[0,1].contourf(plobject.lons, plobject.lats,surface[-1,0,:,:],
                     levels=np.arange(16.5,22.5,0.25), 
                     extend='both', cmap='cividis')
    cs2 = ax[0,1].contour(plobject.lons, plobject.lats, geop[-1,0,:,:], colors='black', alpha=0.5)
    ax[0,1].clabel(cs2, cs2.levels, inline=True)
    ax[0,1].set_title(f'b) h={np.round(plobject.heights[levs[0]],0)} km, day 12', fontsize=fsize)
    cbar2 = plt.colorbar(cf2, orientation='vertical', fraction=0.05)
    cbar2.set_label(f'years', loc='center')

    cf3 = ax[1,0].contourf(plobject.lons, plobject.lats, surface[0,-1,:,:],
                     levels=np.arange(23.5,27,0.25), 
                     extend='both', cmap='cividis')
    cs3 = ax[1,0].contour(plobject.lons, plobject.lats, geop[0,-1,:,:], colors='black', alpha=0.5)
    ax[1,0].clabel(cs3, cs3.levels, inline=True)
    cbar3 = plt.colorbar(cf3, orientation='vertical', fraction=0.05)
    cbar3.set_label(f'years', loc='center')
    ax[1,0].set_ylabel('Latitude / deg', fontsize=fsize)
    ax[1,0].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[1,0].set_title(f'c) h={np.round(plobject.heights[levs[-1]],0)} km, day 0', fontsize=fsize)
    
    cf4 = ax[1,1].contourf(plobject.lons, plobject.lats, surface[-1,-1,:,:],
                     levels=np.arange(23.5,27,0.25), 
                     extend='both', cmap='cividis')
    cs4 = ax[1,1].contour(plobject.lons, plobject.lats, geop[-1,-1,:,:], colors='black', alpha=0.5)
    ax[1,1].clabel(cs4, cs4.levels, inline=True)
    cbar4 = plt.colorbar(cf4, orientation='vertical', fraction=0.05)
    cbar4.set_label(f'years', loc='center')
    ax[1,1].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[1,1].set_title(f'd) h={np.round(plobject.heights[levs[-1]],0)} km, day 12', fontsize=fsize)

    fig.suptitle('Age of air and geopotential height anomaly in the deep atmosphere', fontsize=fsize+2, y=0.95)

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def wind_composites(plobject, times=[1856,1858], levs=[15,22],
               fsize=14, qscale=1, n=4, clevs=np.arange(-0.04,0.05,0.01),
               savearg=False, savename='fig6_wind_composites.png', 
               sformat='png'):
    """ Figure with four subplots showing two timesnaps for two different
        levels with horizontal wind vectors and vertical wind as contourfill"""
    
    u = plobject.data['vitu'][times[0]:times[1]+1,levs[0]:levs[1]+1,:,:]
    v = plobject.data['vitv'][times[0]:times[1]+1,levs[0]:levs[1]+1,:,:]
    omega = plobject.data['vitw'][times[0]:times[1]+1,levs[0]:levs[1]+1,:,:]
    temp = plobject.data['temp'][times[0]:times[1]+1,levs[0]:levs[1]+1,:,:]
    pres = plobject.data['pres'][times[0]:times[1]+1,levs[0]:levs[1]+1,:,:]
    w = -(omega*temp*plobject.RCO2)/(pres*plobject.g)

    X, Y = np.meshgrid(plobject.lons, plobject.lats)
    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)

    cf1 = ax[0,0].contourf(plobject.lons, plobject.lats, w[0,0,:,:],
                     levels=clevs, cmap='coolwarm', extend='both', norm=TwoSlopeNorm(0))
    q1 = ax[0,0].quiver(X[::n, ::n], Y[::n, ::n], -u[0,0,::n,::n],
                   v[0,0,::n,::n], angles='xy', scale_units='xy', scale=qscale)
    ax[0,0].quiverkey(q1, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    ax[0,0].set_ylabel('Latitude / deg', fontsize=fsize)
    ax[0,0].set_title(f'a) h={np.round(plobject.heights[levs[0]],0)} km, day 0', fontsize=fsize)
    
    cf2 = ax[0,1].contourf(plobject.lons, plobject.lats, w[-1,0,:,:],
                     levels=clevs, cmap='coolwarm', extend='both', norm=TwoSlopeNorm(0))
    q2 = ax[0,1].quiver(X[::n, ::n], Y[::n, ::n], -u[-1,0,::n,::n],
                   v[-1,0,::n,::n], angles='xy', scale_units='xy', scale=qscale)
    ax[0,1].quiverkey(q2, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    ax[0,1].set_title(f'b) h={np.round(plobject.heights[levs[0]],0)} km, day 12', fontsize=fsize)
    
    cf3 = ax[1,0].contourf(plobject.lons, plobject.lats, w[0,-1,:,:],
                     levels=clevs, cmap='coolwarm', extend='both', norm=TwoSlopeNorm(0))
    q3 = ax[1,0].quiver(X[::n, ::n], Y[::n, ::n], -u[0,-1,::n,::n],
                   v[0,-1,::n,::n], angles='xy', scale_units='xy', scale=qscale)
    ax[1,0].quiverkey(q3, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    ax[1,0].set_ylabel('Latitude / deg', fontsize=fsize)
    ax[1,0].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[1,0].set_title(f'c) h={np.round(plobject.heights[levs[-1]],0)} km, day 0', fontsize=fsize)
    
    cf4 = ax[1,1].contourf(plobject.lons, plobject.lats, w[-1,-1,:,:],
                     levels=clevs, cmap='coolwarm', extend='both', norm=TwoSlopeNorm(0))
    q4 = ax[1,1].quiver(X[::n, ::n], Y[::n, ::n], -u[-1,-1,::n,::n],
                   v[-1,-1,::n,::n], angles='xy', scale_units='xy', scale=qscale)
    ax[1,1].quiverkey(q4, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    ax[1,1].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[1,1].set_title(f'd) h={np.round(plobject.heights[levs[-1]],0)} km, day 12', fontsize=fsize)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf4, cax=cbar_ax)
    cbar.set_label('Vertical wind / m/s', loc='center')
    fig.suptitle('General circulation of the deep atmosphere', fontsize=fsize+2, y=0.95)

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def vega_series(bobject1, bobject2, plobject, fsize=14,
                savearg=False, savename='fig8_vega_series.png',
                sformat='png'):
    """ Figure with 3 subplots showing:
        a) Vega balloon 1 vertical winds and background level
        b) Vega balloon 2 vertical winds and background level 
        c) Surface simulation vertical winds for 2/3 nightside 1/3 daysid
           selection, and background level
           
        Background wind levels are determined by filtering out all frequencies"""
    
    vega1_w = bobject1.data['W_b'].values
    freq1 = bobject1.data['Time'][1] - bobject1.data['Time'][0]
    vega1_background = bandpass(vega1_w, frequnit=freq1, plot=False)

    vega2_w = bobject2.data['W_b'].values
    freq2 = bobject2.data['Time'][1] - bobject1.data['Time'][0]
    vega2_background = bandpass(vega2_w, frequnit=freq2, plot=False)

    if not hasattr(plobject, 'w_wind'):
        plobject.calc_w()

    vpcm_w = np.mean(plobject.w_wind[:,25,52,:],axis=0)
    # Mean VPCM vertical wind in m/s at h=54 km, lat=8 deg (position of Vega balloon 1)
    freq3 = 1./len(plobject.lons)
    vpcm_background = bandpass(vpcm_w.values, frequnit=freq3, plot=False)
   
    fig, ax = plt.subplots(3,1, figsize=(8,12))
    ax[0].plot(bobject1.data['Hours'], vega1_w, color='k', linestyle='dashed')
    ax[0].plot(bobject1.data['Hours'], vega1_background, color='r')
    ax[0].text(0.4,0.85,f'Background wind = {np.round(vega1_background[0],3)} m/s', color='r', 
               fontsize=fsize, transform=ax[0].transAxes)
    ax[0].set_xlabel('Time / hours', fontsize=fsize)
    ax[0].set_ylabel('Vertical wind / m/s', fontsize=fsize)
    ax[0].set_ylim(-2,3)
    ax[0].set_title('a) Vega 1 balloon, 8 deg north, 54 km altitude', fontsize=fsize)

    ax[1].plot(bobject2.data['Hours'], vega2_w, color='k', linestyle='dashed')
    ax[1].plot(bobject2.data['Hours'], vega2_background, color='r')
    ax[1].text(0.4,0.85,f'Background wind = {np.round(vega2_background[0],3)} m/s', color='r', 
               fontsize=fsize, transform=ax[1].transAxes)
    ax[1].set_xlabel('Time / hours', fontsize=fsize)
    ax[1].set_ylabel('Vertical wind / m/s', fontsize=fsize)
    ax[1].set_ylim(-2,3)
    ax[1].set_title('b) Vega 2 balloon, 6 deg south, 54 km altitude', fontsize=fsize)

    ax[2].plot(plobject.lons, vpcm_w, color='k')
    ax[2].plot(plobject.lons, vpcm_background, color='r')
    ax[2].text(0.4,0.85,f'Background wind = {vpcm_background[0]:.3f} m/s', color='r', 
               fontsize=fsize, transform=ax[2].transAxes)
    ax[2].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[2].set_ylabel('Vertical wind / m/s', fontsize=fsize)
    ax[2].set_ylim(0.,0.010)
    ax[2].set_title('c) Venus PCM model output, 8 deg north, 54 km altitude', fontsize=fsize)

    fig.suptitle('Vertical winds measured by Vega compared to Venus PCM', y=0.9375, fontsize=fsize+2)
    plt.subplots_adjust(wspace=None, hspace=0.375)

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def convergence(plobjects, cloud_lev=49, surf_lev=8, surf_lev2=18, surf_lev3=22, 
                lat=48, fsize=14, savearg=False, savename='convergence.png', sformat='png'):
    """ 6 panel figure illustrating convergence of Cloud sim in top 3 panels
        and progress towards convergence of Surface sim in bottom 3 panels"""

    surface = np.mean(plobjects[0].data['age'][:-1,surf_lev,lat,:], axis=-1)/(60*60*24*360)
    surface2 = np.mean(plobjects[0].data['age'][:-1,surf_lev2,lat,:], axis=-1)/(60*60*24*360)
    surface3 = np.mean(plobjects[0].data['age'][:-1,surf_lev3,lat,:], axis=-1)/(60*60*24*360)
    cloud = np.mean(plobjects[1].data['age'][:,cloud_lev,lat,:], axis=-1)/(60*60*24*360)

    surf_filtered = savgol_filter(surface, 500, 1)
    surf_filtered2 = savgol_filter(surface2, 500, 1)
    surf_filtered3 = savgol_filter(surface3, 500, 1)
    cloud_filtered = savgol_filter(cloud, 500, 1)
    surf_grad = savgol_filter(np.gradient(surf_filtered), 500, 1)
    surf_grad2 = savgol_filter(np.gradient(surf_filtered2), 500, 1)
    surf_grad3 = savgol_filter(np.gradient(surf_filtered3), 500, 1)
    cloud_grad = savgol_filter(np.gradient(cloud_filtered), 500, 1)
    surf_axis = np.arange(0, surf_filtered.shape[0])*plobjects[0].tinterval/(60*60*24*360)
    cloud_axis = np.arange(0, cloud_filtered.shape[0])*plobjects[1].tinterval/(60*60*24*360)

    surf_popt, surf_pcov = curve_fit(ageline_fit, surf_axis, surf_filtered)
    surf_popt2, surf_pcov2 = curve_fit(ageline_fit, surf_axis, surf_filtered2)
    surf_popt3, surf_pcov3 = curve_fit(ageline_fit, surf_axis, surf_filtered3)
    cloud_popt, cloud_pcov = curve_fit(ageline_fit, cloud_axis, cloud_filtered)

    leg_loc = 'lower right'

    fig, ax = plt.subplots(4, 3, figsize=(15, 9))
    ax[0,0].plot(surf_axis, surface, color='b', label='Raw data')
    ax[0,0].plot(surf_axis, surf_filtered, color='r', label='Smoothed data')
    ax[0,0].set_title(f'a) Surface simulation, h={plobjects[0].heights[surf_lev]} km',
                    fontsize=fsize)
    ax[0,0].set_ylabel('Age of air / years')
    ax[0,0].legend(fontsize=10, loc=leg_loc)

    ax[0,1].plot(surf_axis, surf_grad, color='g')
    ax[0,1].plot(surf_axis, np.zeros_like(surf_axis), color='r', linestyle='dashed')
    ax[0,1].set_title('b) Rate of change of age of air', fontsize=fsize)
    ax[0,1].set_ylabel('Change per year')

    ax[0,2].plot(surf_axis, surf_filtered, color='r', label='Smoothed data')
    ax[0,2].plot(surf_axis, ageline_fit(surf_axis, *surf_popt), 'k--',
                label=r'Fit: a=%5.3f, $\tau$=%5.3f' % tuple(surf_popt))
    ax[0,2].set_title(r'c) Data fit to $a(1 - e^{-t/\tau})$', fontsize=fsize)
    ax[0,2].set_ylabel('Age of air / years')
    ax[0,2].legend(fontsize=10, loc=leg_loc)

    ax[1,0].plot(surf_axis, surface2, color='b', label='Raw data')
    ax[1,0].plot(surf_axis, surf_filtered2, color='r', label='Smoothed data')
    ax[1,0].set_title(f'd) Surface simulation, h={plobjects[0].heights[surf_lev2]} km',
            fontsize=fsize)
    ax[1,0].set_ylabel('Age of air / years')
    ax[1,0].legend(fontsize=10, loc=leg_loc)

    ax[1,1].plot(surf_axis, surf_grad2, color='g')
    ax[1,1].plot(surf_axis, np.zeros_like(surf_axis), color='r', linestyle='dashed')
    ax[1,1].set_title('e)', fontsize=fsize)
    ax[1,1].set_ylabel('Change per year')

    ax[1,2].plot(surf_axis, surf_filtered2, color='r', label='Smoothed data')
    ax[1,2].plot(surf_axis, ageline_fit(surf_axis, *surf_popt2), 'k--',
                label=r'Fit: a=%5.3f, $\tau$=%5.3f' % tuple(surf_popt2))
    ax[1,2].set_title('f)', fontsize=fsize)
    ax[1,2].set_ylabel('Age of air / years')
    ax[1,2].legend(fontsize=10, loc=leg_loc)

    ax[2,0].plot(surf_axis, surface3, color='b', label='Raw data')
    ax[2,0].plot(surf_axis, surf_filtered3, color='r', label='Smoothed data')
    ax[2,0].set_title(f'g) Surface simulation, h={plobjects[0].heights[surf_lev3]} km',
            fontsize=fsize)
    ax[2,0].set_ylabel('Age of air / years')
    ax[2,0].legend(fontsize=10, loc=leg_loc)

    ax[2,1].plot(surf_axis, surf_grad3, color='g')
    ax[2,1].plot(surf_axis, np.zeros_like(surf_axis), color='r', linestyle='dashed')
    ax[2,1].set_title('h)', fontsize=fsize)
    ax[2,1].set_ylabel('Change per year')

    ax[2,2].plot(surf_axis, surf_filtered3, color='r', label='Smoothed data')
    ax[2,2].plot(surf_axis, ageline_fit(surf_axis, *surf_popt3), 'k--',
                label=r'Fit: a=%5.3f, $\tau$=%5.3f' % tuple(surf_popt3))
    ax[2,2].set_title('i)', fontsize=fsize)
    ax[2,2].set_ylabel('Age of air / years')
    ax[2,2].legend(fontsize=10, loc=leg_loc)

    ax[3,0].plot(cloud_axis, cloud, color='b', label='Raw data')
    ax[3,0].plot(cloud_axis, cloud_filtered, color='r', label='Smoothed data')
    ax[3,0].set_title(f'j) Cloud simulation, h={plobjects[1].heights[cloud_lev]} km',
                    fontsize=fsize)
    ax[3,0].set_xlabel('Time / years')
    ax[3,0].set_ylabel('Age of air / years')
    ax[3,0].legend(fontsize=10, loc=leg_loc)
    ax[3,0].set_yticks([0.0, 0.5, 1.0, 1.5])

    ax[3,1].plot(cloud_axis, cloud_grad, color='g')
    ax[3,1].plot(cloud_axis, np.zeros_like(cloud_axis), color='r', linestyle='dashed')
    ax[3,1].set_title('k)', fontsize=fsize)
    ax[3,1].set_xlabel('Time / years')
    ax[3,1].set_ylabel('Change per year')

    ax[3,2].plot(cloud_axis, cloud_filtered, color='r', label='Smoothed data')
    ax[3,2].plot(cloud_axis, ageline_fit(cloud_axis, *cloud_popt), 'k--',
                label=r'Fit: a=%5.3f, $\tau$=%5.3f' % tuple(cloud_popt))
    ax[3,2].set_title('l)', fontsize=fsize)
    ax[3,2].set_xlabel('Time / years')
    ax[3,2].set_ylabel('Age of air / years')
    ax[3,2].legend(fontsize=10, loc=leg_loc)
    ax[3,2].set_yticks([0.0, 0.5, 1.0, 1.5])

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.align_ylabels(ax)

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def zonal_mean_gradients(plobject):
    """ Make zonal mean plot showing age of air gradient of each gridbox for final output
        In other words, how to close to zero gradient (=convergence) is each gridbox? """

    surface = np.mean(plobject.data['age'][:-2,:,:,:], axis=-1)/(60*60*24*360)
    surface_smoothed = savgol_filter(surface, 500, 1, axis=0)
    surface_grad = np.gradient(surface_smoothed, axis=0)

    fig, ax = plt.subplots(figsize=(6,6))
    cf = ax.contourf(plobject.lats, plobject.heights, surface_grad[-1,:,:], 
                    levels=np.arange(0.0,0.013,0.001),
                    cmap='cividis')
    ax.set_title('Zonal mean gradient of age of air')
    ax.set_xlabel('Latitude / deg')
    ax.set_ylabel('Height / km')
    cbar = plt.colorbar(cf, orientation='vertical')
    cbar.set_label('Change per year')
    plt.show()


# %%
if __name__ == "__main__":

    simulations = init_data(allpaths, names=['surface', 'cloud'])
    # Instantiate data objects used in plots
    # List item 1 is Surface, List item 2 is Cloud
    balloons = init_balloons(allbpaths)
    # Instantiate data objects for Vega balloons
    # List item 1 is balloon 1, list item 2 is balloon 2

    # Figure 1
    baseline_atmosphere(simulations[0], trange=(0,-1), savearg=True,
                        sformat=saveas, savename=outpath+'fig1_baseline.'+saveas)
    
    # Figure 2
    convergence(simulations, cloud_lev=49, surf_lev=8, surf_lev2=18, surf_lev3=22,
                lat=48, fsize=14, savearg=True, savename=outpath+'fig2_convergence.'+saveas, 
                sformat=saveas)
    # Figure 3
    zonal_mean_aoa(simulations, savearg=True, sformat=saveas,
                   savename=outpath+'fig3_zonalmean_aoa.'+saveas)
    # Figure 4
    aoa_profiles(simulations, savearg=True, sformat=saveas,
                 savename=outpath+'fig4_profiles_aoa.'+saveas)
    # Figure 5
    polar_aoa(simulations[0], savearg=True, sformat=saveas,
              savename=outpath+'fig5_polar_aoa.'+saveas)
    # Figure 6
    aoa_slices(simulations[0], savearg=True, sformat=saveas,
               savename=outpath+'fig6_aoa_slices.'+saveas)
    # Figure 7
    wind_composites(simulations[0], savearg=True, sformat=saveas,
                    savename=outpath+'fig7_wind_composites.'+saveas)
    # Figure 8
    time_series(simulations[0], key='vitw',
                coords=[(16,86,48),(22,86,48),(30,86,48)], 
                ptitle='vertical wind', ylab='Wind velocity', 
                unit='m/s', plot=True, trange=[1777,1876], 
                tunit='Venus days', savename=outpath+'fig8_wtimeseries.'+saveas,
                fsize=14, save=True, saveformat=saveas)
    # Figure 9
    timeseries_transform(simulations[0],key='vitw', fsize=14, plot_transform=True,
                         coords=[(16,86,48),(22,86,48),(30,86,48)],
                         trange=[1777,1876], save=True, saveformat=saveas,
                         savename=outpath+'fig9_wfourier_transform.'+saveas)
    # Figure 10
    vega_series(balloons[0], balloons[1], simulations[0], savearg=True,
                savename=outpath+'fig10_vega_series.'+saveas, sformat=saveas)


# %%
