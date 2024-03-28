""" Uses functions from venuslab to auto-generate plots for
    'The 3-D structure of the Venus atmosphere revealed by mean age of air',
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
outpath = '/exomars/data/analysis/volume_8/mc5526/make_aoa/'
# Where to send the plots
saveas = 'png'
# What format to save the plots - png for viewing, eps for paper

# Import packages
# %%
from venusdata import *
from venusaoa import *
from venusrossby import *
from venusdyn import psi_m, zmzw, time_series
from venuspoles import add_circle_boundary, add_cycl_point

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.integrate import cumtrapz
import netCDF4 as nc
import cartopy.crs as ccrs

# %%
def init_data(pathlist):
    """ Instantiates a Venus Planet object for each 
        dataset in a list of filepaths          """

    plobjects = []
    for path in pathlist:
        plobject = Planet(venusdict)
        plobject.load_file(path)
        plobject.setup()
        plobjects.append(plobject)

    return plobjects

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
    ax1.set_title('Zonal mean zonal wind', fontsize=fsize)
    ax1.set_xlabel('Latitude / deg', fontsize=fsize)
    ax1.set_ylabel('Height / km', fontsize=fsize)
    cbar = plt.colorbar(cf, ax=ax1, orientation='vertical')
    cbar.ax.set_title('m/s')
    lat_ticks = ax1.get_xticks()

    cs = ax2.contour(plobject.lats, plobject.heights[:-1],
                mass_streamfunction, colors='black',
                levels=np.arange(-100,100,10))
    ax2.clabel(cs, cs.levels, inline=True)
    ax2.set_title('Mean meridional mass streamfunction / $10^{10}$ kg/s',
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

    fig, ax = plt.subplots(2, 2, figsize=(12,8), 
                           gridspec_kw={'height_ratios': [1,0.5],
                                        'width_ratios':  [1,1]},
                          sharex=True)
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    fig.suptitle('Zonal mean age of air', y=1.05, fontsize=fsize+6)

    surf_full = ax[0,0].contourf(plobjects[0].lats, plobjects[0].heights,
                                 surface, levels=np.arange(0,29,2), 
                                 cmap='cividis', extend='both')
    ax[0,0].set_title('Surface tracer', fontsize=fsize)
    ax[0,0].set_ylabel('Height / km', fontsize=fsize)
    cbar1 = plt.colorbar(surf_full, ax=ax[0,0], orientation='vertical')
    cbar1.ax.set_title('years')

    cloud_full = ax[0,1].contourf(plobjects[1].lats, plobjects[1].heights,
                                  cloud, levels=np.arange(0,5.5,0.5), 
                                  cmap='cividis', extend='both')
    ax[0,1].set_title('Cloud deck tracer', fontsize=fsize)
    cbar2 = plt.colorbar(cloud_full, ax=ax[0,1], orientation='vertical')
    cbar2.ax.set_title('years')

    surf_zoom = ax[1,0].contourf(plobjects[0].lats, plobjects[0].heights[13:38],
                                 surface[13:38], levels=np.arange(16,29,1), 
                                 cmap='cividis', extend='both')
    ax[1,0].set_title('Zoom', fontsize=fsize)
    ax[1,0].set_ylabel('Height / km', fontsize=fsize)
    ax[1,0].set_xlabel('Latitude / deg', fontsize=fsize)
    cbar3 = plt.colorbar(surf_zoom, ax=ax[1,0], orientation='vertical', aspect=10)
    cbar3.ax.set_title('years')

    cloud_zoom = ax[1,1].contourf(plobjects[1].lats, plobjects[1].heights[25:50],
                                  cloud[25:50], levels=np.arange(0,1.6,0.2), 
                                  cmap='cividis', extend='both')
    ax[1,1].set_title('Zoom', fontsize=fsize)
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
    ax1.set_title('Surface tracer', fontsize=fsize)
    ax1.set_ylabel('Height / km', fontsize=fsize)
    ax1.set_xlabel('Age / years', fontsize=fsize)
    ax1.grid()
    ax1.legend()

    for coord in coords:
        lat_lab = plobjects[1].lats[coord]
        ax2.plot(cloud[:,coord], plobjects[1].heights,
                 label=f'{lat_lab}$^\circ$ lat')
    ax2.set_title('Cloud tracer', fontsize=fsize)
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
    timedict = {'1':'Day 0', '2':'Day 6', '3':'Day 12', 
                '4':'Day 18', '5':'Day 24', '6': 'Day 30'}
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

    fig, ax = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)

    cf1 = ax[0,0].contourf(plobject.lons, plobject.lats, surface[0,0,:,:],
                     levels=np.arange(16.5,22.5,0.25), 
                     extend='both', cmap='cividis')
    ax[0,0].set_ylabel('Latitude / deg', fontsize=fsize)
    ax[0,0].set_title(f'h={np.round(plobject.heights[levs[0]],0)} km, day 0', fontsize=fsize)
    cbar1 = plt.colorbar(cf1, orientation='vertical', fraction=0.05)
    cbar1.set_label(f'years', loc='center')

    cf2 = ax[0,1].contourf(plobject.lons, plobject.lats,surface[-1,0,:,:],
                     levels=np.arange(16.5,22.5,0.25), 
                     extend='both', cmap='cividis')
    ax[0,1].set_title(f'h={np.round(plobject.heights[levs[0]],0)} km, day 12', fontsize=fsize)
    cbar2 = plt.colorbar(cf2, orientation='vertical', fraction=0.05)
    cbar2.set_label(f'years', loc='center')

    cf3 = ax[1,0].contourf(plobject.lons, plobject.lats, surface[0,-1,:,:],
                     levels=np.arange(23.5,27,0.25), 
                     extend='both', cmap='cividis')
    cbar3 = plt.colorbar(cf3, orientation='vertical', fraction=0.05)
    cbar3.set_label(f'years', loc='center')
    ax[1,0].set_ylabel('Latitude / deg', fontsize=fsize)
    ax[1,0].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[1,0].set_title(f'h={np.round(plobject.heights[levs[-1]],0)} km, day 0', fontsize=fsize)
    
    cf4 = ax[1,1].contourf(plobject.lons, plobject.lats, surface[-1,-1,:,:],
                     levels=np.arange(23.5,27,0.25), 
                     extend='both', cmap='cividis')
    cbar4 = plt.colorbar(cf4, orientation='vertical', fraction=0.05)
    cbar4.set_label(f'years', loc='center')
    ax[1,1].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[1,1].set_title(f'h={np.round(plobject.heights[levs[-1]],0)} km, day 12', fontsize=fsize)

    fig.suptitle('Age of air in the deep atmosphere', fontsize=fsize+2, y=0.95)

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def wind_composites(plobject, times=[1856,1858], levs=[15,22],
               fsize=14, qscale=1, n=4, clevs=np.arange(-0.06,0.07,0.01),
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
    ax[0,0].set_title(f'h={np.round(plobject.heights[levs[0]],0)} km, day 0', fontsize=fsize)
    
    cf2 = ax[0,1].contourf(plobject.lons, plobject.lats, w[-1,0,:,:],
                     levels=clevs, cmap='coolwarm', extend='both', norm=TwoSlopeNorm(0))
    q2 = ax[0,1].quiver(X[::n, ::n], Y[::n, ::n], -u[-1,0,::n,::n],
                   v[-1,0,::n,::n], angles='xy', scale_units='xy', scale=qscale)
    ax[0,1].quiverkey(q2, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    ax[0,1].set_title(f'h={np.round(plobject.heights[levs[0]],0)} km, day 12', fontsize=fsize)
    
    cf3 = ax[1,0].contourf(plobject.lons, plobject.lats, w[0,-1,:,:],
                     levels=clevs, cmap='coolwarm', extend='both', norm=TwoSlopeNorm(0))
    q3 = ax[1,0].quiver(X[::n, ::n], Y[::n, ::n], -u[0,-1,::n,::n],
                   v[0,-1,::n,::n], angles='xy', scale_units='xy', scale=qscale)
    ax[1,0].quiverkey(q3, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    ax[1,0].set_ylabel('Latitude / deg', fontsize=fsize)
    ax[1,0].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[1,0].set_title(f'h={np.round(plobject.heights[levs[-1]],0)} km, day 0', fontsize=fsize)
    
    cf4 = ax[1,1].contourf(plobject.lons, plobject.lats, w[-1,-1,:,:],
                     levels=clevs, cmap='coolwarm', extend='both', norm=TwoSlopeNorm(0))
    q4 = ax[1,1].quiver(X[::n, ::n], Y[::n, ::n], -u[-1,-1,::n,::n],
                   v[-1,-1,::n,::n], angles='xy', scale_units='xy', scale=qscale)
    ax[1,1].quiverkey(q4, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %str(qscale*10),
                 labelpos='E', coordinates='axes')
    ax[1,1].set_xlabel('Longitude / deg', fontsize=fsize)
    ax[1,1].set_title(f'h={np.round(plobject.heights[levs[-1]],0)} km, day 12', fontsize=fsize)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf4, cax=cbar_ax)
    cbar.set_label(f'Vertical wind / m/s', loc='center')
    fig.suptitle('General circulation of the deep atmosphere', fontsize=fsize+2, y=0.95)

    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
if __name__ == "__main__":

    simulations = init_data(allpaths)
    # Instantiate data objects used in plots
    # List item 1 is Surface, List item 2 is Cloud

    # Figure 1
    baseline_atmosphere(simulations[0], trange=(0,-1), savearg=False,
                        sformat=saveas)
    # Figure 2
    zonal_mean_aoa(simulations, savearg=False, sformat=saveas)
    # Figure 3
    aoa_profiles(simulations, savearg=False, sformat=saveas)
    # Figure 4
    polar_aoa(simulations[0], savearg=False, sformat=saveas)
    # Figure 5
    aoa_slices(simulations[0], savearg=False, sformat=saveas)
    # Figure 6
    wind_composites(simulations[0], savearg=False, sformat=saveas)
    # Figure 7
    time_series(simulations[0], key='vitw', 
                coords=[(16,86,48),(22,86,48),(30,86,48)], 
                ptitle='vertical wind', ylab='Wind velocity', 
                unit='m/s', trange=[1777,1877], tunit='Venus days',
                fsize=14, savearg=False, sformat=saveas)

# %%
