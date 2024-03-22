""" Uses functions from venuslab to auto-generate plots for
    'Tidally locked circulation regimes in the deep atmosphere of Venus',
     Cohen et al. 2024                                       """

""" Usage python: make_aoa.py
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
from venusdyn import psi_m, zmzw, wind_vectors
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
    ax1.set_xlabel('Latitude [deg]', fontsize=fsize)
    ax1.set_ylabel('Height [km]', fontsize=fsize)
    cbar = plt.colorbar(cf, ax=ax1, orientation='vertical')
    cbar.ax.set_title('m/s')
    lat_ticks = ax1.get_xticks()

    cs = ax2.contour(plobject.lats, plobject.heights[:-1],
                mass_streamfunction, colors='black',
                levels=np.arange(-100,100,10))
    ax2.clabel(cs, cs.levels, inline=True)
    ax2.set_title('Mean meridional mass streamfunction [$10^{10}$ kg/s]',
                  fontsize=fsize)
    ax2.set_xlabel('Latitude [deg]', fontsize=fsize)
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
    ax[0,0].set_ylabel('Height [km]', fontsize=fsize)
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
    ax[1,0].set_ylabel('Height [km]', fontsize=fsize)
    ax[1,0].set_xlabel('Latitude [deg]', fontsize=fsize)
    cbar3 = plt.colorbar(surf_zoom, ax=ax[1,0], orientation='vertical', aspect=10)
    cbar3.ax.set_title('years')

    cloud_zoom = ax[1,1].contourf(plobjects[1].lats, plobjects[1].heights[25:50],
                                  cloud[25:50], levels=np.arange(0,1.6,0.2), 
                                  cmap='cividis', extend='both')
    ax[1,1].set_title('Zoom', fontsize=fsize)
    ax[1,1].set_xlabel('Latitude [deg]', fontsize=fsize)
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
    ax1.set_ylabel('Heights [km]', fontsize=fsize)
    ax1.set_xlabel('Age [years]', fontsize=fsize)
    ax1.grid()
    ax1.legend()

    for coord in coords:
        lat_lab = plobjects[1].lats[coord]
        ax2.plot(cloud[:,coord], plobjects[1].heights,
                 label=f'{lat_lab}$^\circ$ lat')
    ax2.set_title('Cloud tracer', fontsize=fsize)
    ax2.set_xlabel('Age [years]', fontsize=fsize)
    ax2.grid()
    ax2.legend()

    plt.subplots_adjust(wspace=0.1)
    
    if savearg==True:
        plt.savefig(savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def polar_aoa(plobject, trange=np.arange(1860,1866,1), lev=25,
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
                                    levels=np.arange(24.5,27,0.2),
                                    extend='both', 
                                    transform=ccrs.PlateCarree())
        axitem.set_title(timedict[str(counter)])

    cbar_ax = fig.add_axes([0.15, -0.05, 0.75, 0.02])
    cbar = fig.colorbar(polarsnap, cax=cbar_ax, orientation='horizontal',
                        extend='both')
    cbar.set_label(label='years', fontsize=fsize-2)
    fig.suptitle(f'Age of air at south pole, \n h={np.round(plobject.heights[lev],0)} km', fontsize=fsize+2, y=1.1)
    plt.subplots_adjust(wspace=None, hspace=0.18)

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
    baseline_atmosphere(simulations[0], trange=(0,-1), savearg=True,
                        sformat=saveas)
    # Figure 2
    zonal_mean_aoa(simulations, savearg=True, sformat=saveas)
    # Figure 3
    aoa_profiles(simulations, savearg=True, sformat=saveas)
    # Figure 4
    polar_aoa(simulations[0], savearg=True, sformat=saveas)
    # Subplots for Figure 5
    wind_vectors(simulations[0], meaning=False, time_slice=1850, lev=16,
                 savearg=True, savename='wind_vectors_lev6_t1850.png', sformat=saveas)


# %%
