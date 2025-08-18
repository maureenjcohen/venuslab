
# %%
## Import packages

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from venuspoles import add_cycl_point

# %%
def summ_stats(plobject, key, lev, t0, tf, savename='stats.png',
               savepath='/exomars/projects/mc5526/VPCM_full_chemistry_runs/scratch_plots/',
               save=False, sformat='png'):
    
    """ Compute variability of a chemical species
        Standard deviation on lon-lat grid """
    
    cube = plobject.data[key][t0:tf,lev,:,:]
    title_name = cube.long_name or cube.name
    title_height = np.round(plobject.heights[lev],2)
    if key=='n2':
        cube = cube*100
        unit = '%'
    else:
        cube = cube*1e6
        unit = 'ppm'

    std = cube.std(dim='time_counter',skipna=True, keep_attrs=True)
    avg = cube.mean(dim='time_counter', skipna=True, keep_attrs=True)

    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    abd_plot = ax1.contourf(plobject.lons, plobject.lats, avg, cmap='afmhot')
    ax1.set_title(f'Mean {title_name} at {title_height} km')
    ax1.set_xlabel('Longitude / deg')
    ax1.set_ylabel('Latitude / deg')
    cbar1 = plt.colorbar(abd_plot, orientation='horizontal', ax=ax1)
    cbar1.set_label(unit)

    std_plot = ax2.contourf(plobject.lons, plobject.lats, 100*std/avg, cmap='copper')
    ax2.set_title(f'Coeff of variation in {title_name} at {title_height} km')
    ax2.set_xlabel('Longitude / deg')
    cbar2 = plt.colorbar(std_plot, orientation='horizontal',ax=ax2)
    cbar2.set_label('%')

    if save==True:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    plt.show()
    

# %%
def prod_loss(plobject, lev, t0, tf, savename='co_prod_loss.png',
              savepath='/exomars/projects/mc5526/VPCM_full_chemistry_runs/scratch_plots/',
              save=False, sformat='png'):
    """ Difference in production and loss of CO, area-weighted mean """
    title_height = np.round(plobject.heights[lev],2)

    prod = plobject.data['prod_co'][t0:tf,lev,:,:]
    loss = plobject.data['loss_co'][t0:tf,lev,:,:]

    diff = prod - loss
    mean_diff = diff.mean(dim='time_counter', skipna=True, keep_attrs=True)
    
    fig, ax = plt.subplots(figsize=(8,6))
    cf = ax.contourf(plobject.lons, plobject.lats, mean_diff, cmap='hot')
    ax.set_title(f'Mean CO prod minus loss at {title_height} km')
    ax.set_xlabel('Longitude / deg')
    ax.set_ylabel('Latitude / deg')
    cbar = plt.colorbar(cf)
    cbar.set_label('cm-3.s-1')

    if save==True:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    plt.show()

# %%
def vex_compare(plobject, time_slice, i=0, j=-60, lev=18, key='co', 
                savename='vex_compare.png',
                savepath='/exomars/projects/mc5526/VPCM_full_chemistry_runs/scratch_plots/',
                save=False, sformat='png'):
    """ Projection onto sphere of CO abundance at 35 km
        Direct comparison with Tsang et al. 2008, Figs. 6, 8, 9, 10 
        doi: 10.1029/2008JE003089                               """
    cube = plobject.data[key][time_slice,lev,:,:]*1e6
    cube_name = plobject.data[key].long_name or plobject.data[key].name
    height = np.round(plobject.heights[lev],2)
    new_cube, new_lon = add_cycl_point(cube, cube.lon, -1)

    ortho = ccrs.Orthographic(central_longitude=i, central_latitude=j)
    # Specify orthographic projection centered at lon/lat i, j
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ortho)
    ax.set_global()

    plimg = ax.contourf(new_lon, plobject.lats, new_cube, transform=ccrs.PlateCarree(),
                        cmap='nipy_spectral')
    ax.set_title(f'{cube_name}, {height} km', color='black', y=1.05, fontsize=14)
    ax.gridlines(draw_labels=True, linewidth=1.5, linestyle='dotted', color='black', alpha=0.5)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('ppm', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    if save==True:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
# %%
def fig11_compare(plobject, lev=18, trange=(0, 20), key='co',
                  savename='CO_profiles.png',
                  savepath='/exomars/projects/mc5526/VPCM_full_chemistry_runs/scratch_plots/',
                  save=False, sformat='png'):
    """ Latitudinal profiles of CO abundance at 35 km
        Direct comparison to Tsang et al. 2009, Fig. 11 
        doi: 10.1016/j.icarus.2009.01.001               """
    cube = plobject.data[key][trange[0]:trange[1],lev,:,:]*1e6
    cube_name = plobject.data[key].long_name or plobject.data[key].name
    height = np.round(plobject.heights[lev],2)

    zonal_means = cube.mean(dim='lon')
    fig, ax = plt.subplots(figsize=(8,4))
    color = iter(plt.cm.rainbow(np.linspace(0, 1, trange[1]-trange[0])))
    for time_slice in range(trange[1]-trange[0]):
        c = next(color)
        ax.plot(plobject.lats, zonal_means[time_slice,:], color=c)
    ax.grid(axis='y', linestyle='dashed', color='grey', alpha=0.5)
    ax.set_xlabel('Latitude / deg')
    ax.set_ylabel('Zonal mean CO abundance / ppm')
    ax.set_ylim([4,12])
    ax.set_title(f'{cube_name} profiles at {height} km')
    if save==True:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
# %%
