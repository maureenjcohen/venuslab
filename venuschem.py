
# %%
## Import packages

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %%
def summ_stats(plobject, key, lev, t0, tf, savename='stats.png',
               savepath='/exomars/projects/mc5526/VPCM_full_chemistry_runs/scratch_plots/',
               save=False, sformat='png'):
    
    """ Compute variability of a chemical species
        Standard deviation on lon-lat grid """
    
    cube = plobject.data[key][t0:tf,lev,:,:]
    title_name = cube.long_name or cube.name
    title_height = np.round(plobject.heights[lev],2)
    cube = cube*1e6
    std = cube.std(dim='time_counter',skipna=True, keep_attrs=True)
    avg = cube.mean(dim='time_counter', skipna=True, keep_attrs=True)

    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    abd_plot = ax1.contourf(plobject.lons, plobject.lats, avg, cmap='afmhot')
    ax1.set_title(f'Mean {title_name} at {title_height} km')
    ax1.set_xlabel('Longitude / deg')
    ax1.set_ylabel('Latitude / deg')
    cbar1 = plt.colorbar(abd_plot, orientation='horizontal', ax=ax1)
    cbar1.set_label('ppm')

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
