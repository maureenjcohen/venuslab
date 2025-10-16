# Filepaths and such
# %%
datadir = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/envi_files/'
fn = 'Accumulated_Grids_DATA_VI00_CO_bandratio_2.30_2.32_151-159K_Nightside.DAT'
hdr = 'Accumulated_Grids_DATA_VI00_CO_bandratio_2.30_2.32_151-159K_Nightside.HDR'

# %%
# Import SpectralPython and other stuff
import spectral.io.envi as envi
from spectral import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# %%
img = envi.open(datadir+hdr,datadir+fn)
lib = envi.read_envi_header(datadir+hdr)

# %%
# Get list of orbits, no duplicates
orbs = [x.split('_')[0] for x in lib['band names']]
unique_orbits = list(dict.fromkeys(orbs))

# %%
def convert_rad_to_CO(arr):
    """ EXTREMELY ROUGH conversion of radiance ratio to CO ppm
        Literally based on eyeballing Fig. 6 in Tsang+2009"""
    
    ppm = (arr - 1.344) / 0.0464

    return ppm

# %%
def orbit(imobject, libobject, orbit,
          levels=np.arange(0,40,0.5),
          plot=True, save=False,
          savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Plot nanmean of X observations onto one contourf
        Represents one orbit                                               """
    band_names = [x for x in libobject['band names'] if x.split('_')[0] == orbit]
    idx = [libobject['band names'].index(x) for x in libobject['band names'] if x in band_names]
 
    if len(idx) > 1:
        np_array = convert_rad_to_CO(imobject.read_bands(np.arange(idx[0],idx[-1]+1)))
        orbit_mean = np.nanmean(np_array, axis=-1)
    else:
        orbit_mean = convert_rad_to_CO(imobject.read_band(idx[0]))

    if plot==True:
        lat_pix = np.arange(0, orbit_mean.shape[0])
        lon_pix = np.arange(0, orbit_mean.shape[1])

        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf(lon_pix, lat_pix, orbit_mean, levels=levels, cmap='jet')
        ax.set_title(f'VIRTIS CO map ~35 km for orbit {str(orbit)}')
        ax.set_xlabel('Longitude / pixels')
        ax.set_ylabel('Latitude / pixels')
        cbar = plt.colorbar(cf)
        cbar.set_label('ppmv', color='black')

        if save==True:
            plt.savefig(savepath + 'virtis_co_' + str(orbit) + '.png', format='png', bbox_inches='tight')

        plt.show()

    return orbit_mean


# %%
def animate_obs(imobject, t0, tf, levels=np.arange(0,40,0.5),
               savename='virtis_co_anim',
               savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):

    """ Input:  """
    
    np_array = convert_rad_to_CO(imobject.read_bands(np.arange(t0,tf)))
    lat_pix = np.arange(0, np_array.shape[0])
    lon_pix = np.arange(0, np_array.shape[1])
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define an update function that will be called for each frame  
    def animate(frame):
        plt.cla()
        cf = ax.contourf(lon_pix, lat_pix, np_array[:,:,frame], levels=levels, cmap='jet')


        ax.set_title('VIRTIS CO maps ~35 km', color='black', y=1.05, fontsize=14)
        ax.set_xlabel('Longitude / pixels')
        ax.set_ylabel('Latitude / pixels')
    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=range(0,tf-t0), interval=500, repeat=False)

    #mask = np.where(np_array >= np.nanmax(np_array))
    # max_time = mask[2][0]
    #Define the colorbar. The colorbar method needs a mappable object from which to take the colorbar
    cbar = plt.colorbar(ax.contourf(lon_pix, lat_pix, np_array[:,:,3], levels=levels, cmap='jet'))
    cbar.set_label('ppmv', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
    
    #plt.show()

    # Save the animation as an mp4 file
    ani.save(savepath + f'{savename}.mp4', writer='ffmpeg')
    # ani.save('myanimation.gif', writer='pillow') #alternative
    
# %%
def animate_orbits(orbits_list, imobject, libobject,
                   levels=np.arange(0,40,0.5),
                   savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):

    """ Input:  List of names of orbits to include in animation"""
    orbit_means = []
    for orb in orbits_list:
        orbit_mean = orbit(imobject, libobject, orb, levels=levels,
                             plot=False, save=False, savepath=savepath)
        orbit_means.append(orbit_mean)

    lat_pix = np.arange(0, orbit_means[0].shape[0])
    lon_pix = np.arange(0, orbit_means[0].shape[1])
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define an update function that will be called for each frame  
    def animate(frame):
        plt.cla()
        cf = ax.contourf(lon_pix, lat_pix, orbit_means[frame], levels=levels, cmap='jet')

        ax.set_title(f'VIRTIS CO maps ~35 km, orbit {orbits_list[frame]}', color='black', y=1.05, fontsize=14)
        ax.set_xlabel('Longitude / pixels')
        ax.set_ylabel('Latitude / pixels')
    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=range(0,len(orbit_means)), interval=750, repeat=False)

    #mask = np.where(np_array >= np.nanmax(np_array))
    # max_time = mask[2][0]
    #Define the colorbar. The colorbar method needs a mappable object from which to take the colorbar
    cbar = plt.colorbar(ax.contourf(lon_pix, lat_pix, orbit_means[0], levels=levels, cmap='jet'))
    cbar.set_label('ppmv', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
    
    #plt.show()

    # Save the animation as an mp4 file
    ani.save(savepath + f'{orbits_list[0]}_{len(orbits_list)}_orbits.mp4', writer='ffmpeg')
    # ani.save('myanimation.gif', writer='pillow') #alternative

# %%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w
# %%
