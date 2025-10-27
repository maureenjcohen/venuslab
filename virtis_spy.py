# Filepaths and such
# %%
datadir = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/200_orbits/200_orbits/'
fn29 = 'Accumulated_Grids_DATA_VI01-2_CO_band_2.29_152-158K_ExpGT0.1_maxEM85_minINC100_1x1_radGT0.02.DAT'
hdr29 = 'Accumulated_Grids_DATA_VI01-2_CO_band_2.29_152-158K_ExpGT0.1_maxEM85_minINC100_1x1_radGT0.02.HDR'
fn30 = 'Accumulated_Grids_DATA_VI01-2_CO_band_2.30_152-158K_ExpGT0.1_maxEM85_minINC100_1x1_radGT0.02.DAT'
hdr30 = 'Accumulated_Grids_DATA_VI01-2_CO_band_2.30_152-158K_ExpGT0.1_maxEM85_minINC100_1x1_radGT0.02.HDR'
fn32 = 'Accumulated_Grids_DATA_VI01-2_CO_band_2.32_152-158K_ExpGT0.1_maxEM85_minINC100_1x1.DAT'
hdr32 = 'Accumulated_Grids_DATA_VI01-2_CO_band_2.32_152-158K_ExpGT0.1_maxEM85_minINC100_1x1.HDR'
fnrat = 'Accumulated_Grids_DATA_VI01-2_CO_bandratio_2.30_2.32_152-158K_ExpGT0.1_maxEM85_minINC100_1x1_radGT0.02.DAT'
hdrrat = 'Accumulated_Grids_DATA_VI01-2_CO_bandratio_2.30_2.32_152-158K_ExpGT0.1_maxEM85_minINC100_1x1_radGT0.02.HDR'
# %%
# Import SpectralPython and other stuff
import spectral.io.envi as envi
from spectral import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# %%
img = envi.open(datadir+hdrrat,datadir+fnrat)
lib = envi.read_envi_header(datadir+hdrrat)

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
def convert_m2_to_cm2(arr):
    """ Convert radiance array from W/m2 to W/cm2
        For Barstow 2012 formula"""
    
    rad = arr/10000

    return rad

# %%
def read_ratio(imobject, idx, convert2ppm):

    if len(idx) > 1:
        arr = imobject.read_bands(np.arange(idx[0],idx[-1]+1))
    else:
        arr = imobject.read_band(idx[0])
    
    if convert2ppm==True:
        arr = convert_rad_to_CO(arr)
    
    return arr

# %%
def orbit(data, libobject, orbit,
          levels=np.arange(0,40,0.5), convert2ppm=True,
          plot=True, save=False, savename='virtis_co_',
          savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Plot nanmean of X observations onto one contourf
        Represents one orbit                                               """
    band_names = [x for x in libobject['band names'] if x.split('_')[0] == orbit]
    idx = [libobject['band names'].index(x) for x in libobject['band names'] if x in band_names]
 
    if type(data).__name__ == 'BsqFile':
        arr = read_ratio(imobject=data, idx=idx, convert2ppm=convert2ppm)
    elif type(data).__name__ == 'ndarray':
        arr = data[:,:,idx[0]:idx[-1]+1]
    else:
        print('Wrong data type')

    if len(arr.shape) == 3:
        orbit_mean = np.nanmean(arr, axis=-1)
    elif len(arr.shape) == 2:
        orbit_mean = arr
    else:
        print('Something wrong with dimensions of array')

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
            plt.savefig(savepath + savename + str(orbit) + '.png', format='png', bbox_inches='tight')

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
def moving_average(a, n=5):
    ret = np.cumsum(a.filled(0))
    ret[n:] = ret[n:] - ret[:-n]
    counts = np.cumsum(~a.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ret[~a.mask] /= counts[~a.mask]
    ret[a.mask] = np.nan

    return ret

# %%
def barstow_co(band29, lib29, band32, lib32, t0, tf):

    common_elements, indices_29, indices_32 = np.intersect1d(lib29['wavelength'], lib32['wavelength'], return_indices=True)

    band29_in_cm2 = convert_m2_to_cm2(band29.read_bands(indices_29[t0:tf]))
    band32_in_cm2 = convert_m2_to_cm2(band32.read_bands(indices_32[t0:tf]))
    
    numerator = band32_in_cm2 - 738.57*(band29_in_cm2**2) - 0.31681*band29_in_cm2 - 9.8043e-09
    denominator = -417.62*(band29_in_cm2**2) - 0.22764*band29_in_cm2 + 8.9506e-09
    co_in_ppmv = 35*np.exp(numerator/denominator)

    return co_in_ppmv

# %%
def tsang_co(band30, lib30, band32, lib32, t0, tf):

    common_elements, indices_30, indices_32 = np.intersect1d(lib30['wavelength'], lib32['wavelength'], return_indices=True)

    bands30 = band30.read_bands(indices_30[t0:tf])
    bands32 = band32.read_bands(indices_32[t0:tf])
    co_in_ppmv = convert_rad_to_CO(bands30/bands32)

    return co_in_ppmv

# %%
def tsang_v2(band29, lib29, band32, lib32, t0, tf):

    common_elements, indices_29, indices_32 = np.intersect1d(lib29['wavelength'], lib32['wavelength'], return_indices=True)

    bands29 = band29.read_bands(indices_29[t0:tf])
    bands32 = band32.read_bands(indices_32[t0:tf])
    co_in_ppmv = convert_rad_to_CO(bands29/bands32)

    return co_in_ppmv

# %%
def counts(data):
    """ Count non-nan values for each grid box  
        Input: numpy array                  """

    counts = np.count_nonzero(~np.isnan(data), axis=-1)
    lat_max = np.where(counts==np.max(counts))[0]
    lon_max = np.where(counts==np.max(counts))[1]

    fig, ax = plt.subplots(figsize=(8,6))
    cf = ax.contourf(counts, cmap='plasma')
    ax.set_title('Number of observations')
    plt.colorbar(cf)
    plt.show()

    return lat_max, lon_max

# %%
def orbit_mean_loc(arr, lib, lat_idx, lon_idx, orbit):
    """ Orbit mean for a lat x lon location """

    band_names = [x for x in lib['band names'] if x.split('_')[0] == orbit]
    idx = [lib['band names'].index(x) for x in lib['band names'] if x in band_names]

    if len(idx) > 1:
        loc_mean = np.nanmean(arr[lat_idx, lon_idx, idx[0]:idx[-1]])
    else:
        loc_mean = np.nan

    return loc_mean
 
# %%
def loc_time_series(arr, lib, lat_idx, lon_idx, orbit_list,
                    plot=True, save=False, savename='barstow_orbit_time_series_',
                    savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Time series of orbit means for a particular lat x lon"""

    vals_list = []
    for orbit in orbit_list:
        print(str(orbit))
        val = orbit_mean_loc(arr, lib, lat_idx, lon_idx, orbit)
        vals_list.append(val)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(np.arange(0, len(vals_list)), np.array(vals_list))
    ax.set_title('Time series')
    ax.set_xlabel('Orbit')
    ax.set_ylabel('Mean CO / ppmv')

    if save==True:
        plt.savefig(savepath + savename + str(lat_idx) + '_' + str(lon_idx) + '.png', format='png', bbox_inches='tight')

    plt.show()

    return np.array(vals_list)
# %%
