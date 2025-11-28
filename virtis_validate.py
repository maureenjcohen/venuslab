# Validation & testing of VIRTIS v_geo_grid projection datasets

## Filepaths
# %%
datadir = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/ALL_orbits/'
common_prefix = 'Accumulated_Grids_DATA_VI0_CO_bandratio_'
case_dict = {
'case1' : '2.29_2.32_152-158K',
'case2' : '2.29_2.32_interpol1_152-158K',
'case3' : '2.30_2.32_152-158K',
'case4' : '2.30_2.32_interpol1_152-158K',
'case5' : '2.29_2.32_150-165K',
'case6' : '2.29_2.32_interpol1_150-165K',
'case7' : '2.30_2.32_150-165K',
'case8' : '2.30_2.32_interpol1_150-165K',
'case9' : '2.29_2.32_interpol1_152-158K_ALLexp'}

test_orbits = ['VI0066','VI0093','VI0105', 'VI0153']
# VI0901 is a hot orbit
test_obs = ['VI0093_05']

## Imports
# %%
import spectral.io.envi as envi
from spectral import *
import matplotlib.pyplot as plt
import numpy as np

## Class definition for v_geo_grid output
# %%
class Vgeo:
    """ A class object that stores an output from a v_geo_grid pipeline run """
    
    def __init__(self, name, dat, hdr):
        """ Initialize Vgeo object from ENVI .hdr and .dat files 
            Get number of observations and number of unique orbits"""
        self.name = name
        self.img = envi.open(hdr, dat)
        self.lib = envi.read_envi_header(hdr)

        self.observations = [x.split('.')[0] for x in self.lib['band names']]

        self.unique_orbits = list(dict.fromkeys([x.split('_')[0] for x in self.lib['band names']]))

        self.arr = self.img.read_bands(np.arange(0,len(self.observations)))

    def counts(self):
        """ Count number of observations per pixel (i.e., non-nan values) 
            Return indices of pixels with max counts            """
        count = np.count_nonzero(~np.isnan(self.arr), axis=-1)
        lat_max = np.where(count==np.max(count))[0]
        lon_max = np.where(count==np.max(count))[1]
        return lat_max, lon_max
    
    def orbit_counts(self, orbit):
        """ Count number of observations per pixel for a given orbit """
        band_names = [x for x in self.lib['band names'] if x.split('_')[0] == orbit]
        idx = [self.lib['band names'].index(x) for x in self.lib['band names'] if x in band_names]
 
        count = np.count_nonzero(~np.isnan(self.arr[:,:,idx[0]:idx[-1]+1]), axis=-1)
        lat_max = np.where(count==np.max(count))[0]
        lon_max = np.where(count==np.max(count))[1]
        return lat_max, lon_max
## Functions
# %%
def convert_rad_to_CO(arr):
    """ EXTREMELY ROUGH conversion of radiance ratio to CO ppm
        Literally based on eyeballing Fig. 6 in Tsang+2009"""
    
    #ppm = (arr - 1.344) / 0.0464
    ppm = arr

    return ppm

# %%
def orbit_obs(imobject, libobject, orbit, lat_idx, lon_idx):
    """ Get all observations for one orbit and one pixel """
    band_names = [x for x in libobject['band names'] if x.split('_')[0] == orbit]
    idx = [libobject['band names'].index(x) for x in libobject['band names'] if x in band_names]
 
    if len(idx) > 1:
        np_array = convert_rad_to_CO(imobject.read_bands(np.arange(idx[0],idx[-1]+1)))
        orbit_points = np_array[lat_idx,lon_idx,:]
    else:
        np_array = convert_rad_to_CO(imobject.read_band(idx[0]))
        orbit_points = np_array[lat_idx,lon_idx]

    return orbit_points
# %%
def orbit(imobject, libobject, orbit,
          levels=np.arange(0,40,0.5),
          plot=True, save=False, savename='orbit_placerholder',
          savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Plot nanmean of one orbit onto a contourf """
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
        cbar.set_label('ratio', color='black')

        if save==True:
            plt.savefig(savepath + 'virtis_validate_' + savename + '_' + str(orbit) + '.png', format='png', bbox_inches='tight')

        plt.show()

    return orbit_mean

# %%
def orbit_map(imobject, libobject, orbit):
    """ Get array of values of one orbit"""

    band_names = [x for x in libobject['band names'] if x.split('_')[0] == orbit]
    idx = [libobject['band names'].index(x) for x in libobject['band names'] if x in band_names]

    if len(idx) > 5:
        orbit_arr = convert_rad_to_CO(imobject.read_bands(np.arange(idx[0],idx[-1]+1)))
        return orbit_arr
    else:
        return None  # Not enough data to make an array

    

# %%
if __name__ == "__main__":

    # Load vgeo_grid projections
    pref = datadir + common_prefix
    projections = []
    for case in range(1, len(case_dict)+1):
        dict_key= 'case' + str(case)
        fn = pref + case_dict[dict_key]
        proj = Vgeo(name=case_dict[dict_key], dat=fn+'.DAT', hdr=fn+'.HDR')
        projections.append(proj)
    print('Loaded {} v_geo_grid projections.'.format(len(projections)))

    # Test 1: 150-165K range should be longer than 152-158K range
    if projections[4].img.shape[2] > projections[0].img.shape[2]:
        print('Test 1a passed: 150-165K range has more observations than 152-158K range.')
    else:
        print('Test 1a failed: 150-165K range does not have more observations than 152-158K range.')

    if projections[6].img.shape[2] > projections[2].img.shape[2]:
        print('Test 1b passed: 150-165K range has more observations than 152-158K range.')
    else:
        print('Test 1b failed: 150-165K range does not have more observations than 152-158K range.')

    if projections[8].img.shape[2] > projections[1].img.shape[2]:
        print('Test 1c passed: All exposures has more observations than exposure limits.')
    else:
        print('Test 1c failed: All exposures does not have more observations than exposure limits.')

    # Test 2: Plot and compare orbits of different datasets
    for o in test_orbits:
        for proj in projections:
            print(f'Plotting orbit {o} of projection {proj.name}')
            orbit(imobject=proj.img,
                  libobject=proj.lib,
                  orbit=o,
                  levels=np.arange(1.5,3.1,0.1),
                  plot=True,
                  save=True,
                  savename=proj.name)
    print('Test 2 completed: Orbits plotted for visual comparison.')

    # Test 3: Compare scatter in data
    for o in test_orbits:
        idx_lat, idx_lon = projections[0].orbit_counts(orbit=o)
        case1_points = orbit_obs(imobject=projections[0].img,
                                    libobject=projections[0].lib,
                                    orbit=o,
                                    lat_idx=idx_lat[0],
                                    lon_idx=idx_lon[0])
        case2_points = orbit_obs(imobject=projections[1].img,
                                    libobject=projections[1].lib,
                                    orbit=o,
                                    lat_idx=idx_lat[0],
                                    lon_idx=idx_lon[0])
        std_case1 = np.nanstd(case1_points)
        std_case2 = np.nanstd(case2_points)
        fig, ax = plt.subplots()
        ax.scatter(range(len(case1_points)), case1_points, label=f'2.29 152-158K, std {str(np.round(std_case1,3))}', alpha=0.7)
        ax.scatter(range(len(case2_points)), case2_points, label=f'2.29 152-158K + interpol, std {str(np.round(std_case2,3))}', alpha=0.7)
        ax.set_title(f'Orbit {o} CO observations at pixel ({idx_lat[0]}, {idx_lon[0]})')
        ax.set_xlabel('Observation')
        ax.set_ylabel('CO ratio')
        ax.legend()
        plt.savefig(f'/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/virtis_validate_scatter_{o}_1vs2.png', format='png', bbox_inches='tight')
        plt.show()
        print(f'Orbit {o} - Case 1 std: {std_case1:.2f}, Case 2 std: {std_case2:.2f}')

        max_lat, max_lon = projections[6].orbit_counts(orbit=o)
        case7_points = orbit_obs(imobject=projections[6].img,
                                    libobject=projections[6].lib,
                                    orbit=o,
                                    lat_idx=max_lat[0],
                                    lon_idx=max_lon[0])
        case8_points = orbit_obs(imobject=projections[7].img,
                                    libobject=projections[7].lib,
                                    orbit=o,
                                    lat_idx=max_lat[0],
                                    lon_idx=max_lon[0])
        std_case7 = np.nanstd(case7_points)
        std_case8 = np.nanstd(case8_points)
        fig, ax = plt.subplots()
        ax.scatter(range(len(case7_points)), case7_points, label=f'2.30 150-165K, std {str(np.round(std_case7,3))}', alpha=0.7)
        ax.scatter(range(len(case8_points)), case8_points, label=f'2.30 150-165K + interpol, std {str(np.round(std_case8,3))}', alpha=0.7)
        ax.set_title(f'Orbit {o} CO observations at pixel ({max_lat[0]}, {max_lon[0]})')
        ax.set_xlabel('Observation')
        ax.set_ylabel('CO ratio')
        ax.legend()
        plt.savefig(f'/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/virtis_validate_scatter_{o}_7vs8.png', format='png', bbox_inches='tight')
        plt.show()
        print(f'Orbit {o} - Case 7 std: {std_case1:.2f}, Case 8 std: {std_case2:.2f}')
    print('Test 3 completed: Scatter compared for selected orbits and pixels.')      

    # Test 4: Compare mean scatter of all points and orbits between projections
    # %%
    orbs_list_0 = []
    for orb in projections[0].unique_orbits:
        orb_arr = orbit_map(imobject=projections[0].img,
                             libobject=projections[0].lib,
                             orbit=orb)
        if orb_arr is not None:
            orb_std = np.nanstd(orb_arr, axis=-1)
            orbs_list_0.append(np.nanmean(orb_std))
    mean_std_0 = np.nanmean(np.array(orbs_list_0))

    orbs_list_1 = []
    for orb in projections[1].unique_orbits:
        orb_arr = orbit_map(imobject=projections[1].img,
                             libobject=projections[1].lib,
                             orbit=orb)
        if orb_arr is not None:
            orb_std = np.nanstd(orb_arr, axis=-1)
            orbs_list_1.append(np.nanmean(orb_std))
    mean_std_1 = np.nanmean(np.array(orbs_list_1))  

    orbs_list_2 = []
    for orb in projections[2].unique_orbits:
        orb_arr = orbit_map(imobject=projections[2].img,
                             libobject=projections[2].lib,
                             orbit=orb)
        if orb_arr is not None:
            orb_std = np.nanstd(orb_arr, axis=-1)
            orbs_list_2.append(np.nanmean(orb_std))
    mean_std_2 = np.nanmean(np.array(orbs_list_2))

    orbs_list_3 = []
    for orb in projections[3].unique_orbits:
        orb_arr = orbit_map(imobject=projections[3].img,
                             libobject=projections[3].lib,
                             orbit=orb)
        if orb_arr is not None:
            orb_std = np.nanstd(orb_arr, axis=-1)
            orbs_list_3.append(np.nanmean(orb_std))
    mean_std_3 = np.nanmean(np.array(orbs_list_3))
    print(f'Mean orbit std for case 1 (2.29 152-158K): {mean_std_0:.3f}')
    print(f'Mean orbit std for case 2 (2.29 152-158K + interpol): {mean_std_1:.3f}')
    print(f'Mean orbit std for case 3 (2.30 152-158K): {mean_std_2:.3f}')
    print(f'Mean orbit std for case 4 (2.30 152-158K + interpol): {mean_std_3:.3f}')

# %%
