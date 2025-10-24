# Validation & testing of VIRTIS v_geo_grid projection datasets

## Filepaths
# %%
datadir = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/ALL_orbits/'
common_prefix = 'Accumulated_Grids_Data_VI0_CO_bandratio_'
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

test_orbits = ['VI0093']
test_orbs = ['VI0093_05']

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
        mask = np.count_nonzero(~np.isnan(self.arr))
        lat_max, lon_max = mask[0][0], mask[0][1]
        return lat_max, lon_max

## Functions
# %%
def convert_rad_to_CO(arr):
    """ EXTREMELY ROUGH conversion of radiance ratio to CO ppm
        Literally based on eyeballing Fig. 6 in Tsang+2009"""
    
    ppm = (arr - 1.344) / 0.0464

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
          plot=True, save=False,
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
        cbar.set_label('ppmv', color='black')

        if save==True:
            plt.savefig(savepath + 'virtis_validate_' + str(orbit) + '.png', format='png', bbox_inches='tight')

        plt.show()

    return orbit_mean


# %%
if __name__ == "__main__":

    # Load vgeo_grid projections
    pref = datadir + common_prefix
    projections = []
    for case in range(1, len(case_dict)+1):
        dict_key= 'case' + str(case)
        fn = pref + case_dict[dict_key]
        proj = Vgeo(name=case_dict[dict_key], dat=fn+'.dat', hdr=fn+'.hdr')
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
                  plot=True,
                  save=True)
    print('Test 2 completed: Orbits plotted for visual comparison.')

    # Test 3: Compare scatter in data
    for o in test_orbits:
        idx_lat, idx_lon = projections[0].counts()
        case1_points = orbit_obs(imobject=projections[0].img,
                                    libobject=projections[0].lib,
                                    orbit=o,
                                    lat_idx=idx_lat,
                                    lon_idx=idx_lon)
        case2_points = orbit_obs(imobject=projections[1].img,
                                    libobject=projections[1].lib,
                                    orbit=o,
                                    lat_idx=idx_lat,
                                    lon_idx=idx_lon)
        std_case1 = np.nanstd(case1_points)
        std_case2 = np.nanstd(case2_points)
        fig, ax = plt.subplots()
        ax.scatter(range(len(case1_points)), case1_points, label='Case 1', alpha=0.7)
        ax.scatter(range(len(case2_points)), case2_points, label='Case 2', alpha=0.7)
        ax.set_title(f'Orbit {o} CO observations at pixel ({idx_lat}, {idx_lon})')
        ax.set_xlabel('Observation')
        ax.set_ylabel('CO ppmv')
        ax.legend()
        plt.savefig(f'/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/virtis_validate_scatter_{o}_1vs2.png', format='png', bbox_inches='tight')
        plt.show()
        print(f'Orbit {o} - Case 1 std: {std_case1:.2f}, Case 2 std: {std_case2:.2f}')

        max_lat, max_lon = projections[6].counts()
        case7_points = orbit_obs(imobject=projections[6].img,
                                    libobject=projections[6].lib,
                                    orbit=o,
                                    lat_idx=idx_lat,
                                    lon_idx=idx_lon)
        case8_points = orbit_obs(imobject=projections[7].img,
                                    libobject=projections[7].lib,
                                    orbit=o,
                                    lat_idx=max_lat,
                                    lon_idx=max_lon)
        std_case7 = np.nanstd(case7_points)
        std_case8 = np.nanstd(case8_points)
        fig, ax = plt.subplots()
        ax.scatter(range(len(case7_points)), case7_points, label='Case 7', alpha=0.7)
        ax.scatter(range(len(case8_points)), case2_points, label='Case 8', alpha=0.7)
        ax.set_title(f'Orbit {o} CO observations at pixel ({max_lat}, {max_lon})')
        ax.set_xlabel('Observation')
        ax.set_ylabel('CO ppmv')
        ax.legend()
        plt.savefig(f'/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/virtis_validate_scatter_{o}_7vs8.png', format='png', bbox_inches='tight')
        plt.show()
        print(f'Orbit {o} - Case 7 std: {std_case1:.2f}, Case 8 std: {std_case2:.2f}')
    print('Test 3 completed: Scatter compared for selected orbits and pixels.')      