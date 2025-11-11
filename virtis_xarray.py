""" Extract VIRTIS data from ENVI files, preprocess, and write to netcdf"""

# %%
# Import packages
import spectral.io.envi as envi
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# %%
# File paths
datadir = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/bands_ALL_orbits_LST/'
band29 = 'Accumulated_Grids_DATA_VI0_CO_band_2.29_interpol1_150-165K_ALLexp_LST'
band32 = 'Accumulated_Grids_DATA_VI0_CO_band_2.32_interpol1_150-165K_ALLexp_LST'
virtis_log = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/VIRTIS_log_v5.0_20130129.csv'
which_x = 'lst'

# %%
class Band:
    """ A class object that stores an output from a v_geo_grid pipeline run 
        in an xarray Dataset object"""
    
    def __init__(self, name, x_coord, dat, hdr, log_path):
        """ Initialize Vgeo object from ENVI .hdr and .dat files 
            Get number of observations and number of unique orbits"""
        self.name = name
        self.img = envi.open(hdr, dat)
        self.lib = envi.read_envi_header(hdr)

        self.observations = [x.split('.')[0] for x in self.lib['band names']]

        self.unique_orbits = list(dict.fromkeys([x.split('_')[0] for x in self.lib['band names']]))

        arr = self.img.read_bands(np.arange(0,len(self.observations)))

        if x_coord == 'lst':
            lst1 = np.linspace(12,24,120) # Local solar time
            lst2 = np.linspace(0,12,120)
            x_values = np.concat([lst1,lst2])
        elif x_coord == 'lon':
            x_values = np.linspace(0,360,360) # Longitude
        else:
            print('Incorrect x_coord input, should be lst or lon')
        y_values = np.linspace(-90,90,180) # Latitude

        full_log = pd.read_csv(log_path, skiprows=1)
        virtis_m = full_log.loc[full_log['CHANNEL_ID'] == 'VIRTIS_M_IR']
        no_qub = [x.split('.')[0] for x in virtis_m['PRODUCT_ID']]
        virtis_m['PRODUCT_ID'] = no_qub
        obs_log = virtis_m.loc[virtis_m['PRODUCT_ID'].isin(self.observations)]
        self.log = obs_log

        time_values = pd.to_datetime(self.log['START_TIME'].values)
        da = xr.DataArray(arr, name=self.name, dims=('lat', x_coord, 'time'), coords={'lat': y_values, x_coord: x_values, 'time': time_values})
        da.attrs['description'] = 'Radiance at ' + self.name
        da.attrs['unit'] = 'W m-2 str-1 s-1'
        self.da = da

        ds = xr.Dataset({})
        ds[name] = da
        orb_only = [x.split('_')[0] for x in self.log['PRODUCT_ID']]
        ds['ORBIT'] = (('time'), orb_only)
        for item in self.log.columns.values:
            ds[item] = (('time'), self.log[item].values)
        self.ds = ds

# %%
class BandRatio:
    """ A class object that holds a band ratio from two Band class objects """

    def __init__(self, name, top_band, bottom_band):
        """ Initialise BandRatio using two Band objects and their attributes """
        self.name = name
        common_elements, indices_top, indices_bottom = np.intersect1d(top_band.lib['wavelength'], bottom_band.lib['wavelength'], return_indices=True)
        common_orbs, orbs_top, orbs_bottom = np.intersect1d(top_band.unique_orbits, bottom_band.unique_orbits, return_indices=True)
        common_obs, obs_top, obs_bottom = np.intersect1d(top_band.observations, bottom_band.observations, return_indices=True)
        
        self.da = top_band.da[:,:,indices_top]/bottom_band.da[:,:,indices_bottom]
        self.unique_orbits = common_orbs
        self.observations = common_obs

        mask = top_band.ds['PRODUCT_ID'].isin(self.observations)
        ds = top_band.ds.where(mask, drop=True)
        ratio_ds = ds.drop_vars(top_band.name)
        ratio_ds['RATIO'] = self.da
        self.ds = ratio_ds

    def counts(self, plot=False):
        """ Count number of observations per pixel (i.e., non-nan values) 
            Return indices of pixels with max counts            """
        count = np.count_nonzero(~np.isnan(self.data), axis=-1)
        lat_max = np.where(count==np.max(count))[0]
        lon_max = np.where(count==np.max(count))[1]
        return lat_max, lon_max

### Functions
# %%
def time_series(data, y_idx, x_idx,
                plot=True, save=False, savename='barstow_orbit_time_series_',
                savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Time series of orbit means for a particular lat x lon"""

    vals_list = []
    for orbit in data.unique_orbits:
        print(str(orbit))
        val = np.nanmean(data.data[x_idx, y_idx,])
        vals_list.append(val)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(np.arange(0, len(vals_list)), np.array(vals_list))
    ax.set_title('Time series')
    ax.set_xlabel('Orbit')
    ax.set_ylabel('Mean CO / ppmv')

    if save==True:
        plt.savefig(savepath + savename + str(y_idx) + '_' + str(x_idx) + '.png', format='png', bbox_inches='tight')

    plt.show()

    return np.array(vals_list)
# %%
if __name__ == 'main':

    # %%
    # Read in data as XR DataArrays with DateTime object timestamps
    data29 = Band('band29', which_x, datadir+band29+'.DAT', datadir+band29+'.HDR', virtis_log)
    data32 = Band('band32', which_x, datadir+band32+'.DAT', datadir+band32+'.HDR', virtis_log)

    band_ratio = BandRatio('tsang_v2', data29, data32)


