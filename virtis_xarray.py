""" Extract VIRTIS data from ENVI files, preprocess, and write to netcdf"""

# %%
# Import packages
import spectral.io.envi as envi
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy as sp

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
        self.x_coord = x_coord

        self.observations = [x.split('.')[0] for x in self.lib['band names']]

        self.unique_orbits = list(dict.fromkeys([x.split('_')[0] for x in self.lib['band names']]))

        arr = self.img.read_bands(np.arange(0,len(self.observations)))

        if x_coord == 'lst':
            lst1 = np.linspace(12,24,120) # Local solar time
            lst2 = np.linspace(0,12,120)
            x_values = np.concat([lst2[::-1],lst1[::-1]]) # Venus go brrrackwards
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
        self.x_coord = top_band.x_coord

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

    def counts(self):
        """ Count number of observations per pixel (i.e., non-nan values) 
            Return indices of pixels with max counts            """
        count = np.count_nonzero(~np.isnan(self.da), axis=-1)
        y_max = np.where(count==np.max(count))[0]
        x_max = np.where(count==np.max(count))[1]
        self.y_max = y_max
        self.x_max = x_max
    
    def orbit_means(self,trange=(0,None)):
        """ Time series of orbit means for all """

        vals_list = []
        tvals_list = []
        for orbit in self.unique_orbits[trange[0]:trange[1]]:
            print('Processing ' + str(orbit))
            subset = self.ds.where(self.ds['ORBIT'] == orbit)
            val = np.nanmean(subset['RATIO'].values, axis=-1)
            tval = pd.to_datetime(subset['START_TIME']).mean()
            vals_list.append(val)
            tvals_list.append(tval)
            del subset

        vals_arr = np.array(vals_list)
        print(vals_arr.shape)
        vals_arr = np.transpose(vals_arr, (1,2,0))
        print(vals_arr.shape)
        time_values = pd.to_datetime(np.array(tvals_list))
        omeans = xr.DataArray(vals_arr, name='Orbit means', dims=('lat', self.x_coord, 'time'), coords={'lat': self.ds.coords['lat'].values, self.x_coord: self.ds.coords[self.x_coord].values, 'time': time_values})
        omeans.attrs['description'] = 'Band ratio meaned per orbit'
        omeans.attrs['unit'] = 'Ratio of radiances'
        self.omeans = omeans

### Functions
# %%
def spectral_transform(da, frequnit=1, save=False,savename='fft.png',
                       savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Input time series
        Out plot of time series and spectral transform of time series """
    
    values = da.interpolate_na(dim='time').values
    times = da.coords['time'].values
    print(values)
    fft = sp.fftpack.fft(values)
    psd = np.abs(fft)**2
    freqs = sp.fft.fftfreq(len(fft),d=frequnit)
    i = freqs > 0
    print(freqs[i])
    print(psd[i])
    fig, ax = plt.subplots(2,1,figsize=(10,6))
    ax[0].plot(times, values)
    ax[0].set_title('Band ratio (CO proxy)')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Ratio')

    ax[1].plot(1/freqs[i], psd[i])
    ax[1].set_title('Vertical wind wave frequencies')
    ax[1].set_xlabel('Period / Earth days')
    ax[1].set_ylabel(r'Power spectral density / m$^2$s$^{-2}$Hz$^{-1}$')
    #ax[1].set_xlim(0,5)
    plt.subplots_adjust(hspace=0.4)
    if save==True:
        plt.savefig(savepath + savename, format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()



# %%
if __name__ == 'main':

    # %%
    # Read in data as XR DataArrays with DateTime object timestamps
    data29 = Band('band29', which_x, datadir+band29+'.DAT', datadir+band29+'.HDR', virtis_log)
    data32 = Band('band32', which_x, datadir+band32+'.DAT', datadir+band32+'.HDR', virtis_log)

    band_ratio = BandRatio('tsang_v2', data29, data32)


