""" Extract VIRTIS data from ENVI files, preprocess, and write to netcdf"""

# %%
# Import packages
import spectral.io.envi as envi
import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy as sp
import cartopy
import cartopy.crs as ccrs

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
            x_values = np.concatenate([lst2[::-1],lst1[::-1]]) # Venus go brrrackwards
            #x_values = np.linspace(0,24,240)
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

    def count(self):
        """ Count number of observations per pixel (i.e., non-nan values) 
            Return indices of pixels with max counts            """
        counts = np.count_nonzero(~np.isnan(self.da), axis=-1)
        y_max = np.where(counts==np.max(counts))[0]
        x_max = np.where(counts==np.max(counts))[1]
        self.y_max = y_max
        self.x_max = x_max
        self.counts = counts
        ix, jx = np.unravel_index(self.counts.argsort(axis=None), self.counts.shape)
        self.most_counts = list(zip(ix,jx))[::-1]
  
    def orbit_means(self,trange=(0,None)):
        """ Time series of orbit means for all """

        vals_list = []
        tvals_list = []
        for orbit in self.unique_orbits[trange[0]:trange[1]]:
            print('Processing ' + str(orbit))
            subset = self.ds.where(self.ds['ORBIT'] == orbit)
            val = np.nanmean(subset[self.name].values, axis=-1)
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
        omeans.attrs['description'] = 'Band radiance meaned per orbit'
        omeans.attrs['unit'] = 'W m-2 str-1 s-1'
        self.omeans = omeans

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
        
        if self.name == 'tsang_v2':
            self.da = top_band.da[:,:,indices_top]/bottom_band.da[:,:,indices_bottom]
        elif self.name == 'barstow':
            top_in_cm2 = self.convert_m2_to_cm2(top_band.da[:,:,indices_top])
            bottom_in_cm2 = self.convert_m2_to_cm2(bottom_band.da[:,:,indices_bottom])
            numerator = bottom_in_cm2 - 738.57*(top_in_cm2**2) - 0.31681*top_in_cm2 - 9.8043e-09
            denominator = -417.62*(top_in_cm2**2) - 0.22764*top_in_cm2 + 8.9506e-09
            self.da = 35*np.exp(numerator/denominator)
        self.unique_orbits = common_orbs
        self.observations = common_obs

        mask = top_band.ds['PRODUCT_ID'].isin(self.observations)
        ds = top_band.ds.where(mask, drop=True)
        ratio_ds = ds.drop_vars(top_band.name)
        ratio_ds['RATIO'] = self.da
        self.ds = ratio_ds

    def convert_m2_to_cm2(self, arr):
        """ Convert radiance array from W/m2 to W/cm2
            For Barstow 2012 formula """
    
        rad = arr/10000

        return rad

    def count(self):
        """ Count number of observations per pixel (i.e., non-nan values) 
            Return indices of pixels with max counts            """
        counts = np.count_nonzero(~np.isnan(self.da), axis=-1)
        y_max = np.where(counts==np.max(counts))[0]
        x_max = np.where(counts==np.max(counts))[1]
        self.y_max = y_max
        self.x_max = x_max
        self.counts = counts
        ix, jx = np.unravel_index(self.counts.argsort(axis=None), self.counts.shape)
        self.most_counts = list(zip(ix,jx))[::-1]
  
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
def spectral_transform(da, ycoord, xcoord, trange, frequnit=1, save=False,savename='fft.png',
                       savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Input time series
        Out plot of time series and spectral transform of time series """
    
    values = da[ycoord, xcoord, trange[0]:trange[1]].interpolate_na(dim='time').values
    times = da[ycoord, xcoord, trange[0]:trange[1]].coords['time'].values

    fft = sp.fftpack.fft(values)
    psd = np.abs(fft)**2
    freqs = sp.fft.fftfreq(len(fft),d=frequnit)
    i = freqs > 0

    fig, ax = plt.subplots(2,1,figsize=(10,6))
    ax[0].plot(times, values, marker='o')
    ax[0].set_title(f'CO at {np.round(da.coords['lat'].values[ycoord],2)} deg lat, {np.round(da.coords[which_x].values[xcoord],2)} {which_x}')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('CO band ratio')

    ax[1].plot(1/freqs[i], psd[i])
    ax[1].set_title('Spectral transform')
    ax[1].set_xlabel('Period / Earth days')
    ax[1].set_ylabel(r'Power spectral density')
    #ax[1].set_xlim(0,5)
    plt.subplots_adjust(hspace=0.4)
    if save==True:
        plt.savefig(savepath + savename, format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return 1/freqs[i], psd[i]

# %%
def mean_spectrum(periods, mean_psd, std_psd, save=False, savename='mean_psd.png',
                  savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Plot mean power spectral density with std dev shading """
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(periods, mean_psd, marker='o', color='blue', label='Mean PSD')
    ax.fill_between(periods, mean_psd - std_psd, mean_psd + std_psd, color='blue', alpha=0.3, label='1 Std Dev')
    ax.set_title('Mean Power Spectral Density of Top 30 CO time series')
    ax.set_xlabel('Period / Earth days')
    ax.set_ylabel('Power Spectral Density')
    ax.legend()
    if save==True:
        plt.savefig(savepath + savename, format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()          

# %%
def hovmoeller(inputda, lat=30, trange=(10,80), xrange=(60,180), levels=np.arange(1.5,2.3,0.1),
               save=False, savename='VIRTIS_hovmoeller.png',
               savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):
    """ Plot Hovmoeller diagram of input DataArray """

    da = inputda[lat,xrange[0]:xrange[1],trange[0]:trange[1]]
    print(da.shape)
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.contourf(da.transpose(), cmap='plasma', levels=levels, origin='lower')
    ax.set_title(f'Band ratio (CO proxy) at {np.round(da.coords['lat'].values,0)} lat')
    ax.set_ylabel('Orbits (approx. Earth days)')
    ax.set_xlabel('Local solar time / hours')
    ax.set_xticks(np.arange(0,da.shape[0],10))
    ax.set_xticklabels(np.round(da.coords[which_x].values[::10],0), rotation=45)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(da.attrs['unit'])
    if save==True:
        plt.savefig(savepath + savename, format='png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# %%
def sphere(inputda, levels, i=0, j=-90,
           save=False, savename='VIRTIS_lons_sphere_southpole.png',
           savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'):

    """ Use with longitude projection only!"""
    if inputda.coords[which_x].name == 'lon':
        x_axis = inputda.coords[which_x]
    elif inputda.coords[which_x].name == 'lst':
        x_axis = np.arange(0,240)*(360/240)

    ortho = ccrs.Orthographic(central_longitude=i, central_latitude=j)
    # Specify orthographic projection centered at lon/lat i, j
    fig = plt.figure(figsize=(8, 6))
    #fig.patch.set_facecolor('black') # Background colour
    ax = plt.axes(projection=ortho)
    ax.set_global()
    # Create the figure
    plimg = ax.contourf(x_axis, inputda.coords['lat'], inputda, transform=ccrs.PlateCarree(), 
                        levels=levels,
                        cmap='jet')
    ax.set_title(f'{inputda.coords['time'].values}')
    gl = ax.gridlines(draw_labels=True, linewidth=1.5, color='silver', alpha=0.5, y_inline=True, x_inline=False)
    gl.xlocator = mticker.FixedLocator([0, 180])
    if inputda.coords[which_x].name == 'lst':
        gl.xformatter = mticker.FuncFormatter(lambda x_val, tick_pos: (x_val/15)+12)
    gl.ylocator = mticker.FixedLocator([-80,-60,-40])
    gl.ylabel_style = {'rotation':45}
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('Ratio', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
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
    # %%
    # Generate some secondary data we want
    band_ratio.count()
    band_ratio.orbit_means()

    # %%
    # Plot time series of band ratio at a lat and x-coord value with the longest time series
    # + perform spectral transform
    spectral_transform(band_ratio.omeans, band_ratio.y_max[0], band_ratio.x_max[0], (10,80))
    # Hovmoeller plot of nightside at 30 deg south
    hovmoeller(band_ratio.omeans, lat=30, trange=(10,80), xrange=(60,180))

    # %%
    # Create spectral transforms from rolling mean of top 30 time series with most counts
    roll3 = band_ratio.omeans.rolling(time=3, center=True).mean()
    psds = []
    for item in band_ratio.most_counts[:30]:
        period, psd = spectral_transform(roll3, item[0], item[1], (10,80), save=False, savename=f'VIRTIS_fft_roll3_lat{item[0]}_x{item[1]}.png')
        psds.append(psd)
    mean_psd = np.mean(np.array(psds), axis=0)
    std_psd = np.std(np.array(psds), axis=0)
    mean_spectrum(period, mean_psd, std_psd, save=False, savename='VIRTIS_mean_psd_roll3_top30.png')

    # %%