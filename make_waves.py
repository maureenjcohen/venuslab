"""
Auto-generate plots for 'A travelling wave in the Venus deep atmosphere'
(Cohen et al. 2026).

Usage:
    python make_waves.py    
No inputs required as filepaths are specified in the script.
"""
# %%
# File paths
datadir = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/bands_ALL_orbits_LST/'
band29 = 'Accumulated_Grids_DATA_VI0_CO_band_2.29_interpol1_150-165K_ALLexp_LST'
band32 = 'Accumulated_Grids_DATA_VI0_CO_band_2.32_interpol1_150-165K_ALLexp_LST'
virtis_log = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/VIRTIS_log_v5.0_20130129.csv'
which_x = 'lst'
chempath = '/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/high_cadence_data/Xins_HC.nc'
savedir = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/figures/'
filetype = 'png'

# %%
# Import packages
import spectral.io.envi as envi
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy as sp
import cartopy
import cartopy.crs as ccrs
import numpy.ma as ma

# %% Definition of basic Venus model constants
# radius: km, g: m/s^2, periods: days, psurf: bar
# molmass: kg/mol, R: J/(mol K), scaleh: km
venusdict = {'radius': 6051.3, 'g': 8.87, 'rotperiod' : 243.0,
             'revperiod': 224.7, 'rotrate': 2.99e-07, 'psurf': 92.,
             'molmass': 0.04401, 'R' : 8.3143, 'RCO2' : 188.92, 'rhoconst': 65.,
             'scaleh': 16.,
             'name': 'Venus'}
# Altitude levels in km for VPCM 78 level outputs
heights78 = [9.45306290e-03, 4.83510271e-02, 1.53046221e-01, 3.73352647e-01,
       7.54337728e-01, 1.33960199e+00, 2.17016840e+00, 3.28359985e+00,
       4.71279430e+00, 6.48478937e+00, 8.61961460e+00, 1.11300869e+01,
       1.40210838e+01, 1.72868519e+01, 2.09060841e+01, 2.47483273e+01,
       2.85367489e+01, 3.21012650e+01, 3.54501762e+01, 3.85974350e+01,
       4.15684090e+01, 4.43950195e+01, 4.71000137e+01, 4.96712570e+01,
       5.20832405e+01, 5.43336220e+01, 5.64306679e+01, 5.83991661e+01,
       6.02850800e+01, 6.21196480e+01, 6.39083824e+01, 6.56521988e+01,
       6.73622131e+01, 6.90550003e+01, 7.07164383e+01, 7.23295288e+01,
       7.38933105e+01, 7.54046097e+01, 7.68734818e+01, 7.83174973e+01,
       7.97435608e+01, 8.11459122e+01, 8.25166473e+01, 8.38473129e+01,
       8.51307755e+01, 8.65320282e+01, 8.83726730e+01, 9.07071533e+01,
       9.34497375e+01, 9.65266037e+01, 9.90068893e+01, 1.00908829e+02,
       1.02824753e+02, 1.04765404e+02, 1.06736778e+02, 1.08729904e+02,
       1.10737305e+02, 1.12745575e+02, 1.14689415e+02, 1.16525398e+02,
       1.18274094e+02, 1.19964310e+02, 1.21610733e+02, 1.23218185e+02,
       1.24797585e+02, 1.26358299e+02, 1.27900208e+02, 1.29420120e+02,
       1.30924683e+02, 1.32434402e+02, 1.33977646e+02, 1.35579971e+02,
       1.37256210e+02, 1.39007202e+02, 1.40822159e+02, 1.42683762e+02,
       1.44574203e+02, 1.46479889e+02]

# Classes
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

# %%
class Simulation:
    """ Class object containing output from a VPCM simulation """

    def __init__(self, planetdict, plume_dict, run):
        """
        Initiate a Simulation object.

        Args:
            planetdict (dict): Dictionary of planet constants.
            model (str): Name of the model ('vpcm').
            run (str): Name of the run, should be the scaling factor.
        """
        self.name = planetdict['name']
        self.plumes = plume_dict
        self.run = run
        # Easter egg
        print(f'Welcome to Venus. Your lander will melt in 57 minutes.')
        print(f'This is the {self.run} dataset')
        for key, value in planetdict.items():
            setattr(self, key, value)

    def load_file(self, fn):
        """
        Load a netCDF file using the xarray package and store it in the object.

        Lists dictionary key, name, dimensions, and shape of each data cube
        and stores text in a reference list.

        Args:
            fn (str or list): Filename or list of filenames to load.
        """
        if isinstance(fn, str):
            ds = xr.open_dataset(fn, decode_cf=False)
        elif isinstance(fn, list):
            ds = xr.open_mfdataset(fn, combine='nested', concat_dim='time_counter', decode_cf=False)
        else:
            print('Improper filename input, must be string or list')
        reflist = []
        str1 = 'File contains:'
        print(str1)
        reflist.append(str1)
        for key in ds.data_vars:
            if 'long_name' in ds[key].attrs:
                keystring = f"{key}: {ds[key].long_name}, {ds[key].dims}, {ds[key].shape}"
                print(keystring)
                reflist.append(keystring)
            else:
                keystring = f"{key}: {ds[key].dims}, {ds[key].shape}"
                print(keystring)
                reflist.append(keystring)
        self.data = ds
        self.reflist = reflist

    def close(self):
        """
        Close the netCDF file packaged in the PlumeSim data object.
        """
        self.data.close()
        print('PlumeSim object associated dataset has been closed')

    def set_resolution(self):
        """
        Automatically detect file resolution and assign aesthetically
        pleasing coordinate arrays to the object for use in labelling plots.
        """
        self.lons = np.round(self.data.variables['lon'].values)
        self.lats = np.round(self.data.variables['lat'].values)
        self.tinterval = np.diff(self.data['time_counter'][0:2])[0]
        self.areas = self.data.variables['aire'].values
        if len(self.data.variables['presnivs'][:]) == 50:
            self.heights = np.array(heights50)
        elif len(self.data.variables['presnivs'][:]) == 78:
            self.heights = np.array(heights78)
        else:
            print('Altitude in km not available')       
        self.set_vertical()
        print(f"Resolution is {len(self.lats)} lats, {len(self.lons)} lons, {self.vert} levs")
        print(f'Vertical axis is {self.vert_axis}')

    def set_vertical(self):
        """
        Identify and set vertical axis and units.
        """
        self.levs = self.data['presnivs'].values
        self.vert = len(self.levs)
        self.vert_unit = self.data['presnivs'].units
        try:
            self.vert_axis = self.data['presnivs'].long_name
        except:
            self.vert_axis = self.data['presnivs'].standard_name

    def local_time(self, time_slice=-1, silent='yes'):
        
        """ A function that calculates the local time for a
        snapshot from a given timestep."""
        equator = np.argmin(np.abs(self.lats))
        # Find row number of latitude closest to 0,
        # aka the equator
        rad_toa = self.data['tops']
        # Solar radiation at top of atmosphere
        subsol = int(rad_toa[time_slice,equator,:].argmax(dim='lon').values)
        # Find column number of longitude where solar
        # radiation is currently at a maximum
        if silent=='no':
            print('Local noon is at col ' + str((subsol)))
            print('Local noon is at lon ' + str(self.lons[subsol]))
        else:
            pass
        dt = 24/len(self.lons)
        hours = np.arange(24,0,-dt)
        # Array of hour coordinates with same
        # dimension as longitude coordinates
        roll_step = int(subsol - (len(self.lons)/2))
        new_hours = list(np.roll(hours, roll_step))

        return new_hours

    def all_times(self):
        """ Create array of local times for entire time dimension"""
        time_list = []
        for t in range(0, len(self.data['time_counter'])):
            hours = self.local_time(time_slice=t, silent='yes')
            time_list.append(hours)
        time_array = np.array(time_list)

        return time_array
    
    def set_times(self):
        """ Calculate local time array for each time output
            and add to Planet object"""
        self.lst = self.all_times()

# Functions
# %%
def add_cycl_point(data, coord=None, axis=-1):
        
    """ Ripped from cartopy but removing requirement for
        data to be equally spaced"""

    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError(f'The length of the coordinate does not match '
                             f'the size of the corresponding dimension of '
                             f'the data array: len(coord) = {len(coord)}, '
                             f'data.shape[{axis}] = {data.shape[axis]}.')
        delta_coord = np.diff(coord)
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value

# %%
def vpcm_wave(simobject, key='co', time_slice=-1, lev=18, save=False,
               savename='vpcm_wave.png',
               savepath='/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/',
               sformat='png'):
    """
    Docstring for vpcm_wave
    
    :param simobject: Description
    """
    # Get data, correct for cyclic point
    cube = simobject.data[key][time_slice,lev,:,:]
    new_cube, new_lon = add_cycl_point(cube, cube.lon, -1)
    cube_name = cube.long_name or cube.name
    cube = new_cube
    level_name = np.round(simobject.heights[lev],2)
    # Orthographic projection
    ortho_proj = ccrs.Orthographic(central_longitude=0, central_latitude=0)

    # Setup the figure
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)

    # Define the mosaic layout
    layout = [
        ['map', 'right_top'],
        ['map', 'right_bottom']
    ]

    # Create the axes based on the layout

    ax_dict = fig.subplot_mosaic(
    layout, 
    width_ratios=[2, 1],
    per_subplot_kw={
        'map': {'projection': ortho_proj}
    })

    # --- Formatting the Left Figure (Map) ---
    ax_dict['map'].set_title(f"Snapshot of {key.upper()} at {level_name} km")
    ax_dict['map'].set_global()
    levels=np.linspace(cube.min(), cube.max(), 30)
    plimg = ax_dict['map'].contourf(new_lon, simobject.lats, cube, transform=ccrs.PlateCarree(), 
                        levels=levels,
                        cmap='plasma')
    ax_dict['map'].gridlines(draw_labels=True, linewidth=1.5, color='silver', alpha=0.5)
    cbar = fig.colorbar(plimg, orientation='vertical', extend='max')
    cbar.set_label('vmr')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    # --- Formatting the Right Figures ---
    ax_dict['right_top'].set_title("Right Top")
    ax_dict['right_bottom'].set_title("Right Bottom")

    # Show the plot
    if save==True:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()  

# %%
if __name__ == "__main__":

    # Load model data
    vpcm = Simulation(venusdict, 'vpcm', 'chem_hc')
    vpcm.load_file(chempath)
    vpcm.set_resolution()
    vpcm.set_times()