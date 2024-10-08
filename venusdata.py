""" Data organisation"""

# %%
## Import packages
#import netCDF4 as nc
import xarray as xr
import numpy as np

# %%
## Definition of basic Venus model constants
## Radius in km, gravitational constant in m/s2, periods in Earth days,
## rotation rate in s-1, surface pressure in bars, molecular mass in kg/mol
## gas constant for CO2 in SI units, density in kg/m3,
## scale height in km
venusdict = {'radius': 6051.3, 'g': 8.87, 'rotperiod' : 243.0, 
             'revperiod': 224.7, 'rotrate': 2.99e-07, 'psurf': 92.,
             'molmass': 0.04401, 'R' : 8.3143, 'RCO2' : 188.92, 'rhoconst': 65.,
             'scaleh': 16.,
             'name': 'Venus'}
## Heights in km for 50-level sims from Lebonnois 2010
heights50_old = [0.00, 0.03, 0.12, 0.32, 0.68, 1.23, 2.03, 3.10, 4.50, 6.23, 8.35,
               10.8, 13.7, 17.0, 20.7, 24.6, 28.3, 31.9, 35.2, 38.4, 41.4, 44.2,
               46.9, 49.5, 51.9, 54.1, 56.2, 58.1, 60.1, 61.9, 63.7, 65.5, 67.2,
               68.8, 70.5, 72.2, 73.8, 75.5, 77.1, 78.7, 80.2, 81.8, 83.3, 84.8,
               86.2, 87.8, 90.1, 92.9, 94.9, 101.]
## Heights in km calculated from global area-weighted long-term mean of geopotential height
heights50 = [0.,  0.05,  0.2,  0.4,  0.8,  1.3,  2.2,  3.3,  4.7,  6.5,  8.6,
       11.1, 14., 17.3, 20.9, 24.7, 28.5, 32.1, 35.4, 38.6, 41.6, 44.4,
       47.1, 49.7, 52.1, 54.3, 56.4, 58.4, 60.3, 62.1, 63.9, 65.6, 67.4,
       69., 70.7, 72.3, 73.9, 75.4, 76.9, 78.4, 79.8, 81.2, 82.6, 84.,
       85.3, 86.8, 88.7, 91.2, 94.1, 97.] 
isentropes70 = [283] + list(np.arange(290,980,10))

### Planet class object definition ###
### Manages planet configuration data and simulation output data.

# %%
class Planet:
    """ A Planet object which contains the output data for a simulation"""
    
    def __init__(self, planetdict, model, run):
        """ Initiates a Planet object using the input dictionary of planet constants,
        the name of the model, and the name of the run. 
        Model names: vpcm or oasis
        Run names: isentropes, isobars, or altitudes """

        self.name = planetdict['name'] # dictionary
        self.model = model # string
        self.run = run # string
        print(f'Welcome to Venus. Your lander will melt in 57 minutes.')
        print(f'This is the {self.run} dataset created by {self.model.upper()}')
        for key, value in planetdict.items():
            setattr(self, key, value)

    def identify(self):
        print(f'This is the {self.run} dataset created by {self.model.upper()}')

    def load_file(self, fn):
        """ Loads a netCDF file using the netCDF4 package and stores in object
            Lists dictionary key, name, dimensions, and shape
            of each data cube and stores text in a reference list"""
        ds = xr.open_dataset(fn, decode_cf=False)
        reflist = []
        str1 = 'File contains:'
        print(str1)
        reflist.append(str1)
        for key in ds.data_vars:
            if 'long_name' in ds[key].attrs:
                keystring = key + ': ' + ds[key].long_name + ', ' + \
                      str(ds[key].dims) + ', ' + \
                      str(ds[key].shape)
                print(keystring)
                reflist.append(keystring)
            else:
                keystring = key + ': ' + str(ds[key].dims) + ', ' \
                      + str(ds[key].shape)
                print(keystring)
                reflist.append(keystring)
        self.data = ds
        self.reflist = reflist

    def close(self):
        """ Closes netCDF file packaged in Planet data object """
        self.data.close()
        print('Planet object associated dataset has been closed')

    def contents(self):
        """ Prints reference list for easy formatted oversight of file contents"""
        print(*self.reflist, sep='\n')

    def set_resolution(self):
        """ Automatically detects file resolution and assigns aesthetically
        pleasing coordinate arrays to object for use in labelling plots"""
        if self.model=='vpcm':
            self.lons = np.round(self.data.variables['lon'].values)
            self.lats = np.round(self.data.variables['lat'].values)
            self.tinterval = np.diff(self.data['time_counter'][0:2])[0]
            self.areas = self.data.variables['aire'].values
            if len(self.data.variables['presnivs'][:]) == 50:
                self.heights = np.array(heights50)
            else:
                print('Altitude in km not available')
        elif self.model=='oasis':
            self.lons = self.data['lon'].values
            self.lats = self.data['lat'].values
            self.tinterval = np.diff(self.data['time_counter'][0:2].values)[0]
            self.area_weights()
        
        self.set_vertical()
        print('Resolution is ' +  str(len(self.lats)) + ' lats, '
        + str(len(self.lons)) + ' lons, ' + str(self.vert) + ' levs')
        print(f'Vertical axis is {self.vert_axis}')

    def set_vertical(self):
        """ Identify and set vertical axis and units """
        self.levs = self.data['presnivs'].values
        self.vert = len(self.levs)
        self.vert_unit = self.data['presnivs'].units
        self.vert_axis = self.data['presnivs'].long_name
        if self.model=='oasis' and self.run=='altitudes':
            self.levs = self.data['presnivs'].values*1e3
            self.vert_unit = 'km'

    def area_weights(self):
        """ Calculate area weights if not included in output, e.g. for OASIS data"""
        xlon, ylat = np.meshgrid(self.lons, self.lats)
        dlat = np.deg2rad(np.abs(np.gradient(ylat, axis=0)))
        dlon = np.deg2rad(np.abs(np.gradient(xlon, axis=1)))
        rad = self.radius*1e3
        dy = dlat*rad
        dx = dlon*rad*np.abs(np.cos(np.deg2rad(ylat)))
        areas = dy*dx
        self.areas = areas
        self.dy = dy
        self.dx = dx

    def calc_cp(self):
        """ Formula VPCM uses to vary the specific heat with temperature"""
        cp0 = 1000 # J/kg/K
        T0 = 460 # K
        v0 = 0.35 # exponent
        cp = cp0*((self.data['temp'][:]/T0)**v0)
        self.cp = cp
        self.cp0 = cp0
        self.T0 = T0
        self.v0 = v0

    def calc_theta(self, pref=9.2e6):
        """ Formula VPCM uses for potential temperature to account for
        specific heat capacity varying with height.
        See Lebonnois et al 2010.   """
        if self.model=='vpcm':
            if not hasattr(self, 'cp'):
                self.calc_cp()
            p0 = pref
            theta_v = (self.data['temp'][:]**self.v0 +
                    self.v0*(self.T0**self.v0)*(np.log((p0/self.data['pres'][:])**(self.RCO2/self.cp0))))
            theta = theta_v**(1/self.v0)
            self.theta = theta
        elif self.model=='oasis':
            p0 = 100000
            theta = self.data['temp']*((p0/self.data['pres'])**(self.RCO2/900))
            self.theta = theta

    def calc_rho(self):
        """ Calculate density of atmosphere using ideal gas law approximation """
        rho = (self.data['pres'][:]*self.molmass)/(self.R*self.data['temp'][:])
        self.rho = rho

    def calc_w(self):
        """ Calculate vertical velocity in m/s from Pa/s. """
        w_wind = -(self.data['vitw'][:]*self.data['temp'][:]*self.RCO2)/(self.data['pres'][:]*self.g)
        self.w_wind = w_wind

    def set_times(self):
        """ Calculate local time array for each time output
            and add to Planet object"""
        self.local_time = all_times(self)

    def total_area(self):
        """ Calculate total surface area in m2"""
        if self.model=='vpcm':
            self.area = np.sum(self.data['aire'][:])
        elif self.model=='oasis':
            self.area = np.sum(self.areas)

    def alt_levels(self):
        """ Approximate altitude levels through long-term mean of
        global area-weighted mean geopotential height above surf """
        if self.model=='vpcm':
            time_mean_geop = self.data['geop'].mean(dim='time_counter')/self.g
            if not hasattr(self, 'area'):
                self.total_area() # Get total surface area if not done already
            glob_mean = np.sum(self.data['aire'].values*time_mean_geop.values, axis=(1,2))/self.area.values
            self.heights = glob_mean/1000


    def setup(self):
        self.set_resolution()
        self.total_area()

# %%
def local_time(plobject, time_slice=-1, silent='no'):
    
    """ A function that calculates the local time for a
    snapshot from a given timestep."""
    equator = np.argmin(np.abs(plobject.lats))
    # Find row number of latitude closest to 0,
    # aka the equator
    rad_toa = plobject.data['tops']
    # Solar radiation at top of atmosphere
    subsol = np.argmax(rad_toa[time_slice,equator,:])
    # Find column number of longitude where solar
    # radiation is currently at a maximum
    if silent=='no':
        print('Local noon is at col ' + str(subsol))
        print('Local noon is at lon ' + str(plobject.lons[subsol]))
    else:
        pass
    dt = 24/len(plobject.lons)
    hours = np.arange(0,24,dt)
    # Array of hour coordinates with same
    # dimension as longitude coordinates
    roll_step = int(subsol - (len(plobject.lons)/2))
    new_hours = list(np.roll(hours, roll_step))

    return new_hours

# %%
def all_times(plobject):

    """ Create array of local times for entire time dimension"""
    time_list = []
    for t in range(0, len(plobject.data['time_counter'])):
        hours = local_time(plobject, time_slice=t, silent='yes')
        time_list.append(hours)
    time_array = np.array(time_list)

    return time_array


# %%
def local_mean(plobject, key, lev, trange):

    """ Calculate the mean of the input field with respect
        to the local time, i.e. mean over longitudes   """
    try:
        plobject.local_time
    except AttributeError:
        print('Local times not calculated')
        print('Calculating local times. This will take a minute.')
        plobject.set_times()

    data = plobject.data[key][trange[0]:trange[1],lev,:,:]
    times_needed = plobject.local_time[trange[0]:trange[1]]
    data_list = []
    for t in range(0,len(trange)):
        noon_col = np.where(times_needed[t]==12.0[0][0])
        shifted_data = np.roll(data[t,:,:], -noon_col, axis=1)
        data_list.append(shifted_data)

    shifted_array = np.array(data_list)
    meaned_data = np.mean(shifted_array, axis=0)

    return meaned_data
    
