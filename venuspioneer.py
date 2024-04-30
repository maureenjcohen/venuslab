""" Module for processing Pioneer Venus data """

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

## File paths ##
# %%
# Old data
north_alts = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/sas_north_atm_state.csv'
night_alts = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/sas_night_atm_state.csv'
day_alts = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/sas_day_atm_state.csv'
north_data = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/north_probe.csv'
night_data = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/night_probe.csv'
day_data = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/day_probe.csv'
savepath = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/'

# %%
# Cleaned data
day_probe = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/cleaned_day_probe.csv'
night_probe = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/cleaned_night_probe.csv'
north_probe = '/exomars/data/analysis/volume_8/mc5526/pioneer_data/cleaned_north_probe.csv'

# %%
def clean_data(datapath, altspath, startrow, savename):
    """ Pretty much one-time use function to clean and reorganise data
        Pioneer Venus probes. Keeping for reference.         
        Day: startrow = 23
        Night: startrow = 26
        North: startrow =  25                    """

    probe_data = pd.read_csv(datapath, usecols=['TIME','DOWN','WEST','NORTH'], sep='\s+')
    altitude_data = pd.read_csv(altspath, skiprows=startrow, usecols=['GRT(SEC)','ALT(KM)','P(BARS)','T(DEG K)','RHO(KG/M3)'])
    # Read in probe data and altitude data from separate csvs
    alts_renamed = altitude_data.rename(columns={'GRT(SEC)':'TIME'})
    # Rename time column in altitudes DF to have same name as probe data DF
    merged_df = pd.merge(probe_data, alts_renamed, on='TIME', how='outer')
    # Merge the two DFs on TIME column
    # Altitude and wind sampling are reported at different times so interpolation is needed
    orig = datetime.datetime(year=1978, month=12, day=9)
    # From DBLI experiment documentation: origin date is December 9, 1978
    # Time is in seconds since the origin date
    merged_df['TIME'] = pd.to_datetime(merged_df['TIME'], unit='s', origin=orig)
    # Change TIME column to a datetime object reported as seconds since Dec 9, 1978
    interp = merged_df.set_index('TIME')[['ALT(KM)','P(BARS)','T(DEG K)','RHO(KG/M3)']].interpolate(method='time').reset_index()
    # New DF with just TIME column and altitudes interpolated for all timestamps
    merged_df[['ALT(KM)','P(BARS)','T(DEG K)','RHO(KG/M3)']] = interp[['ALT(KM)','P(BARS)','T(DEG K)','RHO(KG/M3)']]
    # Put the interpolated/completed altitudes column back in original DF
    cleaned = merged_df.dropna(axis=0, how='any').reset_index(drop=True)
    # Now drop rows containing NaNs, aka rows without wind data
    cleaned.to_csv(savename)

## Analytical stuff
# %%
class Probe:
    """ Holds data from one of the Pioneer Venus descent probes """

    def __init__(self, probepath, name):
        self.name = name # Name of probe, i.e. 'Day','Night','North'
        if name=='North':
            self.lat = 60
        else:
            self.lat = 30

        data = pd.read_csv(probepath, sep=',')
        self.data = data # Add DataFrame to object
        self.RCO2 = 287.05
        self.g = 8.87 
        self.radius = 6051.3 # km

    def profile(self, key):
        if key=='Zonal wind':
            cube = self.data['WEST'].values
            unit = 'm/s'
        elif key=='Meridional wind':
            cube = self.data['NORTH'].values
            unit = 'm/s'
        elif key=='Descent velocity':
            cube = self.data['DOWN'].values
            unit = 'm/s'
        elif key=='Temperature':
            cube = self.data['T(DEG K)']
            unit = 'K'
        elif key=='Pressure':
            cube = self.data['P(BARS)']
            unit = 'bar'
        elif key=='Density':
            cube = self.data['RHO(KG/M3)']
            unit = 'kg/m3'
        elif key=='Potential temperature':
            if not hasattr(self,'theta'):
                self.calc_theta()
            cube = self.theta
            unit = 'K'
        elif key=='BV frequency':
            if not hasattr(self,'bv'):
                self.calc_bv_freq()
            cube = self.bv
            unit = 's-1'
        elif key=='Rotation period':
            if not hasattr(self,'period'):
                self.calc_omega()
            cube = self.period
            unit = 'Earth days'
        else:
            print('Key not recognised.')

        fig, ax = plt.subplots(figsize=(8,6))
        plt.plot(cube, self.data['ALT(KM)'].values)
        plt.title(f'{key} profile from {self.name} probe')
        plt.xlabel(f'{key} / {unit}')
        plt.ylabel('Altitude / km')
        plt.show()

    def calc_cp(self):
        """ Formula LMDZ Venus uses to vary the specific heat with temperature"""
        cp0 = 1000 # J/kg/K
        T0 = 460 # K
        v0 = 0.35 # exponent
        cp = cp0*(self.data['T(DEG K)'][:]/T0)**v0
        self.cp = cp
        self.cp0 = cp0
        self.T0 = T0
        self.v0 = v0

    def calc_theta(self):
        """ Formula LMDZ Venus uses for potential temperature to account for
        specific heat capacity varying with height.
        See Lebonnois et al 2010.   """
        if not hasattr(self, 'cp'):
            self.calc_cp()
        p0 = self.data['P(BARS)'].values[-1]
        theta_v = (self.data['T(DEG K)'][:]**self.v0 +
                   self.v0*(self.T0**self.v0)*(np.log((p0/self.data['P(BARS)'][:])**(self.RCO2/self.cp0))))
        theta = theta_v**(1/self.v0)
        self.theta = theta

    def calc_omega(self):
        """ Calculate effective rotation rate of atmosphere based on
            zonal wind speed at each altitude                   """
        circumf = 2*np.pi*((self.radius + self.data['ALT(KM)'].values)*1000)
        period = (circumf/np.abs(self.data['WEST'].values))
        omega = (2*np.pi)/period
        period_days = period/(60*60*24)
        self.omega = omega
        self.period = period_days

    def calc_bv_freq(self):
        if not hasattr(self,'theta'):
            self.calc_theta()

        th_dz = np.gradient(self.theta)/np.gradient(self.data['ALT(KM)']*1000)
        root_term = self.g*th_dz/self.theta
        freq = np.sqrt(root_term)
        self.bv = freq

    def calc_coriolis(self):
        if not hasattr(self,'omega'):
            self.calc_omega()
        lat_rad = np.deg2rad(self.lat)
        f = 2*self.omega*np.sin(lat_rad)
        self.coriolis = f

    def calc_scale_height(self):
        numerator = (1.38e-23)*self.data['T(DEG K)'].values[:]
        denom = 44.01*(1.67e-27)*self.g
        H = numerator/denom
        self.scale_h = H

    def calc_rossby_radii(self):
        if not hasattr(self,'bv'):
            self.calc_bv_freq()
        if not hasattr(self,'coriolis'):
            self.calc_coriolis()
        if not hasattr(self,'scale_h'):
            self.calc_scale_height()

        extra_r = self.bv*self.scale_h/self.coriolis
        self.extra_r = extra_r

        beta = 2*self.omega/(self.radius*1000) 
        trop_r = np.sqrt(self.bv*self.scale_h/(2*beta))
        self.trop_r = trop_r



# %%
def all_probes(probelist, key):
    """ Plot wind measurements from multiple probes in one figure
        probelist: list of Probe objects (Day, Night, North)
        key: variable to plot (Zonal wind, Meridional wind, Descent velocity)"""
    
    colors=['tab:blue','tab:green','tab:orange']
    fig, ax = plt.subplots(figsize=(6,8))
    for ind, probe in enumerate(probelist):
        if key=='Zonal wind':
            cube = probe.data['WEST'].values
        elif key=='Meridional wind':
            cube = probe.data['NORTH'].values
        elif key=='Descent velocity':
            cube = probe.data['DOWN'].values
        else:
            print('Key not recognised.')

        plt.plot(cube, probe.data['ALT(KM)'].values, color=colors[ind], label=probe.name)
    plt.title(f'{key} profiles from Pioneer Venus descent probes')
    plt.xlabel(f'{key} / m/s')
    plt.ylabel('Altitude [km]')
    plt.legend()
    plt.show()

# %%
