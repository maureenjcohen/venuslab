""" Model outputs for direct comparison to VIRTIS (and Pioneer Venus) data."""

# %%
# Settings
lat_idx = 80  # Latitude index for zonal means and profiles

# %%
## Import packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from venusdata import *
from venuspioneer import *
from venusrossby import *

# %%
# Paths
chempath = '/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/data/Xins7.nc'
northpath = '/exomars/data/external/venus/pioneer_data/cleaned_north_probe.csv'
savepath = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/scratch_plots/'

# Classes
# %%
class VirtisModelComp(Planet):

    """ Inherit everything from the base Planet class. """

    def zonal_mean(self, key, lat, tmean=False, time_slice=-1):
        """ Compute zonal mean of variable at given time index. """

        if tmean:
            zmean = self.data[key].isel(lat=lat).mean(dim=['time_counter', 'lon'])
        else:   
            zmean = self.data[key].isel(time_counter=time_slice, lat=lat).mean(dim='lon')

        return zmean

    def calc_beta(self, lat):
        """ Rossby parameter, 2*Omega*cos(lat)/r """

        self.beta = 2*self.rotrate*np.cos(np.deg2rad(self.lats[lat]))/self.radius

    def calc_k(self, lat, wavenum=1):
        """ Calculate the wavenumber k at given latitude index. """

        circumf = 2*np.pi*self.radius*np.cos(np.deg2rad(self.lats[lat]))/wavenum
        self.k = 2*np.pi/circumf

# Functions
# %%
def rspeed(ubar, beta, k_n, kd_n):
    """ Rossby wave phase speed """

    c = ubar - (beta + ubar*kd_n**2)/(k_n**2 + kd_n**2)

    return c

# %%
def convert_t_c(t, lat=50):
    """ Convert wave period in longitude into phase speed
     at given latitude 

     Inputs: t - period in days
             lat - latitude in degrees"""
    effective_rad = 6051.8e3 * np.cos(np.deg2rad(lat))  # Effective radius at latitude in m
    circumf = 2 * np.pi * effective_rad  # Circumference at latitude in m
    wave_speed = circumf / (t*24*60*60)  # Speed in m/s
    return wave_speed

# %%
def convert_day_c(t_day=116.75):
    """ Convert solar or sidereal day into surface speed  """
    solar_day = (t_day*24*60*60)  # Venus rotation period in s
    circumf = 2 * np.pi * 6051.8e3  # Circumference at equator in m
    surface_speed = circumf / solar_day  # Surface speed in m/s
    return surface_speed

# %%
def convert_c_t(c, lat=50):
    """ Convert phase speed into period in local solar time """

    effective_rad = 6051.8e3 * np.cos(np.deg2rad(lat))  # Effective radius at latitude in m
    circumf = 2 * np.pi * effective_rad  # Circumference at latitude in m
    period = circumf / c  # Period in s
    period = period / (24*60*60)  # Period in days
    return period

# %%
if __name__ == "__main__":

    # Load in model data
    vpcm = VirtisModelComp(venusdict, 'vpcm', 'chem_day7')
    vpcm.load_file(chempath)
    vpcm.setup()

    # Load Pioneer Venus North probe data
    north_probe = Probe(northpath, 'North')

    zmzw = vpcm.zonal_mean('vitu', lat=lat_idx, tmean=True)
    vpcm.calc_beta(lat=lat_idx)
    vpcm.calc_k(lat=lat_idx, wavenum=1)

    L_d = tropical(vpcm, gmean=False, lat=lat_idx, trange=(0,-1),
             constructed=True)
    k_d = 1/L_d

    c_rossby = rspeed(ubar=zmzw, beta=vpcm.beta, k_n=vpcm.k, kd_n=k_d)

    wv_speed = convert_t_c(t=36, lat=60)
    lst_speed = convert_day_c(t_day=116.75)
    lst_period = convert_c_t(c=wv_speed+lst_speed, lat=60)

    plt.plot(zmzw, vpcm.heights, color='blue',label=f'Zonal mean zonal wind (VPCM) {vpcm.lats[lat_idx]} N')
    plt.plot(north_probe.data['WEST'], north_probe.data['ALT(KM)'], color='orange',label='Zonal wind (Pioneer Venus North probe)')
    plt.axhline(vpcm.heights[18], color='k', linestyle='dashed')
    plt.text(100, vpcm.heights[18]+2, f'{vpcm.heights[18]:.2f} km', color='k')
    plt.axhline(vpcm.heights[13], color='k', linestyle='dashed')
    plt.text(100, vpcm.heights[13]+2, f'{vpcm.heights[13]:.2f} km', color='k')
    plt.plot(np.ones_like(zmzw)*zmzw[18].values, vpcm.heights, color='green',linestyle='dashed')
    plt.text(zmzw[18].values+2, vpcm.heights[18]+12, f'{zmzw[18].values:.2f} m/s', color='green')
    plt.plot(np.ones_like(zmzw)*wv_speed, vpcm.heights, color='red',label=f'Measured wave phase speed: {wv_speed:.2f} m/s')
    plt.text(wv_speed+2, vpcm.heights[18]+5, f'{wv_speed:.2f} m/s', color='red')
    plt.legend(loc='upper right')
    plt.title('Zonal winds vs wave phase speed for 36-day wave at 60N')
    plt.xlabel('Phase speed / m s$^{-1}$')
    plt.ylabel('Height / km')
    plt.savefig(savepath + 'zwind_vs_wave_phase_speed_vpcm__pv_lat60.png', bbox_inches='tight')
    plt.show()
# %%
