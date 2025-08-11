""" Script to load a NETCDF file of an atmospheric simulation and
calculate the angular momentum budget                           

Assumes the data array names are from the Venus PCM, e.g. vitu, vitv """

""" Top-line variables """
fpath = '/exomars/data/internal/working/'
# Full filepath of NETCDF file
planet_radius = 6051300
# Planet radius
planet_omega = 2.99e-07 
# Planet rotation rate
molmass = 0.04401
# Molar mass
R_gas = 8.3143
# Gas constant
heights50 = [0.,  0.05,  0.2,  0.4,  0.8,  1.3,  2.2,  3.3,  4.7,  6.5,  8.6,
       11.1, 14., 17.3, 20.9, 24.7, 28.5, 32.1, 35.4, 38.6, 41.6, 44.4,
       47.1, 49.7, 52.1, 54.3, 56.4, 58.4, 60.3, 62.1, 63.9, 65.6, 67.4,
       69., 70.7, 72.3, 73.9, 75.4, 76.9, 78.4, 79.8, 81.2, 82.6, 84.,
       85.3, 86.8, 88.7, 91.2, 94.1, 97.] 
# Altitude in km of each pressure level

""" Imports """
# %%
import xarray as xr
import numpy as np


""" Class definitions"""
# %%
class Simulation:
    """ Class object containing output from a GCM simulation """

    def __init__(self):
        """ Instantiate a Simulation object """
        self.radius = planet_radius
        self.omega = planet_omega
        self.molmass = molmass
        self.R = R_gas
        self.heights = heights50

    def load_file(self, fn):
        """ Open one or more files as xarray DataSet 
            Attach to Simulation object under .data attribute """
        if isinstance(fn, str):
            ds = xr.open_dataset(fn, decode_cf=False)
        elif isinstance(fn, list):
            ds = xr.open_mfdataset(fn, combine='nested', concat_dim='time_counter', decode_cf=False)
        else:
            print('Improper filename input, must be string or list')
        self.data = ds

    def set_resolution(self):
        """ Extract longitude and latitude values in degrees 
            Convert to radians and store                      """
        self.lons = np.round(self.data.variables['lon'].values)
        self.lats = np.round(self.data.variables['lat'].values)
        self.rad_lons = np.deg2rad(self.lons)
        self.rad_lats = np.deg2rad(self.lats)

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

    def calc_rho(self):
        """ Calculate density of atmosphere using ideal gas law approximation """
        rho = (self.data['pres'][:]*self.molmass)/(self.R*self.data['temp'][:])
        self.rho = rho

# %%
class AngularMomentumBudget(Simulation):
    """ Zonal mean angular moment budget in spherical coordinates
        Eq. 2 from Sergeev et al. 2022, doi: 10.3847/PSJ/ac83be """

    def zonal_mean(self, cube):
        """ Calculate zonal mean of input data cube """
        return cube.mean(dim='lon')
    
    def time_mean(self, cube):
        """ Calculate time mean of input data cube """
        return cube.mean(dim='time_counter')
    
    def calc_d_phi(self):
        """ Calculate discrete difference of latitude in radians """
        self.d_phi = self.rad_lats.differentiate()

    def calc_d_z(self):
        """ Calculate discrete difference of altitude in m """
        self.d_z = np.gradient(self.heights*1e3)
    
    def calc_rho_v(self):
        """ Calculate density x meridional wind """
        self.rho_v = self.rho * self.data['vitv']

    def calc_rho_w(self):
        """ Calculate density x vertical wind (in m/s) """
        self.rho_w = self.rho * self.data['vitwz']

    def calc_cos_lat(self):
        """ Calculate cosine of latitudes (in radians) """
        self.cos_lat = xr.ufuncs.cos(self.rad_lats)

    def calc_r_cos_lat(self):
        """ Calculate radius x cosine of latitudes (in radians) """
        self.r_cos_lat = self.radius * self.cos_lat

    def calc_ang_mom(self):
        """ Calculate the angular momentum per unit mass 
            m = [u + Omega*r*cos(lat)]*r*cos(lat) """
        self.ang_mom = (self.data['vitu'] + self.omega*self.r_cos_lat) * self.r_cos_lat

    def calc_rho_v_tzmean(self):
        """ Time mean and zonal mean of density x meridional wind """
        self.rho_v_tzmean = self.zonal_mean(self.time_mean(self.rho_v))

    def calc_rho_w_tzmean(self):
        """ Time mean and zonal mean of density x vertical wind """
        self.rho_w_tzmean = self.zonal_mean(self.time_mean(self.rho_w))

    def calc_ang_mom_tzmean(self):
        """ Time mean and zonal mean of angular moment per unit mass """
        self.ang_mom_tzmean = self.zonal_mean(self.time_mean(self.ang_mom))

    def calc_v_star_tmean(self):
        """ Time mean of eddy rho * v """
        self.v_star_tmean = self.time_mean(self.rho_v - self.zonal_mean(self.rho_v))

    def calc_w_star_tmean(self):
        """ Time mean of eddy rho * w """
        self.w_star_tmean = self.time_mean(self.rho_w - self.zonal_mean(self.rho_w))

    def calc_m_star_tmean(self):
        """ Time mean of eddy angular momentum """
        self.m_star_tmean = self.time_mean(self.ang_mom - self.zonal_mean(self.ang_mom))

    def calc_v_prime_m_prime(self):
        """ Time mean of prime of rho_v * prime of angular momentum """
        v_prime = self.rho_v - self.time_mean(self.rho_v)
        m_prime = self.ang_mom - self.time_mean(self.ang_mom)
        self.v_prime_m_prime_tmean = self.time_mean(v_prime * m_prime)

    def calc_w_prime_m_prime(self):
        """ Time mean of prime of rho_w * prime of angular momentum """
        w_prime = self.rho_w - self.time_mean(self.rho_w)
        m_prime = self.ang_mom - self.time_mean(self.ang_mom)
        self.w_prime_m_prime_tmean = self.time_mean(w_prime * m_prime)

    def mean_horizontal_term(self):
        """ Mean horizontal term """
        self.mh_term = - (self.rho_v_tzmean / self.radius) * (self.ang_mom_tzmean.differentiate(coord='lats')/self.d_phi)

    def mean_vertical_term(self):
        """ Mean vertical term """
        self.mv_term = - self.rho_w_tzmean * (self.ang_mom_tzmean.differentiate(coord='lats') / self.d_z)

    def stat_horizontal_term(self):
        """ Stationary horizontal term """
        zm_term = self.zonal_mean(self.v_star_tmean * self.m_star_tmean) * self.cos_lat
        d_zm_term = np.gradient(zm_term)
        self.sh_term = - d_zm_term / (self.r_cos_lat * self.d_phi)

    def stat_vertical_term(self):
        """ Stationary vertical term """
        zm_term = self.zonal_mean(self.w_star_tmean * self.m_star_tmean) * (self.radius**2)
        d_zm_term = np.gradient(zm_term)
        self.sv_term = - d_zm_term / ((self.radius**2) * self.d_z )

    def trans_horizontal_term(self):
        """ Transient horizontal term """
        zm_term = self.zonal_mean(self.v_prime_m_prime_tmean) * self.cos_lat
        d_zm_term = np.gradient(zm_term)
        self.th_term = - d_zm_term / (self.r_cos_lat * self.d_phi)

    def trans_vertical_term(self):
        """ Transient vertical term """
        zm_term = self.zonal_mean(self.w_prime_m_prime_tmean) * (self.radius**2)
        d_zm_term = np.gradient(zm_term)
        self.tv_term = - d_zm_term / ((self.radius**2) * self.d_z)