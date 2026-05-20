""" Compare spectral band projections produced by IDL and Python versions of the code"""

# %%
# Imports
import os
import numpy as np
import pandas as pd
import xarray as xr
import spectral.io.envi as envi
from spectral import *
import matplotlib.pyplot as plt

from virtis_xarray import Band, datadir, band32, band29, virtis_log

# %%
# Filepaths
idl_maindir = '/exomars/data/external/venus/venus_express/VIRTIS/VIRTIS_M_spectral_band_projections_EXOMARS/'
idl_229 = 'Accumulated_Grids_VEX-V-VIRTIS-2-3-EXT1-V2.0_VI_CO_band_2.29_interp_150-165K_ALLexp_LON_ALLorbs'
idl_232 = 'Accumulated_Grids_VEX-V-VIRTIS-2-3-EXT1-V2.0_VI_CO_band_2.32_interp_150-165K_ALLexp_LON_ALLorbs'

python_maindir = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/python_band_extraction/'
python_229 = 'co_229um_20260514T0716.npz'
python_232 = 'co_232um_20260514T0757.npz'


# %%
data32 = Band('band32', 'lon', idl_maindir+idl_232+'.DAT', idl_maindir+idl_232+'.HDR', virtis_log)
data29 = Band('band29', 'lon', idl_maindir+idl_229+'.DAT', idl_maindir+idl_229+'.HDR', virtis_log)

# %%
idl32_mean = data32.ds.mean(dim='time')
idl29_mean = data29.ds.mean(dim='time')

# %%
py = xr.open_dataset('/exomars/projects/mc5526/VPCM_deep_atmos_CO/python_band_extraction/co_window_mode1_20260516T1811.nc', decode_times=False)

# %%
plt.contourf(idl29_mean['lon'],idl29_mean['lat'],idl29_mean['band29'], cmap='plasma', levels=np.arange(0.015,0.120,0.015))
plt.colorbar()
plt.show()
#plt.savefig('/exomars/projects/mc5526/VPCM_deep_atmos_CO/idl_mean_co_band_229_PCAL.png')

# %%
plt.contourf(idl32_mean['lon'],idl32_mean['lat'],idl32_mean['band32'], cmap='plasma', levels=np.arange(0,0.07,0.01))
plt.colorbar()
plt.show()
#plt.savefig('/exomars/projects/mc5526/VPCM_deep_atmos_CO/idl_mean_co_band_232_PCAL.png')

# %%
plt.contourf(py['lon'], py['lat'], py['radiance'][4,:,:], cmap='plasma', levels=np.arange(0,0.07,0.01))
plt.colorbar()
plt.show()
#plt.savefig('/exomars/projects/mc5526/VPCM_deep_atmos_CO/python_mean_co_band_232_v4.png')
# %%
plt.contourf(py['lon'], py['lat'], py['radiance'][1,:,:], cmap='plasma', levels=np.arange(0.015,0.120,0.015))
plt.colorbar()
#plt.show()
plt.savefig('/exomars/projects/mc5526/VPCM_deep_atmos_CO/python_mean_co_band_229_v4.png')
# %%
