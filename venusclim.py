# %%
import iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# %%
def sw_flux(cube, time_slice=-1, lev=-1):

    sw = cube.copy()
    sw_data = sw[time_slice,lev,:,:].data

    plt.contourf(sw_data, cmap='hot')
    plt.title(f'Net SW flux, level {lev} [W/m$^2$]')
    plt.ylabel('Latitude [deg]')
    plt.xlabel('Longitude [deg]')
    plt.colorbar()
    plt.show()

# %%
def surface_temp(cube, meaning=True, time_slice=-1):

    st = cube.copy()
    if meaning==True:
        st = st.collapsed('time',iris.analysis.MEAN)
        titleterm = 'long-term mean'
    else:
        st = st[time_slice,:,:]
        titleterm = f'at {time_slice}'

    st_data = st.data

    plt.contourf(st_data, cmap='Reds')
    plt.title(f'Surface temperature, {titleterm}')
    plt.ylabel('Latitude [deg]')
    plt.xlabel('Longitude [deg]')
    plt.colorbar()
    plt.show()
# %%
