""" Script for pre-processing 0.5 deg resolution OASIS output """
""" Takes 6 separate OASIS nc cubes
    Creates 1 consolidated cube on altitude levels
    Then creates two interpolated cubes on isobars and isentropes   """
# %%
## Import packages
import numpy as np
import xarray as xr
import iris
from iris.experimental import stratify

from venusdata import *

# Part I:
# %%
cubes = ['/exomars/data/analysis/volume_6/mc5526/OASIS_high_res/pre.nc',
        '/exomars/data/analysis/volume_6/mc5526/OASIS_high_res/rho.nc',
        '/exomars/data/analysis/volume_6/mc5526/OASIS_high_res/temp.nc',
        '/exomars/data/analysis/volume_6/mc5526/OASIS_high_res/u.nc',
        '/exomars/data/analysis/volume_6/mc5526/OASIS_high_res/v.nc',
        '/exomars/data/analysis/volume_6/mc5526/OASIS_high_res/w.nc']

alts_cubes = '/exomars/data/analysis/volume_10/mc5526/OASIS_0.5onalts.nc'
pres_cubes = '/exomars/data/analysis/volume_10/mc5526/OASIS_0.5onisobars.nc'
theta_cubes = '/exomars/data/analysis/volume_10/mc5526/OASIS_0.5onisentropes.nc'

# %%
def merge_cubes(cubelist, outpath):
    """ Merge list of OASIS cubes as XR DataArrays into a single DataSet """

    data_arrays = []
    for cube in cubelist:
        data_array = xr.open_dataarray(cube)
        data_arrays.append(data_array)

    dataset = xr.merge(data_arrays)
    dataset = dataset.rename_vars(name_dict={'pre':'pres','u':'vitu','v':'vitv','w':'vitw'})
    dataset = dataset.rename_dims(dims_dict={'time':'time_counter','vert':'presnivs'})
    dataset = dataset.rename({'vert':'presnivs','time':'time_counter'})
    dataset = dataset.transpose('time_counter','presnivs','lat','lon')

    dataset.to_netcdf(outpath)
    return dataset

# %%
def interp_plevs(iris_cubes, plevs, outpath):
    """ Open an Iris CubeList containing pressure along with other variables
        Interpolate the other variables onto the pressure levels given in input list plevs """

    try:
        for cube in iris_cubes:
            if cube.long_name == 'pressure':
                pcube = cube.copy()
    except:
        print('No pressure cube provided')

    new_cubes = []
    for cube in iris_cubes:
        print(f'Interpolating {cube.long_name} cube')
        new_cube = stratify.relevel(cube, pcube, plevs, axis=1)
        new_cubes.append(new_cube)
        print(f'{cube.long_name} added to new cube list')

    iris.save(new_cubes, outpath)

    return new_cubes

# %%
def interp_isentropes(iris_cubes, thetalevs, outpath):
    """ Open Iris CubeList which is on pressure levels
        Interpolate all variables onto isentropes       """
    
    if thetalevs is None:
        thetalevs = [283] + list(np.arange(290,980,10))
    
    for cube in iris_cubes:
        if cube.long_name == 'temperature':
            temp = cube.copy()
        if cube.long_name == 'pressure':
            pcube = cube.copy()    
    
    p0 = iris.coords.AuxCoord(100000, long_name='reference_pressure', units='Pa')
    theta = temp*((p0/pcube)**(188/900))
    theta.long_name = 'potential temperature'
    theta.units = 'K'

    new_cubes = []
    for cube in iris_cubes:
        print(f'Interpolating {cube.long_name} cube')
        new_cube = stratify.relevel(cube, theta, thetalevs, axis=1)
        new_cubes.append(new_cube)
        print(f'{cube.long_name} added to new cube list')

    iris.save(new_cubes, outpath)

    return new_cubes


# %%
