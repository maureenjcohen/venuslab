## For Arnaud's paper
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from venuschem import chem_local_mean
from venusdata import *

# %%
fns = ['/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/data/Xins4.nc', 
       '/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/data/Xins5.nc', 
       '/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/data/Xins6.nc', 
       '/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/data/Xins7.nc']

# %%
if __name__ == 'main':

    # Load data
    # %%
    chem = Planet(venusdict, 'vpcm','4days_chem')
    chem.load_file(fns)
    chem.setup()

    # Visual check that it makes sense
    # Remember ticks are relabelled to go backwards (24 to 0)
    # But the tick locations are still 0 to 24
    # %%
    chem_local_mean(chem, 'o', 72, (0,None))

    # 18h/western profile at 60N
    # %%
    west_mean = local_mean_profile(chem, 'o', 6, 80, (0,None))
    east_mean = local_mean_profile(chem, 'o', 18, 80, (0,None))

    west_sample1 = local_mean_profile(chem, 'o', 6, 80, (0,1))
    west_sample2 = local_mean_profile(chem, 'o', 6, 80, (11,12))
    west_sample3 = local_mean_profile(chem, 'o', 6, 80, (23,24))
    west_sample4 = local_mean_profile(chem, 'o', 6, 80, (35,36))

    east_sample1 = local_mean_profile(chem, 'o', 18, 80, (0,1))
    east_sample2 = local_mean_profile(chem, 'o', 18, 80, (11,12))
    east_sample3 = local_mean_profile(chem, 'o', 18, 80, (23,24))
    east_sample4 = local_mean_profile(chem, 'o', 18, 80, (35,36))

    df_data = {'altitude [km]' : chem.heights,
               'pressure [Pa]': chem.data['presnivs'],
               'west_mean' : west_mean,
               'east_mean': east_mean,
               'west_sample1': west_sample1,
               'west_sample2': west_sample2,
               'west_sample3': west_sample3,
               'west_sample4': west_sample4,
               'east_sample1': east_sample1,
               'east_sample2': east_sample2,
               'east_sample3': east_sample3,
               'east_sample4': east_sample4
    }
    df = pd.DataFrame(df_data)

    # %%
    df.to_csv('/exomars/projects/mc5526/VPCM_full_chemistry_runs/vpcm_oxygen_profiles_vmr.csv')
# %%
