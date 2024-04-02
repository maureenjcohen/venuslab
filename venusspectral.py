""" Spectral analysis"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from venusdyn import time_series

# %%
def timeseries_transform(plobject, key='vitw', 
                         coords=[(16,86,48),(22,86,48),(30,86,48)],
                         trange=[1777,1877]):

    series_list, coords_list = time_series(plobject, key=key, 
                coords=coords, 
                ptitle='filler', ylab='filler', 
                unit='filler', plot=False, trange=trange, 
                tunit='filler',
                fsize=14, savearg=False, sformat='png')
    
    return series_list