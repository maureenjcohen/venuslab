""" Uses functions from venuslab to auto-generate plots for
    'An altitude-dependent circulation regime change in the Venus atmosphere',
     Cohen et al. 2024                                       """

""" Usage from command line: python make_regimes.py
    No inputs required as filepaths are specified in script """

# Input file paths and outputs - only part of script that should be edited
# %%
vpcm_path = '/home/maureenjcohen/.nc'
# Simulation with surface age of air tracer - baseline model state

# Import packages
# %%
from venusdata import *

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import windspharm

# %%
def init_model_data(inpath):
    """ Instantiate Planet object from Venus PCM output data"""

    plobject = Planet(venusdict)
    plobject.load_file(inpath)
    plobject.setup()

    return plobject
