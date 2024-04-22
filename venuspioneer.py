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

    probe_data = pd.read_csv(datapath, usecols=['TIME','DOWN','WEST','NORTH'], sep='\s+')
    altitude_data = pd.read_csv(altspath, skiprows=startrow, usecols=['GRT(SEC)','ALT(KM)'])
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
    interp = merged_df.set_index('TIME')[['ALT(KM)']].interpolate(method='time').reset_index()
    # New DF with just TIME column and altitudes interpolated for all timestamps
    merged_df['ALT(KM)'] = interp['ALT(KM)']
    # Put the interpolated/completed altitudes column back in original DF
    cleaned = merged_df.dropna(axis=0, how='any')
    # Now drop rows containing NaNs, aka rows without wind data
    cleaned.to_csv(savename)
