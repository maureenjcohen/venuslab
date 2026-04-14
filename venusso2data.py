# %%
from logging import log

from asyncio import log

import pandas as pd
from virtis_validate import orbit
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %%
fpath = '/exomars/projects/mc5526/VPCM_decadal_so2/MC_SO2_SPICAV.txt'
virtis_log = '/exomars/projects/mc5526/VPCM_deep_atmos_CO/VIRTIS_log_v5.0_20130129.csv'

# %%
class SPICAV:
    def __init__(self, filepath, logpath):
        """
        Initializes the SPICAV object by loading data from a text file.
        """
        self.filepath = filepath
        self.logpath = logpath
        self.raw_df = self._load_data()
        self.ds = None

    def _load_data(self):
        """Internal method to parse the specific SPICAV text format."""
        col_names = [
            'so2_ppbv', 'rel_error_pct', 'latitude', 
            'longitude', 'local_solar_time', 'vex_orbit'
        ]

        # Using engine='python' to handle the whitespace-heavy formatting
        df = pd.read_csv(
            self.filepath, 
            sep=r'\s+', 
            names=col_names, 
            comment='#', 
            skipinitialspace=True
        )
        # Ensure orbit is integer for clean indexing
        df['vex_orbit'] = df['vex_orbit'].astype(int)

        log = pd.read_csv(self.logpath, skiprows=1)
        orbit = [x.split('_')[0][2:] for x in log['PRODUCT_ID']]
        log['VEX_ORBIT'] = [pd.to_numeric(x).astype(int) for x in orbit]
        log = log.drop_duplicates(subset=['VEX_ORBIT'], keep='first')
        log['START_DATE'] = pd.to_datetime(log['START_TIME'], utc=True).dt.normalize()
        mapping_dict = dict(zip(log['VEX_ORBIT'], log['START_DATE']))
        df['start_date'] = df['vex_orbit'].map(mapping_dict)
        return df

    def create_dataset(self, grid_type='latlon', lat_res=1.0, lon_res=1.0, lst_res=0.5):
        """
        Populates an xarray.Dataset.
        
        Parameters:
        -----------
        grid_type : str
            'latlon' for Latitude/Longitude grid.
            'lat_lst' for Latitude/Local Solar Time grid.
        lat_res, lon_res, lst_res : float
            The resolution for binning/rounding coordinates.
        """
        df = self.raw_df.copy()

        # 1. Binning the coordinates to create a meaningful grid
        df['latitude'] = (df['latitude'] / lat_res).round() * lat_res
        
        if grid_type == 'latlon':
            df['longitude'] = (df['longitude'] / lon_res).round() * lon_res
            dims = ['vex_orbit', 'latitude', 'longitude']
        elif grid_type == 'latlst':
            df['local_solar_time'] = (df['local_solar_time'] / lst_res).round() * lst_res
            dims = ['vex_orbit', 'latitude', 'local_solar_time']
        else:
            raise ValueError("grid_type must be 'latlon' or 'latlst'")

        # 2. Handling overlapping points within the same bin
        # We take the mean of SO2 and error if multiple shots fall in one grid cell
        df_grouped = df.groupby(dims).mean()

        # 3. Convert to Xarray
        self.ds = df_grouped.to_xarray()

        # 4. Add Metadata
        self._set_metadata()
        
        print(f"Dataset created with dimensions: {list(self.ds.dims)}")
        return self.ds

    def _set_metadata(self):
        """Applies CF-compliant attributes to the variables."""
        if self.ds is None:
            return

        self.ds.so2_ppbv.attrs = {
            'units': 'ppbv',
            'long_name': 'SO2 concentration at 70km',
            'standard_name': 'mole_fraction_of_sulfur_dioxide_in_air'
        }
        self.ds.rel_error_pct.attrs = {
            'units': '%',
            'long_name': 'Relative Error of SO2'
        }
        
        if 'latitude' in self.ds.coords:
            self.ds.latitude.attrs = {'units': 'degrees_north', 'standard_name': 'latitude'}
        if 'longitude' in self.ds.coords:
            self.ds.longitude.attrs = {'units': 'degrees_east', 'standard_name': 'longitude'}
        if 'local_solar_time' in self.ds.coords:
            self.ds.local_solar_time.attrs = {'units': 'hours', 'long_name': 'Local Solar Time'}


## Functions
# %%
def plot_orbit_summary(ds, orbit_idx):
    """
    Plots a 1D profile of SO2 vs Latitude for a specific orbit index.
    Colors the points by the secondary dimension (Longitude or LST).
    """
    # 1. Select the orbit by positional index
    orbit_ds = ds.isel(vex_orbit=orbit_idx)
    
    # 2. Extract the actual Orbit Number for the title
    orb_num = int(orbit_ds.vex_orbit.values)

    # 4. Clean the data (Squeeze out NaNs to get the track points)
    # This converts the sparse 2D slice into a dense 1D set of points
    df_plot = orbit_ds.to_dataframe().dropna(subset=['so2_ppbv']).reset_index()

    if 'longitude' in ds.dims:
        x_coord = 'longitude'
        x_label = 'Lon'
        x_num = df_plot['longitude'].values
    elif 'local_solar_time' in ds.dims:
        x_coord = 'local_solar_time'
        x_label = 'LST'
        x_num = df_plot['local_solar_time'].values

    else:
        raise KeyError("Dataset must contain 'longitude' or 'local_solar_time' as a dimension.")

    if df_plot.empty:
        print(f"Orbit {orb_num} (index {orbit_idx}) contains no valid data.")
        return

    # 5. Create Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Scatter allows us to see the SO2 profile while using color 
    # to show the slight drift in Lon/LST during the orbit
    sc = ax.scatter(df_plot['so2_ppbv'], df_plot['latitude'],
    #                c=df_plot[x_coord], cmap='plasma', 
                    c='r', edgecolor='k', alpha=0.7)
    #plt.colorbar(sc, label=f'{x_label} / {"hr" if x_coord == "local_solar_time" else "deg"}')
    
    ax.set_title(f"SPICAV SO2 Profile | Orbit: {orb_num} | {x_label}: {x_num[0]:.2f}")
    ax.set_xlabel("SO2 / ppbv")
    ax.set_ylabel("Latitude / deg")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.show()

# %%
def plot_orbit_path(ds, orbit_idx):
    """
    Plots the spatial path of the orbit (Latitude vs Lon/LST).
    """
    orbit_ds = ds.isel(vex_orbit=orbit_idx)
    orb_num = int(orbit_ds.vex_orbit.values)
    
    # Determine coordinate type
    is_lst = 'local_solar_time' in ds.dims
    x_dim = 'local_solar_time' if is_lst else 'longitude'
    x_label = 'Local Solar Time [hr]' if is_lst else 'Longitude [deg]'

    # Convert to dataframe for easy scatter plotting
    df_orbit = orbit_ds.to_dataframe().dropna(subset=['so2_ppbv']).reset_index()

    plt.figure(figsize=(8, 5))
    sc = plt.scatter(df_orbit[x_dim], df_orbit['latitude'], 
                     c=df_orbit['so2_ppbv'], cmap='magma', s=40)
    
    plt.colorbar(sc, label='SO2 / ppbv')
    plt.xlabel(x_label)
    plt.ylabel('Latitude / deg')
    plt.title(f'SPICAV Orbit Track | Orbit: {orb_num}')
    plt.grid(True, alpha=0.3)
    plt.show()

# %%
data_processor = SPICAV(fpath)

# %%
dsgeo = data_processor.create_dataset(grid_type='latlon', lat_res=1.0, lon_res=1.0)

# %%
dslst = data_processor.create_dataset(grid_type='latlst', lat_res=1.0, lst_res=0.5)

# %%
