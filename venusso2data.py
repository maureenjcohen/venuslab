# %%
import pandas as pd
from virtis_validate import orbit
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# %%
fpath = '/exomars/projects/mc5526/VPCM_decadal_so2/MC_SO2_SPICAV.txt'
lpath = '/exomars/projects/mc5526/VPCM_decadal_so2/SPICAV_dates.txt'

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

        log_cols = ['filename','product_id','product_creation_time',
                    'data_set_id','release_id','revision_id',
                    'start_time','stop_time','nb_records']

        log = pd.read_csv(self.logpath, header=None, names=log_cols, skiprows=22)
        orbit = [x.split('/')[3][5:] for x in log['filename']]
        log['vex_orbit'] = [pd.to_numeric(x).astype(int) for x in orbit]
        log = log.drop_duplicates(subset=['vex_orbit'], keep='first')
        log['start_date'] = pd.to_datetime(log['start_time'], utc=True).dt.normalize()

        last_orbit = log['vex_orbit'].max()
        last_date = log['start_date'].max()
        target_orbit = 3146
        new_orbits = range(last_orbit + 1, target_orbit + 1)
        new_dates = pd.date_range(start = last_date + pd.Timedelta(days=1), periods=len(new_orbits), freq='D')
        df_extrapolated = pd.DataFrame({'vex_orbit': new_orbits, 'start_date': new_dates})
        log = pd.concat([log, df_extrapolated], ignore_index=True)

        mapping_dict = dict(zip(log['vex_orbit'], log['start_date']))
        df['start_date'] = df['vex_orbit'].map(mapping_dict)
        df['start_date'] = df['start_date'].dt.tz_localize(None)
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
            dims = ['start_date', 'latitude', 'longitude']
        elif grid_type == 'latlst':
            df['local_solar_time'] = (df['local_solar_time'] / lst_res).round() * lst_res
            dims = ['start_date', 'latitude', 'local_solar_time']
        else:
            raise ValueError("grid_type must be 'latlon' or 'latlst'")

        # 2. Handling overlapping points within the same bin
        # We take the mean of SO2 and error if multiple shots fall in one grid cell
        df_for_grid = df.drop(columns=['vex_orbit'])
        df_grouped = df_for_grid.groupby(dims).mean()

        # 3. Convert to Xarray
        self.ds = df_grouped.to_xarray()
        orbit_mapping = self.raw_df.drop_duplicates('start_date').set_index('start_date')['vex_orbit']
        orbits = orbit_mapping.reindex(self.ds.start_date.values).values
        self.ds.coords['vex_orbit'] = ('start_date', orbits)

        # Promote the non-dimension variable to a multidimensional coordinate
        if grid_type == 'latlon':
            self.ds = self.ds.set_coords('local_solar_time')
        elif grid_type == 'latlst':
            self.ds = self.ds.set_coords('longitude')

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
        if 'start_date' in self.ds.coords:
            self.ds.start_date.attrs = {'long_name': 'Start Date of Orbit'} 
        if 'vex_orbit' in self.ds.coords:
            self.ds.vex_orbit.attrs = {'units': '1', 'long_name': 'VEX Orbit Number'}  


## Functions
# %%
def plot_orbit_summary(ds, orbit_idx):
    """
    Plots a 1D profile of SO2 vs Latitude for a specific orbit index.
    Colors the points by the secondary dimension (Longitude or LST).
    """
    # 1. Select the orbit by positional index
    orbit_ds = ds.isel(start_date=orbit_idx)
    
    # 2. Extract the actual Orbit Number for the title
    orb_num = int(orbit_ds.vex_orbit.values)
    day = pd.to_datetime(orbit_ds.start_date.values).strftime('%Y-%m-%d')

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
    
    ax.set_title(f"SPICAV SO2 Profile | Orbit: {orb_num} | {x_label}: {x_num[0]:.2f} | Date: {day}")
    ax.set_xlabel("SO2 / ppbv")
    ax.set_ylabel("Latitude / deg")
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.show()

# %%
def plot_orbit_path(ds, orbit_idx):
    """
    Plots the spatial path of the orbit (Latitude vs Lon/LST).
    """
    orbit_ds = ds.isel(start_date=orbit_idx)
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
def marcq_fig1(ds, weighted=False, save=False,
               savepath='/exomars/projects/mc5526/VPCM_decadal_so2/scratch_plots/'):
    """
    Reproduces Fig. 1 from Marcq. 2013 
    """
    if weighted==True:
        # Get latitude-weighted so2 values
        weights = np.cos(np.deg2rad(ds.latitude))
        so2_weighted = ds.so2_ppbv.weighted(weights).mean(dim=['latitude', 'longitude'])
        error_mean = ds.rel_error_pct.mean(dim=['latitude', 'longitude'])
        error_absolute = error_mean * so2_weighted / 100.0

    else:
        df_valid = ds['so2_ppbv'].to_dataframe().dropna().reset_index()
        df_error = ds['rel_error_pct'].to_dataframe().dropna().reset_index()
        error_absolute = df_error['rel_error_pct'] * df_valid['so2_ppbv'] / 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    if weighted==True:
        ax.errorbar(ds.start_date.values, so2_weighted, yerr=error_absolute, 
                     fmt='o', markerfacecolor='r', markeredgecolor='k', markersize=3, markeredgewidth=0.5, 
                     ecolor='k',linewidth=1, capsize=5, capthick=2)
        ax.set_title('Latitude-weighted SO2 at 70km')
    else:
        ax.errorbar(df_valid['start_date'], df_valid['so2_ppbv'], yerr=error_absolute, 
                    fmt='o', markerfacecolor='r', markeredgecolor='k', markersize=3, markeredgewidth=0.5, 
                     ecolor='k',linewidth=1, capsize=5, capthick=2)
        ax.set_title('All SO2 observations at 70km')
    ax.set_xlabel('Date')
    ax.set_ylabel('SO2 / ppbv')
    ax.set_yscale('log')
    ax.set_xlim(pd.to_datetime('2006-01-01'), pd.to_datetime('2015-01-01'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    if save:
        fig.savefig(savepath+'marcq_fig1_weighted.png', bbox_inches='tight') if weighted else fig.savefig(savepath+'marcq_fig1_all.png', bbox_inches='tight')

    plt.show()

# %%
data_processor = SPICAV(fpath, lpath)

# %%
dsgeo = data_processor.create_dataset(grid_type='latlon', lat_res=1.0, lon_res=1.0)

# %%
dslst = data_processor.create_dataset(grid_type='latlst', lat_res=1.0, lst_res=0.5)

# %%
