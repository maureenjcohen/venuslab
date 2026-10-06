# %%
import xarray as xr
from venusdata import *

fname = 'insert file name'

bio = Planet(venusdict, 'vpcm', 'biosphere')
bio.load_file(fname)
bio.setup()

# --- Configuration ---
variables_to_extract = ["vitwz", "Kz", "temp", "pres", "rho", "m0_mode1drop", "m0_mode2drop"]  # adjust to your variable names

time_idx = -1    # integer index along the time dimension
lon_idx  = 48   # integer index along the longitude dimension
lat_idx  = 48   # integer index along the latitude dimension

output_dir = "/exomars/projects/mc5526/biosphere/profiles/"  # must exist, or use pathlib to create it

# --- Load dataset ---
ds = bio.data

# --- Select the profile by index ---
profile_ds = ds[variables_to_extract].isel(
    time_counter=time_idx,
    lon=lon_idx,
    lat=lat_idx,
)

# --- Write each variable to its own .txt file ---
vertical_dim = "altitude"  # <-- change to match your dataset (e.g. "presnivs", "lev")

for var_name in variables_to_extract:
    da = profile_ds[var_name]

    z    = bio.heights
    vals = da.values

    # Retrieve the actual coordinate values for the header
    time_val = ds.time_counter.values[time_idx]
    lon_val  = ds.lon.values[lon_idx]
    lat_val  = ds.lat.values[lat_idx]

    units    = da.attrs.get("units", "unknown")
    longname = da.attrs.get("long_name", var_name)
    z_units  = "km"

    outfile = f"{output_dir}{var_name}_profile.txt"
    with open(outfile, "w") as f:
        f.write(f"# Variable   : {longname} ({var_name})\n")
        f.write(f"# Units      : {units}\n")
        f.write(f"# Time       : {time_val} (index {time_idx})\n")
        f.write(f"# Longitude  : {lon_val}° (index {lon_idx})\n")
        f.write(f"# Latitude   : {lat_val}° (index {lat_idx})\n")
        f.write(f"#\n")
        f.write(f"# {vertical_dim} [{z_units}]    {var_name} [{units}]\n")

        for zi, vi in zip(z, vals):
            f.write(f"{zi:.6e}    {vi:.6e}\n")

    print(f"Written: {outfile}")