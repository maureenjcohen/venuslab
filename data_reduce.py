""" Reduce size of Xins.nc files"""

import sys
from pathlib import Path
import xarray as xr

# Hard-coded base directory
BASE_DIR = Path("/exomars/projects/mc5526/ML_so2/")


def main():
    # Find all target netCDF files matching the directory and filename pattern
    files = list(BASE_DIR.glob("**/chem*96x96x78/Xins_*.nc"))

    if not files:
        print(f"No files found matching the pattern under {BASE_DIR}")
        sys.exit(0)

    vars_to_keep = [
        "time_instant_bounds",
        "time_counter_bounds",
        "phis",
        "aire",
        "tops",
        "tsol",
        "psol",
        "temp",
        "pres",
        "vitu",
        "vitv",
        "vitwz",
        "so2",
    ]

    print(f"Found {len(files)} files to process.")

    for file_path in files:
        # Extracts 'chem0.7287_96x96x78' from path
        dir_name = file_path.parent.name

        clean_dir_name = dir_name.replace("_96x96x78", "")

        # Construct output: e.g., chem0.7287_Xins_3_reduced.nc
        output_name = f"{clean_dir_name}_{file_path.stem}_reduced.nc"
        output_path = BASE_DIR / output_name
        try:
            # decode_times=False handles non-standard planetary/model time calendars
            with xr.open_dataset(file_path, decode_times=False) as ds:
                # Isolate variables and take the vertical index slice
                ds_subset = ds[vars_to_keep].isel(presnivs=slice(30, 40))

                # Write directly back to base directory
                ds_subset.to_netcdf(output_path)
                print(f"Successfully created: {output_name}")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()