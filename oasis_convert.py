""" Convert OASIS data into LMD-style file readable by venuslab"""

# %%
## Import packages
import netCDF4 as nc
import numpy as np

# %%
## Variable keys of data cubes
ukey = 'vitu' # U wind
vkey = 'vitv' # V wind
wkey = 'vitw' # W wind
tkey = 'temp' # Air temp
gkey = 'geop' # Geopotential height
tckey = 'time_counter' # Time counter

# %%
# Processing info
outdir = '/home/maureenjcohen/misc_data/'
infile = 'OASIS_Venus/OASIS2Deg.nc'
outfile = 'OASIS_conv.nc'
tstep = 9 # Number of timesteps to write to output file at a time

# %%
def make_nc(ncout, udata, vdata, wdata, tdata, gdata, plevs, lats, lons, time):
    """ Makes netCDF file to LMD metadata standard"""
    # Create dimensions of new file with LMD dimension names
    ncout.createDimension('time_counter', None)
    ncout.createDimension('presnivs', len(plevs))
    ncout.createDimension('lat', len(lats))
    ncout.createDimension('lon', len(lons))

    # Create longitude variable
    longitude = ncout.createVariable('lon', 'float32', ('lon',))
    longitude.units = 'degrees_east'
    longitude.axis = 'X'
    longitude.long_name = 'Longitude'
    longitude[:] = lons

    # Create latitude variable
    latitude = ncout.createVariable('lat', 'float32', ('lat',))
    latitude.units = 'degrees_north'
    latitude.axis = 'Y'
    latitude.long_name = 'Latitude'
    latitude[:] = lats

    # Create pressure levels
    pressures = ncout.createVariable('presnivs', 'float32', ('presnivs',))
    pressures.units = 'Pa'
    pressures.axis = 'Z'
    pressures.long_name = 'Vertical levels'
    pressures[:] = plevs

    # Create times
    times = ncout.createVariable('time_counter','float32', ('time_counter',))
    times.units = 'seconds since 1111-01-01 00:00:00'
    times.axis = 'T'
    times.long_name = 'Time axis'
    times.time_origin = '1111-JAN-01 00:00:00'
    times.calendar = '360_day'
    times[:] = time

    # Create u-wind
    uout = ncout.createVariable('vitu', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    uout.units = 'm/s'
    uout.long_name = 'Zonal wind'
    uout[:,:,:,:] = udata

    # Create v-wind
    vout = ncout.createVariable('vitv', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    vout.units = 'm/s'
    vout.long_name = 'Meridional wind'
    vout[:,:,:,:] = vdata

    # Create w-wind
    wout = ncout.createVariable('vitw', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    wout.units = 'Pa/s'
    wout.long_name = 'Vertical wind'
    wout[:,:,:,:] = wdata

    # Create air temperature
    tout = ncout.createVariable('temp', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    tout.units = 'K'
    tout.long_name = 'Air temperature'
    tout[:,:,:,:] = tdata

    # Create geopotential height
    gout = ncout.createVariable('geop', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    gout.units = 'm'
    gout.long_name = 'Geopotential height'
    gout[:,:,:,:] = gdata
    print('File written')


# %%
def extract_mdata(ncfile):
    """ Input a netcdf4 file and extract the metadata that will be used
    to create a new, reformatted netcdf4 file 
    
    Outputs: arrays of longitudes, latitudes, timestamps, and scalar value
             of the time interval between each output cube (in seconds)  """
    lons = ncfile['lon'][:]
    lats = ncfile['lat'][:]
    plevs = ncfile['presnivs'][:]
    time = ncfile['time_counter'][:tstep]

    return lons, lats, plevs, time

# %%
def transpose_data(incube):

    """ OASIS data has shape time, height, lon, lat
        LMD data has shape time, height, lat, lon 
        
        This function swaps the last two axes of OASIS 
        to match LMD standard"""

    outcube = np.swapaxes(incube, 2, 3)

    return outcube


# %%
def initial_convert(inputfile, outputfile):

    # Get sizes of dimensions from input file
    inputds = nc.Dataset(inputfile, 'r')
    lons, lats, plevs, time = extract_mdata(inputds)

    # Swap axes of first batch of data
    u = transpose_data(inputds[ukey][0:tstep,:,:,:])
    v = transpose_data(inputds[vkey][0:tstep,:,:,:])
    w = transpose_data(inputds[wkey][0:tstep,:,:,:])
    t = transpose_data(inputds[tkey][0:tstep,:,:,:])
    g = transpose_data(inputds[gkey][0:tstep,:,:,:])

    # Create output file
    ncout = nc.Dataset(outputfile,'w',
                       format='NETCDF4')
    
    # Read in dimensions, variables, and first batch of data
    make_nc(ncout, u, v, w, t, g, plevs, lats, lons, time)

    inputds.close()
    ncout.close(); del ncout

# %%
def append_data(inpath, appendpath):

    inputfile = nc.Dataset(inpath, 'r')
    print(inputfile.variables.keys())
    appendfile = nc.Dataset(appendpath, 'a')
    print(appendfile.variables.keys())

    current_time = len(appendfile['time_counter'])
    max_time = len(inputfile['time_counter'])
    print(f'Existing file up to time: {current_time}')
    batch_end = current_time + tstep

    if batch_end <= max_time:
        print(current_time, batch_end)

        u = transpose_data(inputfile[ukey][current_time:batch_end,:,:,:])
        v = transpose_data(inputfile[vkey][current_time:batch_end,:,:,:])
        w = transpose_data(inputfile[wkey][current_time:batch_end,:,:,:])
        t = transpose_data(inputfile[tkey][current_time:batch_end,:,:,:])
        g = transpose_data(inputfile[gkey][current_time:batch_end,:,:,:])
        add_time = inputfile[tckey][current_time:batch_end]

        appendfile[ukey][current_time:batch_end,:,:,:] = u
        appendfile[vkey][current_time:batch_end,:,:,:] = v
        appendfile[wkey][current_time:batch_end,:,:,:] = w
        appendfile[tkey][current_time:batch_end,:,:,:] = t
        appendfile[gkey][current_time:batch_end,:,:,:] = g
        appendfile[tckey][current_time:batch_end] = add_time

        print(appendfile[tckey].shape)

        inputfile.close()
        appendfile.close()

        print(f'Data appended up to time: {batch_end}')

    else:
        print(f'Batch end time: {batch_end}')
        print(f'Maximum time is: {max_time}')

        inputfile.close()
        appendfile.close()

    return batch_end



# %%
