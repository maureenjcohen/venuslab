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

def initial_wpharm(ncout, etadata, zetadata, divdata, uchidata, vchidata, upsidata, vpsidata):

    eta_out = ncout.createVariable('eta', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    eta_out.units = 's-1'
    eta_out.long_name = 'Absolute vorticity'
    eta_out[:tstep,:,:,:] = etadata
    print('Initial eta written')

    zeta_out = ncout.createVariable('zeta', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    zeta_out.units = 's-1'
    zeta_out.long_name = 'Relative vorticity'
    zeta_out[:tstep,:,:,:] = zetadata
    print('Initial zeta written')

    div_out = ncout.createVariable('div', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    div_out.units = 's-1'
    div_out.long_name = 'Divergence'
    div_out[:tstep,:,:,:] = divdata
    print('Initial div written')

    uchi_out = ncout.createVariable('uchi', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    uchi_out.units = 's-1'
    uchi_out.long_name = 'Divergent component of wind (u)'
    uchi_out[:tstep,:,:,:] = uchidata
    print('Initial uchi written')

    vchi_out = ncout.createVariable('vchi', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    vchi_out.units = 's-1'
    vchi_out.long_name = 'Divergent component of wind (v)'
    vchi_out[:tstep,:,:,:] = vchidata
    print('Initial vchi written')

    upsi_out = ncout.createVariable('upsi', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    upsi_out.units = 's-1'
    upsi_out.long_name = 'Rotational component of wind (u)'
    upsi_out[:tstep,:,:,:] = upsidata
    print('Initial upsi written')

    vpsi_out = ncout.createVariable('vpsi', 'float32', ('time_counter', 'presnivs', 'lat', 'lon'))
    vpsi_out.units = 's-1'
    vpsi_out.long_name = 'Rotational component of wind (v)'
    vpsi_out[:tstep,:,:,:] = vpsidata
    print('Initial vpsi written')
    print('Initial data block written')


# %%
def add_wpharm(ncout, tstart):

    """ Add windspharm calculated components to LMD-formatted dataset

        Input: a netcdf4 dataset which has been opened in r+ mode

        Requires: import windspharm
                  from windspharm.standard import VectorWind

        This function should be run in a loop with the structure:
            for t in range(0, len(nc['time_counter][:])+1, tstep):
                add_wpharm(ncout, t) """

    print(ncout.variables.keys())
    tend = tstart+tstep
    print(f'Start time: {tstart}, end time: {tend}')

    etalist = []
    zetalist = []
    divlist = []
    uchilist = []
    upsilist = []
    vchilist = []
    vpsilist = []
    for t in range(tstart,tend):
        eta_h = []
        zeta_h = []
        div_h = []
        uchi_h = []
        upsi_h = []
        vchi_h = []
        vpsi_h = []
        for h in range(0,len(ncout['presnivs'][:])):
            winds = VectorWind(ncout['vitu'][t,h,:,:], ncout['vitv'][t,h,:,:])

            eta = winds.absolutevorticity()
            eta_h.append(eta)

            zeta, div = winds.vrtdiv()
            zeta_h.append(zeta)
            div_h.append(div)

            uchi,vchi,upsi,vpsi = winds.helmholtz()
            uchi_h.append(uchi)
            vchi_h.append(vchi)
            upsi_h.append(upsi)
            vpsi_h.append(vpsi)
        
        etalist.append(eta_h)
        zetalist.append(zeta_h)
        divlist.append(div_h)
        uchilist.append(uchi_h)
        upsilist.append(upsi_h)
        vchilist.append(vchi_h)
        vpsilist.append(vpsi_h)
        print(f'Completed time {t}')

    etadata = np.array(etalist)
    zetadata = np.array(zetalist)
    divdata = np.array(divlist)
    uchidata = np.array(uchilist)
    upsidata = np.array(upsilist)
    vchidata = np.array(vchilist)
    vpsidata = np.array(vpsilist)

    if tstart==0:
        print('Initial write of new variables')      
        initial_wpharm(ncout, etadata, zetadata, divdata, uchidata, vchidata, upsidata, vpsidata)
        print(ncout.variables.keys())
        print('Flushing data in memory to disk')
        ncout.sync()

    else:
        ncout['eta'][tstart:tend,:,:,:] = etadata
        ncout['zeta'][tstart:tend,:,:,:] = zetadata
        ncout['div'][tstart:tend,:,:,:] = divdata
        ncout['uchi'][tstart:tend,:,:,:] = uchidata
        ncout['vchi'][tstart:tend,:,:,:] = vchidata
        ncout['upsi'][tstart:tend,:,:,:] = upsidata
        ncout['vpsi'][tstart:tend,:,:,:] = vpsidata
        print(f'Variables written for {tstart} to {tend}')
        print('Flushing data in memory to disk')
        ncout.sync()


     

# %%
