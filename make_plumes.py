"""
Auto-generate plots for 'How long could volcanic plumes persist in the Venus atmosphere?'
(Cohen et al. 2026).

Usage:
    python make_plumes.py

No inputs required as filepaths are specified in the script.
"""

# %% Filepaths
plumes = '/exomars/data/internal/working/mc5526/VPCM_plumes/Xins_1.5scale.nc'
surf = '/exomars/data/internal/working/mc5526/VPCM_age_of_air/surf_96x96x50/Xins_211to220.nc'
chem_dir = '/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/data/'
chem = [chem_dir + 'Xins4.nc',  chem_dir + 'Xins5.nc', chem_dir + 'Xins6.nc', chem_dir + 'Xins7.nc']
savedir = '/exomars/projects/mc5526/VPCM_volcanic_plumes/figures/'
filetype = 'png'

# %% Imports
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %% Definition of basic Venus model constants
# radius: km, g: m/s^2, periods: days, psurf: bar
# molmass: kg/mol, R: J/(mol K), scaleh: km
venusdict = {'radius': 6051.3, 'g': 8.87, 'rotperiod' : 243.0,
             'revperiod': 224.7, 'rotrate': 2.99e-07, 'psurf': 92.,
             'molmass': 0.04401, 'R' : 8.3143, 'RCO2' : 188.92, 'rhoconst': 65.,
             'scaleh': 16.,
             'name': 'Venus'}
# Altitude levels in km for VPCM 50 level outputs
heights50 = [0.,  0.05,  0.2,  0.4,  0.8,  1.3,  2.2,  3.3,  4.7,  6.5,  8.6,
       11.1, 14., 17.3, 20.9, 24.7, 28.5, 32.1, 35.4, 38.6, 41.6, 44.4,
       47.1, 49.7, 52.1, 54.3, 56.4, 58.4, 60.3, 62.1, 63.9, 65.6, 67.4,
       69., 70.7, 72.3, 73.9, 75.4, 76.9, 78.4, 79.8, 81.2, 82.6, 84.,
       85.3, 86.8, 88.7, 91.2, 94.1, 97.] 
# Altitude levels in km for VPCM 78 level outputs
heights78 = [9.45306290e-03, 4.83510271e-02, 1.53046221e-01, 3.73352647e-01,
       7.54337728e-01, 1.33960199e+00, 2.17016840e+00, 3.28359985e+00,
       4.71279430e+00, 6.48478937e+00, 8.61961460e+00, 1.11300869e+01,
       1.40210838e+01, 1.72868519e+01, 2.09060841e+01, 2.47483273e+01,
       2.85367489e+01, 3.21012650e+01, 3.54501762e+01, 3.85974350e+01,
       4.15684090e+01, 4.43950195e+01, 4.71000137e+01, 4.96712570e+01,
       5.20832405e+01, 5.43336220e+01, 5.64306679e+01, 5.83991661e+01,
       6.02850800e+01, 6.21196480e+01, 6.39083824e+01, 6.56521988e+01,
       6.73622131e+01, 6.90550003e+01, 7.07164383e+01, 7.23295288e+01,
       7.38933105e+01, 7.54046097e+01, 7.68734818e+01, 7.83174973e+01,
       7.97435608e+01, 8.11459122e+01, 8.25166473e+01, 8.38473129e+01,
       8.51307755e+01, 8.65320282e+01, 8.83726730e+01, 9.07071533e+01,
       9.34497375e+01, 9.65266037e+01, 9.90068893e+01, 1.00908829e+02,
       1.02824753e+02, 1.04765404e+02, 1.06736778e+02, 1.08729904e+02,
       1.10737305e+02, 1.12745575e+02, 1.14689415e+02, 1.16525398e+02,
       1.18274094e+02, 1.19964310e+02, 1.21610733e+02, 1.23218185e+02,
       1.24797585e+02, 1.26358299e+02, 1.27900208e+02, 1.29420120e+02,
       1.30924683e+02, 1.32434402e+02, 1.33977646e+02, 1.35579971e+02,
       1.37256210e+02, 1.39007202e+02, 1.40822159e+02, 1.42683762e+02,
       1.44574203e+02, 1.46479889e+02]
# Dictionary of plume coordinates
plume_dict = {  'plume_1': {'name': 'h2o_lev10_eq', 'lev': 10, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
                'plume_2': {'name': 'h2o_lev10_hl', 'lev': 10, 'lat_idx': 82, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
                'plume_3': {'name': 'h2o_hcl_lev14_eq', 'lev': 14, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
                'plume_4': {'name': 'h2o_hcl_lev14_hl', 'lev': 14, 'lat_idx': 82, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
                'plume_5': {'name': 'four_gases_lev18_eq', 'lev': 18, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
                'plume_6': {'name': 'four_gases_lev18_hl', 'lev': 18, 'lat_idx': 82, 'lon_idx': 47, 'start_time': 4, 'end_time': 127},
                'plume_7': {'name': 'so2_lev35_eq', 'lev': 35, 'lat_idx': 49, 'lon_idx': 92, 'start_time': 4, 'end_time': 127},
                'plume_8': {'name': 'so2_lev35_hl', 'lev': 35, 'lat_idx': 82, 'lon_idx': 47, 'start_time': 4, 'end_time': 127}
            }

# %% Classes
class PlumeSim:
    """
    A PlumeSim object which contains the output data for a simulation.
    """
    
    def __init__(self, planetdict, plume_dict, run):
        """
        Initiate a PlumeSim object.

        Args:
            planetdict (dict): Dictionary of planet constants.
            model (str): Name of the model ('vpcm').
            run (str): Name of the run, should be the scaling factor.
        """
        self.name = planetdict['name']
        self.plumes = plume_dict
        self.run = run
        # Easter egg
        print(f'Welcome to Venus. Your lander will melt in 57 minutes.')
        print(f'This is the {self.run} dataset')
        for key, value in planetdict.items():
            setattr(self, key, value)

    def load_file(self, fn):
        """
        Load a netCDF file using the xarray package and store it in the object.

        Lists dictionary key, name, dimensions, and shape of each data cube
        and stores text in a reference list.

        Args:
            fn (str or list): Filename or list of filenames to load.
        """
        if isinstance(fn, str):
            ds = xr.open_dataset(fn, decode_cf=False)
        elif isinstance(fn, list):
            ds = xr.open_mfdataset(fn, combine='nested', concat_dim='time_counter', decode_cf=False)
        else:
            print('Improper filename input, must be string or list')
        reflist = []
        str1 = 'File contains:'
        print(str1)
        reflist.append(str1)
        for key in ds.data_vars:
            if 'long_name' in ds[key].attrs:
                keystring = f"{key}: {ds[key].long_name}, {ds[key].dims}, {ds[key].shape}"
                print(keystring)
                reflist.append(keystring)
            else:
                keystring = f"{key}: {ds[key].dims}, {ds[key].shape}"
                print(keystring)
                reflist.append(keystring)
        self.data = ds
        self.reflist = reflist

    def close(self):
        """
        Close the netCDF file packaged in the PlumeSim data object.
        """
        self.data.close()
        print('PlumeSim object associated dataset has been closed')

    def set_resolution(self):
        """
        Automatically detect file resolution and assign aesthetically
        pleasing coordinate arrays to the object for use in labelling plots.
        """
        self.lons = np.round(self.data.variables['lon'].values)
        self.lats = np.round(self.data.variables['lat'].values)
        self.tinterval = np.diff(self.data['time_counter'][0:2])[0]
        self.areas = self.data.variables['aire'].values
        if len(self.data.variables['presnivs'][:]) == 50:
            self.heights = np.array(heights50)
        elif len(self.data.variables['presnivs'][:]) == 78:
            self.heights = np.array(heights78)
        else:
            print('Altitude in km not available')       
        self.set_vertical()
        print(f"Resolution is {len(self.lats)} lats, {len(self.lons)} lons, {self.vert} levs")
        print(f'Vertical axis is {self.vert_axis}')

    def set_vertical(self):
        """
        Identify and set vertical axis and units.
        """
        self.levs = self.data['presnivs'].values
        self.vert = len(self.levs)
        self.vert_unit = self.data['presnivs'].units
        try:
            self.vert_axis = self.data['presnivs'].long_name
        except:
            self.vert_axis = self.data['presnivs'].standard_name

# %% Functions
def find_plume(plobject, key, lev, threshold):
    """
    Find array indices of maximum tracer value.

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        key (str): Dictionary key of the data variable.
        lev (int): Vertical level index.
        threshold (float or None): Threshold value. If None, max value is used.

    Returns:
        tuple: (start_time, end_time, lat_idx, lon_idx) indices.
    """
    # Extract age of air tracer cube at desired model level
    cube = plobject.data[key][:,lev,:,:]
    
    # What is the peak mmr of the plume injection?
    if threshold is None:
        threshold = cube.max()
    
    mask = np.where(cube >= threshold)
    # print(mask) # Debug
    
    # Get 0th element of 0th dimension (first time occurrence)
    start_time = mask[0][0]
    
    # Get last element of 0th dimension (last time occurrence)
    end_time = mask[0][-1]
    
    # Get 0th element of 1st dimension (latitude) - all should be identical
    lat_idx = mask[1][0]
    
    # Get 0th element of 2nd dimension (longitude) - all should be identical
    lon_idx = mask[2][0]

    return start_time, end_time, lat_idx, lon_idx

# %%
def zmage(plobject, hmin=0, hmax=None, time_slice=-1, convert2yr=True,
          plume_markers=None, levels=None, savepath=None,
          save=False, sformat='png', savename='zmage.png'):
    """
    Plot the zonal mean age of air.

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        hmin (int): Minimum height index. Defaults to 0.
        hmax (int): Maximum height index. Defaults to None.
        time_slice (int): Time index to select. Defaults to -1.
        convert2yr (bool): Whether to convert units to years. Defaults to True.
        levels (list, optional): Contour levels. Defaults to None.
        savepath (str): Directory path to save the plot. Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        saveformat (str): Format to save the plot. Defaults to 'png'.
        savename (str): Filename for the saved plot. Defaults to 'zmage.png'.
    """
    ageo = plobject.data['age']
    # Select time slice
    ageo = ageo[time_slice,:,:,:]
    
    # Calculate zonal mean
    zmageo = np.mean(ageo, axis=-1)

    if convert2yr:
        zmageo = zmageo/(60*60*24*360)
        cunit = 'Earth years'
    else:
        cunit = 'seconds'
 
    zmslice = zmageo[hmin:hmax,:]
    
    fig = plt.figure(figsize=(6, 6))
    plt.contourf(plobject.lats, plobject.heights[hmin:hmax],
                 zmslice,
                 levels=levels,
                 cmap='jet')
    if plume_markers is not None:
        for plume in plume_markers:
            plt.plot(plobject.lats[plume_markers[plume]['lat_idx']], plobject.heights[plume_markers[plume]['lev']], marker='*', color='black', markersize=10)
    plt.title('Zonal mean age of air', fontsize=14)
    plt.xlabel('Latitude / deg')
    plt.ylabel('Height / km')
    cbar = plt.colorbar()
    cbar.set_label(f'{cunit}')
    
    if save:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# %%
def dispersal_time(plobject, lev, keys, lats, lons,
                   axis_len=400, save=False,
                   savename='plume_dispersal.png',
                   savepath=None,
                   sformat='png'):
    """
    Find the time for the tracer value in the gridbox to return to the background value.

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        key (str): Dictionary key of the data variable.
        lev (int): Vertical level index.
        threshold (float or None): Threshold value for defining the plume.
        save (bool): Whether to save the plot. Defaults to False.
        savename (str): Filename for the saved plot. Defaults to 'plume_dispersal.png'.
        savepath (str): Directory path to save the plot. Defaults to None.
        sformat (str): Format to save the plot. Defaults to 'png'.
    """
    # Extract data at desired model level
    if isinstance(keys, str):
        keys = [keys]
    num_subplots = len(keys)
    if num_subplots == 1:
        num_cols, num_rows = 1, 1
    elif num_subplots == 2:
        num_cols, num_rows = 2, 1
    elif num_subplots == 4:
        num_cols, num_rows = 2, 2

    position = range(1, num_subplots+1)
    interval = np.diff(plobject.data.time_counter.values)[0]
    time_axis = np.arange(plobject.plumes['plume_1']['start_time']-2, axis_len)*interval/(60*60)

    fig = plt.figure(figsize=(num_cols*4, num_rows*4), tight_layout=True)

    for i, key in enumerate(keys):
        series1 = plobject.data[key][:,lev,lats[0],lons[0]]
        series2 = plobject.data[key][:,lev,lats[1],lons[1]]
        # Get background value of tracer before plume starts
        if key=='co':
            background_val = 8.0e-06
        else:
            background_val = series1[plobject.plumes['plume_1']['start_time']-1].values*1.005
        print(background_val)
    
        counter1 = 0
        for t in range(plobject.plumes['plume_1']['end_time'], series1.shape[0]):
            if series1[t].values > background_val:
                # Count how many time steps tracer value remains above background
                counter1 = counter1 + 1
            else:
                # Stop loop when tracer returns to background
                break

        counter2 = 0
        for t in range(plobject.plumes['plume_1']['end_time'], series2.shape[0]):
            if series2[t].values > background_val:
                # Count how many time steps tracer value remains above background
                counter2 = counter2 + 1
            else:
                # Stop loop when tracer returns to background
                break
        
        # Total time in seconds until tracer valuer returns to background
        disp_time1 = counter1 * interval
        disp_time2 = counter2 * interval
        
        # Convert to hours for convenience
        disp_hours1 = disp_time1 / (60*60)
        disp_hours2 = disp_time2 / (60*60)

        # Get data, including 5 time steps before and after plume
        data1 = series1[plobject.plumes['plume_1']['start_time']-2:axis_len]*1e6
        data2 = series2[plobject.plumes['plume_1']['start_time']-2:axis_len]*1e6
    
        ax = fig.add_subplot(num_rows, num_cols, position[i])
        ax.plot(time_axis, data1, color='blue', label=f'Lat {np.round(plobject.lats[lats[0]],2)} deg, {np.round(disp_hours1,2)} hrs')
        ax.plot(time_axis, data2, color='green', label=f'Lat {np.round(plobject.lats[lats[1]],2)} deg, {np.round(disp_hours2,2)} hrs')
        ax.plot(time_axis, np.ones_like(data1)*background_val*1e6, color='red',
                linestyle='dashed', label='Background value')
        ax.set_title(f'{key.upper()}')
        ax.set_ylabel(f'{key.upper()} vmr / ppm')
        ax.set_ylim([background_val*1e6*0.8, data1.max()*1.2])
        ax.set_xlabel('Time / hours')
        plt.legend()
    fig.suptitle(f'Plume dispersal times, h = {np.round(plobject.heights[lev], 2)} km', y=0.97, fontsize=14)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if save:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    plt.show()

# %%
def animate_chem_plume(plobject, lev, t0, tf, n=4, qscale=1,
                  savepath=None):
    """
    Create an animation of a chemical plume.

    Args:
        plobject (PlumeSim): PlumeSim class object containing the data.
        lev (int): Vertical level index to be visualised.
        t0 (int): First frame (time index).
        tf (int): Final frame (time index).
        n (int): Quiver plot sampling interval. Defaults to 4.
        qscale (float): Scale for quiver plot. Defaults to 1.
        savepath (str): Directory path to save the output. Defaults to None.
    """
    # Get height in km rounded to 2 decimal points (for title)
    height = np.round(plobject.heights[lev],2)
    
    # Extract data for desired altitude
    h2o_cube = plobject.data['h2o'][t0:tf,lev,:,:]*1e6
    co_cube = plobject.data['co'][t0:tf,lev,:,:]*1e6
    ocs_cube = plobject.data['ocs'][t0:tf,lev,:,:]*1e6
    hcl_cube = plobject.data['hcl'][t0:tf,lev,:,:]*1e6

    # Extract zonal and meridional wind for desired altitude
    u = plobject.data['vitu'][t0:tf,lev,:,:]
    v = plobject.data['vitv'][t0:tf,lev,:,:]

    # Create figure
    fig, ax = plt.subplots(2,2,figsize=(16, 10), sharex=True, sharey=True)
    X, Y = np.meshgrid(plobject.lons, plobject.lats)

    interval = np.diff(h2o_cube.time_counter.values)[0]/(60*60)
    time_axis = np.round(np.arange(0,tf-t0)*interval,0)
 
    quiv_args = {
    'angles': 'xy',
    'scale_units': 'xy',
    'scale': qscale,
    'color': 'black'
    }
    
    # Define an update function that will be called for each frame
    def animate(frame):
        # H2O plot
        cf_h2o = ax[0,0].contourf(plobject.lons, plobject.lats, h2o_cube[frame,:,:],
                                  cmap='Blues',
                                  vmin=np.min(h2o_cube), vmax=45.0)
        q1 = ax[0,0].quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax[0,0].quiverkey(ax[0,0].quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label=f'{qscale*10} m/s',
                 labelpos='E', coordinates='axes', color='black')
        ax[0,0].set_title('H2O', color='black', y=1.05, fontsize=14)
        ax[0,0].set_ylabel('Latitude / deg')

        # CO plot
        cf_co = ax[0,1].contourf(plobject.lons, plobject.lats, co_cube[frame,:,:],
                                  cmap='Purples', vmin=np.min(co_cube), vmax=37.5)
        q2 = ax[0,1].quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax[0,1].quiverkey(ax[0,1].quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label=f'{qscale*10} m/s',
                 labelpos='E', coordinates='axes', color='black')
        ax[0,1].set_title('CO', color='black', y=1.05, fontsize=14)

        # OCS plot
        cf_ocs = ax[1,0].contourf(plobject.lons, plobject.lats, ocs_cube[frame,:,:],
                                  cmap='YlOrBr', vmin=np.min(ocs_cube), vmax=4.5)
        q3 = ax[1,0].quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax[1,0].quiverkey(ax[1,0].quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label=f'{qscale*10} m/s',
                 labelpos='E', coordinates='axes', color='black')
        ax[1,0].set_title('OCS', color='black', y=1.05, fontsize=14)
        ax[1,0].set_ylabel('Latitude / deg')
        ax[1,0].set_xlabel('Longitude / deg')

        # HCl plot
        cf_hcl = ax[1,1].contourf(plobject.lons, plobject.lats, hcl_cube[frame,:,:],
                                  cmap='Reds', vmin=np.min(hcl_cube), vmax=0.6)
        q4 = ax[1,1].quiver(X[::n, ::n], Y[::n, ::n], -u[frame,::n,::n],
                   v[frame,::n,::n], **quiv_args)
        ax[1,1].quiverkey(ax[1,1].quiver(X[::n, ::n], Y[::n, ::n], -u[0,::n,::n],
                   v[0,::n,::n], **quiv_args), X=0.9, Y=1.05, U=qscale*10, label=f'{qscale*10} m/s',
                 labelpos='E', coordinates='axes', color='black')
        ax[1,1].set_title('HCl', color='black', y=1.05, fontsize=14)
        ax[1,1].set_xlabel('Longitude / deg')
        plt.subplots_adjust(wspace=0.1)
        fig.suptitle(f'Volcanic plume at {height} km, {time_axis[frame]} hrs', y=0.97, fontsize=24)

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=range(0,tf-t0), interval=200, repeat=False)
    
    # Add colorbars
    cbar_h2o = plt.colorbar(ax[0,0].contourf(plobject.lons, plobject.lats, h2o_cube[0,:,:],
                                             cmap='Blues', vmin=np.min(h2o_cube), vmax=45.0),
                                             ax=ax[0,0])
    cbar_h2o.set_label('ppm', color='black')
    
    cbar_co = plt.colorbar(ax[0,1].contourf(plobject.lons, plobject.lats, co_cube[0,:,:],
                                             cmap='Purples', vmin=np.min(co_cube), vmax=37.5), ax=ax[0,1])
    cbar_co.set_label('ppm', color='black')
    
    cbar_ocs = plt.colorbar(ax[1,0].contourf(plobject.lons, plobject.lats, ocs_cube[0,:,:],
                                             cmap='YlOrBr', vmin=np.min(ocs_cube), vmax=4.5), ax=ax[1,0])
    cbar_ocs.set_label('ppm', color='black')
    
    cbar_hcl = plt.colorbar(ax[1,1].contourf(plobject.lons, plobject.lats, hcl_cube[0,:,:],
                                             cmap='Reds', vmin=np.min(hcl_cube), vmax=0.6), ax=ax[1,1])
    cbar_hcl.set_label('ppm', color='black')

    # Save the animation as an mp4 file
    ani.save(savepath + f'deep_plume_{height}km.mp4', writer='ffmpeg')
    # ani.save('myanimation.gif', writer='pillow') #alternative

# %%
def summ_stats(plobject, keys, lev, t0, tf, savename='stats.png',
               savepath=None,
               save=False, sformat='png'):
    """
    Compute variability of chemical species (Standard deviation on lon-lat grid).

    Args:
        plobject (PlumeSim): PlumeSim object containing the data.
        keys (str or list): Dictionary key(s) of the data variable(s).
        lev (int): Vertical level index.
        t0 (int): Start time index.
        tf (int): End time index.
        savename (str): Filename for the saved plot. Defaults to 'stats.png'.
        savepath (str): Directory path to save the plot. Defaults to None.
        save (bool): Whether to save the plot. Defaults to False.
        sformat (str): Format to save the plot. Defaults to 'png'.
    """
    if isinstance(keys, str):
        keys = [keys]

    num_rows = len(keys)
    num_cols = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 4*num_rows), sharex=False, sharey=True, tight_layout=True)
    
    # Ensure axes is 2D array [row, col]
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)

    for i, key in enumerate(keys):
        # Extract data for the specified time range and level
        cube = plobject.data[key][t0:tf,lev,:,:]
        
        title_name = cube.long_name or cube.name
        title_height = np.round(plobject.heights[lev],2)
        
        # Convert units based on species
        if key=='n2':
            cube = cube*100
            unit = '%'
        else:
            cube = cube*1e6
            unit = 'ppm'

        # Calculate standard deviation and mean over time
        std = cube.std(dim='time_counter',skipna=True, keep_attrs=True)
        avg = cube.mean(dim='time_counter', skipna=True, keep_attrs=True)

        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        
        # Plot mean abundance
        abd_plot = ax1.contourf(plobject.lons, plobject.lats, avg, cmap='afmhot')
        ax1.set_title(f'Mean {title_name} at {title_height} km')
        ax1.set_ylabel('Latitude / deg')
        cbar1 = plt.colorbar(abd_plot, orientation='horizontal', ax=ax1, pad=0.2)
        cbar1.set_label(unit)
        cbar1.ax.tick_params(rotation=45)

        # Plot coefficient of variation
        std_plot = ax2.contourf(plobject.lons, plobject.lats, 100*std/avg, cmap='copper')
        ax2.set_title(f'Coeff of variation in {title_name} at {title_height} km')
        cbar2 = plt.colorbar(std_plot, orientation='horizontal',ax=ax2, pad=0.2)
        cbar2.set_label('%')
        cbar2.ax.tick_params(rotation=45)
        
        # Only set xlabel on bottom plots
       # if i == num_rows - 1:
        ax1.set_xlabel('Longitude / deg')
        ax2.set_xlabel('Longitude / deg')
    fig.suptitle(f'Background chemical variability at h = {np.round(plobject.heights[lev],2)} km', y=0.99, fontsize=14)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if save:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    plt.show()

# %% Main code block
if __name__ == "__main__":

    # Create PlumeSim object and load data for surface run
    surf_sim = PlumeSim(venusdict, None, 'aoa_surf')
    surf_sim.load_file(surf)
    surf_sim.set_resolution()

    # Figure 1: Zonal mean age of air
    zmage(surf_sim, hmin=0, hmax=None, time_slice=-1,
          convert2yr=True, plume_markers=plume_dict, 
          levels=np.arange(0,50,1), savepath=savedir,
          save=False, sformat=filetype, savename='fig1' + '.' + filetype)
    surf_sim.close()
    del surf_sim

    # Create PlumeSim object and load data for chemistry run
    chem_sim = PlumeSim(venusdict, None, 'chem_4days')
    chem_sim.load_file(chem)
    chem_sim.set_resolution()
    
    # Figure 4: Background variability of H2O at 8 km
    summ_stats(chem_sim, keys='h2o', lev=10, t0=0, tf=None,
               savename='fig4' + '.' + filetype,
               savepath=savedir,
               save=True, sformat=filetype)
    # Figure 7: Background variability of H2O and HCl at 20 km
    summ_stats(chem_sim, keys=['h2o', 'hcl'], lev=14, t0=0, tf=None,
               savename='fig7' + '.' + filetype,
               savepath=savedir,
               save=True, sformat=filetype)
    # Figure 10: Background variability of four gases at 35 km
    summ_stats(chem_sim, keys=['h2o', 'hcl', 'co', 'ocs'], lev=18, t0=0, tf=None,
               savename='fig10' + '.' + filetype,
               savepath=savedir,
               save=True, sformat=filetype)
    # Figure 13: Background variability of SO2 at 70 km
    summ_stats(chem_sim, keys='so2', lev=35, t0=0, tf=None,
               savename='fig13' + '.' + filetype,
               savepath=savedir,
               save=True, sformat=filetype)
    chem_sim.close()
    del chem_sim
    
    # Create PlumeSim object and load data for plume run
    plume_sim = PlumeSim(venusdict, plume_dict, 'scale_1.5')
    plume_sim.load_file(plumes)
    plume_sim.set_resolution()

    # Figure 2: Animation + cover image of H2O plume at 8 km
    # Figure 5: Animation + cover image of H2O and HCl plumes at 20 km
    # Figure 8: Animation + cover image of four gas plumes at 35 km
    # Figure 11: Animation + cover image of SO2 plume at 70 km
    
    # Figure 3: Dispersal time of H2O plumes at 8 km
    dispersal_time(plume_sim, lev=10, keys=['h2o'], lats=[49,82], lons=[92,47], axis_len=500,
                   save=True, savename='fig3' + '.' + filetype, sformat=filetype,
                   savepath=savedir)

    # Figure 6: Dispersal time of H2O and HCl plumes at 20 km
    dispersal_time(plume_sim, lev=14, keys=['h2o', 'hcl'], lats=[49,82], lons=[92,47], axis_len=500,
                   save=True, savename='fig6' + '.' + filetype, sformat=filetype,
                   savepath=savedir)
    # Figure 9: Dispersal time of four gas plumes at 35 km
    dispersal_time(plume_sim, lev=18, keys=['h2o', 'hcl', 'co', 'ocs'], lats=[49,82], lons=[92,47], axis_len=500,
                   save=True, savename='fig9' + '.' + filetype, sformat=filetype,
                   savepath=savedir)
    # Figure 12: Dispersal time of SO2 plumes at 70 km
    dispersal_time(plume_sim, lev=35, keys=['so2'], lats=[49,82], lons=[92,47], axis_len=500,
                   save=True, savename='fig12' + '.' + filetype, sformat=filetype,
                   savepath=savedir)

    plume_sim.close()
    del plume_sim
