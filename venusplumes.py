# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def find_plume(plobject, lev, key):
    """ Find array indices of maximum tracer value """

    cube = plobject.data[key][:,lev,:,:]
    # Extract age of air tracer cube at desired model level
    plume_mmr = cube.max() 
    # What is the peak mmr of the plume injection?
    mask = np.where(cube >= plume_mmr)
    print(mask)
    start_time = mask[0][0]
    # Get 0th element of 0th dimension (first time occurrence)
    end_time = mask[0][-1]
    # Get last element of 0th dimension (last time occurrence)
    lat_idx = mask[1][0]
    # Get 0th element of 1st dimension (latitude)  
    # tho all should be identical
    lon_idx = mask[2][0]
    # Get 0th element of 2nd dimension (longitude)
    # tho all should be identical

    return start_time, end_time, lat_idx, lon_idx


# %%
def dispersal_time(plobject, key, lev,
                   save=False,
                   savename='plume_dispersal.png',
                   savepath='/exomars/projects/mc5526/VPCM_volcanic_plumes/scratch_plots/',
                   sformat='png'):
    """ Find time in seconds/hours for the tracer value in the
        gridbox to return to the background value       """
    
    cube = plobject.data[key][:,lev,:,:]
    # Extract age of air tracer data at desired model level
    start_time, end_time, lat_idx, lon_idx = find_plume(plobject, lev)
    # Get time and space indices of plume
    # lon_idx = 26
    threshold_val = cube[start_time-1, lat_idx, lon_idx].values
    # Get background value of tracer before plume starts
    threshold_val = 1e-08
    counter = 0
    for t in range(end_time, cube.shape[0]):
        if cube[t, lat_idx, lon_idx].values > threshold_val:
            counter = counter + 1
        # Count how many time steps tracer value remains above background
        else:
            break
        # Stop loop when tracer returns to background

    interval = np.diff(cube.time_counter.values)[0]
    # Time interval in seconds between each output
    disp_time = counter * interval
    # Total time in seconds until tracer valuer returns to background
    disp_hours = disp_time / (60*60)
    # Convert to hours for convenience

    time_axis = np.arange(start_time-5, end_time+counter+5)*interval/(60*60)
    # Create time axis in hours

    data = cube[start_time-5:end_time+counter+5, lat_idx, lon_idx]
    # Get data, including 5 time steps before and after plume

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(time_axis, data, color='blue', label='Plume enhancement')
    ax.plot(time_axis, np.ones_like(data)*threshold_val, color='red', 
            linestyle='dashed', label='Background value')
    ax.set_title(f'Plume dispersal time = {np.round(disp_hours,2)} hours')
    ax.set_ylabel('Tracer mmr / kg/kg')
    ax.set_ylim([threshold_val*0.8, cube.max()*1.2])
    ax.set_xlabel('Time after simulation start / hours')
    plt.legend()

    if save==True:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    plt.show()

# %%
def dispersal_map(plobject, lev, key, threshold_val,
                   save=False,
                   savename='dispersal_map.png',
                   savepath='/exomars/projects/mc5526/VPCM_volcanic_plumes/scratch_plots/',
                   sformat='png'):
    
    cube = plobject.data[key][:,lev,:,:]
    interval = np.diff(cube.time_counter.values)[0]
    # Extract age of air tracer data at desired model level
    start_time, end_time, lat_idx, lon_idx = find_plume(plobject, lev)
    # Get time and space indices of plume
    post_eruption = cube[end_time:,:,:]
    # Only consider data after plume forcing has finished
    mask = post_eruption.values > threshold_val
    # Boolean array with same dimension as cube, True is where values are above threshold
    flattened = np.count_nonzero(mask, axis=0)
    # Count how many time outputs are True for each lon x lat point
    map_hours = flattened * interval / (60*60)
    # Convert time outputs to hours

    fig, ax = plt.subplots(figsize=(8,6))
    cf = ax.contourf(plobject.lons, plobject.lats, map_hours, cmap='Blues')
    ax.plot(plobject.lons[lon_idx], plobject.lats[lat_idx], 'ro', label='Eruption')
    ax.set_title('Plume dispersal time')
    ax.set_xlabel('Longitude / deg')
    ax.set_ylabel('Latitude / deg')
    cbar = plt.colorbar(cf)
    cbar.set_label('Hours')
    plt.legend()

    if save==True:
        plt.savefig(savepath + savename, format=sformat, bbox_inches='tight')

    plt.show()
# %%
