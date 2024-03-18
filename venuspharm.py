# %%
import numpy as np
import matplotlib.pyplot as plt
import windspharm


# %%
def decomposition(plobject, meaning=True, time_slice=-1, level=35, qscale=1, 
                  qmultiplier=1, n=3):

    """ Perform a Helmholtz decomposition of the horizontal wind field at input 
        level. """

    u = -np.copy(plobject.data['vitu'][:,level,:,:])
    v = np.copy(plobject.data['vitv'][:,level,:,:])

    if meaning==True:
        u = np.mean(u, axis=0)
        v = np.mean(v, axis=0)
    else: 
        u = u[time_slice,:,:]
        v = v[time_slice,:,:]

    winds = windspharm.standard.VectorWind(u, v)
    # Create a VectorWind data object from the x and y wind cubes
    uchi, vchi, upsi, vpsi = winds.helmholtz(truncation=21)
    # Calculate the Helmholtz decomposition. Truncation is set to 21 because 
    # this is what Hammond and Lewis 2021 used.
    zonal_upsi = np.mean(upsi, axis=-1)
    zonal_vpsi = np.mean(vpsi, axis=-1)
    zonal_upsi = np.tile(zonal_upsi, (len(plobject.lons), 1))
    zonal_vpsi = np.tile(zonal_vpsi, (len(plobject.lons), 1))
    eddy_upsi = upsi - zonal_upsi.T
    eddy_vpsi = vpsi - zonal_vpsi.T

    X, Y = np.meshgrid(plobject.lons, plobject.lats)
    fig1, ax1 = plt.subplots(figsize=(8,5))
    q1 = ax1.quiver(X[::n,::n], Y[::n,::n], uchi[::n,::n], vchi[::n,::n],
                  angles='xy', scale_units='xy', scale=qscale)
    ax1.quiverkey(q1, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %(qscale*10), 
                  labelpos='E', coordinates='axes')
    plt.title(f'Divergent component of wind, h={int(plobject.heights[level])} km')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8,5))
    q2 = ax2.quiver(X[::n,::n], Y[::n,::n], upsi[::n,::n], vpsi[::n,::n],
                  angles='xy', scale_units='xy', scale=(qscale*qmultiplier))
    ax2.quiverkey(q2, X=0.9, Y=1.05, U=(qscale*qmultiplier*10), 
                  label='%s m/s' %(qscale*qmultiplier*10), 
                  labelpos='E', coordinates='axes')
    plt.title(f'Rotational component of wind, h={int(plobject.heights[level])} km')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(8,5))
    q3 = ax3.quiver(X[::n,::n], Y[::n,::n], eddy_upsi[::n,::n], eddy_vpsi[::n,::n],
                  angles='xy', scale_units='xy', scale=(qscale*qmultiplier))
    ax3.quiverkey(q3, X=0.9, Y=1.05, U=(qscale*qmultiplier*10), 
                  label='%s m/s' %(qscale*qmultiplier*10), 
                  labelpos='E', coordinates='axes')
    plt.title(f'Eddy rotational component of wind, h={int(plobject.heights[level])} km')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(8,5))
    q4 = ax4.quiver(X[::n,::n], Y[::n,::n], u[::n,::n], v[::n,::n],
                  angles='xy', scale_units='xy', scale=(qscale*qmultiplier))
    ax4.quiverkey(q4, X=0.9, Y=1.05, U=(qscale*qmultiplier*10), 
                  label='%s m/s' %(qscale*qmultiplier*10), 
                  labelpos='E', coordinates='axes')
    plt.title(f'Wind vectors, h={int(plobject.heights[level])} km')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.show()

# %%
