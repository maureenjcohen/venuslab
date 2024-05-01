# %%
import numpy as np
import matplotlib.pyplot as plt
import windspharm
from matplotlib.colors import TwoSlopeNorm


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
    print(zonal_vpsi.shape)
    zonal_upsi = np.tile(zonal_upsi, (len(plobject.lons), 1))
    zonal_vpsi = np.tile(zonal_vpsi, (len(plobject.lons), 1))
    eddy_upsi = upsi - zonal_upsi.T
    eddy_vpsi = vpsi - zonal_vpsi.T
    print(zonal_vpsi.shape, zonal_vpsi.T.shape)

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
def divergence(plobject, lev, time_slice, qscale=0.5, 
                  qmultiplier=1, n=3):
    """ Plot divergence only
        Maybe with temperature or eddy temperature """
    
    u = -np.flip(plobject.data['vitu'][time_slice,lev,:,:], axis=(0,1))
    v = np.flip(plobject.data['vitv'][time_slice,lev,:,:], axis=(0,1))
    #u, v = windspharm.tools.reverse_latdim(u,v)

    temp = plobject.data['temp'][time_slice,lev,:,:]
    eddy_temp = temp - np.mean(temp, axis=-1)[:,np.newaxis]

    winds = windspharm.standard.VectorWind(u, v, rsphere=plobject.radius*1000)
    div = np.flip(winds.divergence(truncation=21), axis=(0,1))
    uchi, vchi, upsi, vpsi = winds.helmholtz(truncation=21)

    X, Y = np.meshgrid(plobject.lons, plobject.lats)
    fig, ax = plt.subplots(figsize=(8,5))
    cf = ax.contourf(plobject.lons, plobject.lats, div,
                     cmap='coolwarm', norm=TwoSlopeNorm(0))
    q = ax.quiver(X[::n,::n], Y[::n,::n], 
                  np.flip(uchi[::n,::n], axis=(0,1)), np.flip(vchi[::n,::n], axis=(0,1)),
                  angles='xy', scale_units='xy', scale=qscale)
    ax.quiverkey(q, X=0.9, Y=1.05, U=qscale*10, label='%s m/s' %(qscale*10), 
                  labelpos='E', coordinates='axes')
    ax.set_title(f'Divergence, h={np.round(plobject.heights[lev],0)} km')
    cbar = plt.colorbar(cf, ax=ax)
    plt.show()

# %%
def divaltlon(plobject, lat, time_slice, hmin, hmax):
    """ Altitude-longitude plot of divergence"""

    u = -np.flip(plobject.data['vitu'][time_slice,hmin:hmax,:,:], axis=(1,2))
    v = np.flip(plobject.data['vitv'][time_slice,hmin:hmax,:,:], axis=(1,2))
    u = np.transpose(u, (1,2,0))
    v = np.transpose(v, (1,2,0))

    winds = windspharm.standard.VectorWind(u, v, rsphere=plobject.radius*1000)
    div = np.flip(np.transpose(winds.divergence(truncation=21), (2,0,1)), axis=(1,2))

    fig, ax = plt.subplots(figsize=(8,6))
    cf = ax.contourf(plobject.lons, plobject.heights[hmin:hmax], div[:,lat,:],
                    cmap='seismic', norm=TwoSlopeNorm(0))
    ax.set_title(f'Divergence, lat {np.round(plobject.lats[lat],0)}')
    ax.set_xlabel('Longitude / deg')
    ax.set_ylabel('Altitude / km')
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Div / s-1', loc='center')
    plt.show()

# %%
def divhov(plobject, lat, lon, hmin, hmax, trange=(1700,1800)):
    """ Hovmoeller plot of divergence """

    u = -np.flip(plobject.data['vitu'][trange[0]:trange[1],hmin:hmax,:,:], axis=(2,3))
    v = np.flip(plobject.data['vitv'][trange[0]:trange[1],hmin:hmax,:,:], axis=(2,3))
    u = np.transpose(u, (0,2,3,1))
    v = np.transpose(v, (0,2,3,1))

    div_list = []
    for t in range(0, trange[1]-trange[0]):
        winds = windspharm.standard.VectorWind(u[t,:,:,:], v[t,:,:,:], rsphere=plobject.radius*1000)
        div = np.flip(np.transpose(winds.divergence(truncation=21), (2,0,1)), axis=(1,2))
        div_list.append(div)

    div_cube = np.array(div_list)
    time_axis = np.arange(0,len(plobject.data['time_counter'][trange[0]:trange[1]])) 

    fig, ax = plt.subplots(figsize=(8,5))
    cf = ax.contourf(time_axis, plobject.heights[hmin:hmax], div_cube[:,:,lat,lon].T,
                    cmap='coolwarm', norm=TwoSlopeNorm(0))
    ax.set_title(f'Divergence, lat {np.round(plobject.lats[lat],0)}, lon {np.round(plobject.lons[lon],0)}')
    ax.set_xlabel('Time / ')
    ax.set_ylabel('Altitude / km')
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Div / s-1', loc='center')
    plt.show()
    

# %%
