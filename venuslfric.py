""" Data organisation for LFRic-Venus output """

# %%
## Import packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# %%
## Definition of LFRic-Venus model constants
## These mirror the &planet and &extrusion namelists of the Venus
## configuration, so that derived quantities stay consistent with the
## model that produced the output.
## Radius in m, gravity in m/s2, rotation rate in s-1 (negative: Venus is
## retrograde), specific heat and gas constant in SI units, reference
## pressure in Pa, domain height in m.
lfricdict = {'radius': 6.0518e6, 'g': 8.87, 'rotrate': -2.9924e-07,
             'cp': 900.0, 'rd': 191.4, 'p_zero': 9.2e6,
             'domain_height': 9.0e4, 'nlayers': 60,
             'name': 'Venus'}

## Names of the mesh topology variables in LFRic UGRID output.
meshdict = {'face_x': 'Mesh2d_face_x', 'face_y': 'Mesh2d_face_y',
            'edge_x': 'Mesh2d_edge_x', 'edge_y': 'Mesh2d_edge_y',
            'node_x': 'Mesh2d_node_x', 'node_y': 'Mesh2d_node_y',
            'face_nodes': 'Mesh2d_face_nodes',
            'face_edges': 'Mesh2d_face_edges'}

## Candidate names for each physical field. LFRic names diagnostics after
## the XIOS file definition, so a time-meaned run and an instantaneous run
## carry different names for the same quantity.
fielddict = {'u': ['u_mean', 'u_in_w2h', 'u_inst'],
             'v': ['v_mean', 'v_in_w2h', 'v_inst'],
             'theta': ['theta_mean', 'theta', 'theta_inst'],
             'exner': ['exner_mean', 'exner', 'exner_inst'],
             'rho': ['rho_mean', 'rho', 'rho_inst'],
             'dtheta_force': ['dtheta_force_mean', 'dtheta_force']}


### LFRicPlanet class object definition ###
### Manages planet configuration data and LFRic simulation output data.

# %%
class LFRicPlanet:
    """ A Planet object which contains the output data for an LFRic simulation.

    LFRic writes UGRID output on an unstructured cubed-sphere mesh, so the
    horizontal dimensions are cell indices (nMesh2d_face, nMesh2d_edge) rather
    than longitude and latitude. Geolocation is carried alongside in
    Mesh2d_face_x/y and Mesh2d_edge_x/y, both in degrees. This object exposes
    those coordinates and the mesh connectivity so that zonal means and global
    integrals can be taken on the native mesh without regridding. """

    def __init__(self, planetdict, model, run):
        """ Initiates an LFRicPlanet object using the input dictionary of
        planet constants, the name of the model, and the name of the run.
        Model names: lfric
        Run names: free text, e.g. idealised, C24_MG """

        self.name = planetdict['name']
        self.model = model
        self.run = run
        print(f'Welcome to Venus. Your lander will melt in 57 minutes.')
        print(f'This is the {self.run} dataset created by {self.model.upper()}')
        for key, value in planetdict.items():
            setattr(self, key, value)
        self.kappa = self.rd / self.cp

    def identify(self):
        print(f'This is the {self.run} dataset created by {self.model.upper()}')

    def load_file(self, fn):
        """ Loads a netCDF file using the xarray package and stores in object.
            Lists dictionary key, name, dimensions, and shape of each data cube
            and stores text in a reference list. """
        if isinstance(fn, str):
            ds = xr.open_dataset(fn, decode_cf=False)
        elif isinstance(fn, list):
            ds = xr.open_mfdataset(fn, combine='nested', concat_dim='time',
                                   decode_cf=False)
        else:
            print('Improper filename input, must be string or list')
        reflist = []
        str1 = 'File contains:'
        print(str1)
        reflist.append(str1)
        for key in ds.data_vars:
            if 'long_name' in ds[key].attrs:
                keystring = key + ': ' + ds[key].long_name + ', ' + \
                      str(ds[key].dims) + ', ' + \
                      str(ds[key].shape)
            else:
                keystring = key + ': ' + str(ds[key].dims) + ', ' \
                      + str(ds[key].shape)
            print(keystring)
            reflist.append(keystring)
        self.data = ds
        self.reflist = reflist

    def close(self):
        """ Closes netCDF file packaged in LFRicPlanet data object """
        self.data.close()
        print('LFRicPlanet object associated dataset has been closed')

    def contents(self):
        """ Prints reference list for easy formatted oversight of file contents"""
        print(*self.reflist, sep='\n')

    def key(self, field):
        """ Resolves a physical field name onto the variable actually present
        in the file, since XIOS names diagnostics after the file definition. """
        for candidate in fielddict[field]:
            if candidate in self.data.variables:
                return candidate
        raise KeyError(f'No variable in file for {field}; '
                       f'tried {fielddict[field]}')

    def set_resolution(self):
        """ Reads the mesh geolocation and connectivity from the file and
        derives the vertical coordinate from the extrusion namelist values.

        Face and edge latitudes are the scattered cell-centre and edge-midpoint
        positions of the cubed sphere, not a regular axis. """
        self.face_lons = self.data[meshdict['face_x']].values
        self.face_lats = self.data[meshdict['face_y']].values
        self.edge_lons = self.data[meshdict['edge_x']].values
        self.edge_lats = self.data[meshdict['edge_y']].values
        self.node_lons = self.data[meshdict['node_x']].values
        self.node_lats = self.data[meshdict['node_y']].values
        self.face_nodes = self.data[meshdict['face_nodes']].values
        self.face_edges = self.data[meshdict['face_edges']].values

        self.nface = self.face_lats.size
        self.nedge = self.edge_lats.size
        self.time = self.data['time'].values
        if self.time.size > 1:
            self.tinterval = float(np.diff(self.time[0:2])[0])
        else:
            self.tinterval = np.nan

        self.set_vertical()
        self.face_areas()
        print('Resolution is ' + str(self.nface) + ' faces, '
              + str(self.nedge) + ' edges, ' + str(self.nlayers) + ' levs')
        print(f'Vertical axis is height above surface, uniform '
              f'{self.dz:.1f} m layers')

    def set_vertical(self):
        """ Builds height coordinates for the uniform extrusion.

        full_levels and half_levels in the file are level indices, not heights.
        A uniform extrusion places full levels (Wtheta) at k*dz and half levels
        (W3) at (k+0.5)*dz. Heights in km are kept for plotting. """
        self.dz = self.domain_height / self.nlayers
        self.z_full = np.arange(self.nlayers + 1) * self.dz
        self.z_half = (np.arange(self.nlayers) + 0.5) * self.dz
        self.heights_full = self.z_full * 1e-3
        self.heights = self.z_half * 1e-3
        self.vert = self.nlayers
        self.vert_unit = 'km'
        self.vert_axis = 'Height'

    def face_areas(self):
        """ Calculates the spherical area of each mesh face.

        Cubed-sphere cells are not equal-area, so a global integral needs the
        true areas. Each quadrilateral face is split into two spherical
        triangles and the area found from the spherical excess. """
        lon = np.deg2rad(self.node_lons)
        lat = np.deg2rad(self.node_lats)
        xyz = np.stack([np.cos(lat) * np.cos(lon),
                        np.cos(lat) * np.sin(lon),
                        np.sin(lat)], axis=-1)
        corners = xyz[self.face_nodes]
        excess = (self._triangle_excess(corners[:, 0], corners[:, 1], corners[:, 2])
                  + self._triangle_excess(corners[:, 0], corners[:, 2], corners[:, 3]))
        self.areas = excess * self.radius**2
        return self.areas

    @staticmethod
    def _triangle_excess(a, b, c):
        """ Spherical excess of a triangle of unit vectors, by the formula
        E = 2*atan2(|a.(b x c)|, 1 + a.b + b.c + c.a), which stays accurate
        for the small triangles a cubed sphere produces. """
        numer = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
        denom = (1.0 + np.einsum('ij,ij->i', a, b)
                 + np.einsum('ij,ij->i', b, c)
                 + np.einsum('ij,ij->i', c, a))
        return 2.0 * np.arctan2(numer, denom)

    def edges_to_faces(self, field):
        """ Maps an edge-located field onto face centres by averaging the
        edges of each face.

        Winds live in W2h on cell edges while mass lives in W3 at cell centres,
        so any product of the two must be co-located first. The mesh's own
        face-edge connectivity does this without regridding.

        field: array with the edge index as its last dimension. """
        idx = self.face_edges
        valid = idx >= 0
        safe = np.where(valid, idx, 0)
        gathered = field[..., safe]
        gathered = np.where(valid, gathered, np.nan)
        return np.nanmean(gathered, axis=-1)

    def zonal_mean(self, field, lats, nbins=None):
        """ Bins a mesh field into latitude bands and averages within each.

        This is the zonal mean on the native mesh. Face and edge positions are
        scattered in latitude, so a binned average replaces the axis average
        that a regular grid would allow, and avoids interpolating first.

        field: array with the mesh index as its last dimension
        lats:  latitude in degrees for that mesh index
        Returns (bin centres in degrees, binned field). """
        if nbins is None:
            nbins = int(np.sqrt(lats.size / 6.0) * 2)
        edges = np.linspace(-90.0, 90.0, nbins + 1)
        which = np.clip(np.digitize(lats, edges) - 1, 0, nbins - 1)
        counts = np.bincount(which, minlength=nbins)
        flat = field.reshape(-1, lats.size)
        binned = np.stack([np.bincount(which, weights=row, minlength=nbins)
                           for row in flat])
        with np.errstate(invalid='ignore'):
            binned = binned / counts
        binned = binned.reshape(field.shape[:-1] + (nbins,))
        return 0.5 * (edges[:-1] + edges[1:]), binned

    def calc_density(self, time_slice):
        """ Derives air density at cell centres on half levels.

        The Venus diagnostic output carries exner and theta but not rho, so
        density comes from the model's own equation of state:
        p = p_zero*exner**(1/kappa), T = theta*exner, rho = p/(rd*T).
        theta lives on full levels and is averaged onto half levels to meet
        exner. """
        exner = self.data[self.key('exner')][time_slice].values
        theta = self.data[self.key('theta')][time_slice].values
        theta_half = 0.5 * (theta[:-1, :] + theta[1:, :])
        temperature = theta_half * exner
        pressure = self.p_zero * exner**(1.0 / self.kappa)
        return pressure / (self.rd * temperature)

    def calc_relative_am(self, trange=(0, None)):
        """ Calculates the relative angular momentum of the atmosphere.

        M_r = integral of u * r * cos(lat) dm, with r = radius + z for the
        deep atmosphere the configuration uses (shallow=.false.) and
        dm = rho * area * dz. The planetary term is excluded, so the result is
        the wind contribution alone and starts near zero for a run from rest.

        Returns (time in seconds, M_r in kg m2 s-1). """
        if not hasattr(self, 'areas'):
            self.set_resolution()

        ukey = self.key('u')
        nt = self.data[ukey].shape[0]
        stop = nt if trange[1] is None else trange[1]
        indices = np.arange(trange[0], stop)

        radius = self.radius + self.z_half
        coslat = np.cos(np.deg2rad(self.face_lats))
        # Moment arm r*cos(lat) and the cell volume element, both fixed in time.
        arm = radius[:, None] * coslat[None, :]
        volume = self.areas[None, :] * self.dz

        am = np.zeros(indices.size)
        for n, i in enumerate(indices):
            u_edge = self.data[ukey][i].values
            u_face = self.edges_to_faces(u_edge)
            rho = self.calc_density(int(i))
            am[n] = np.nansum(u_face * arm * rho * volume)

        self.am_time = self.time[indices]
        self.am_relative = am
        return self.am_time, self.am_relative


# %%
def zmzw_and_am(plobject, meaning=True, trange=(0, None), time_slice=-1,
                nbins=None, plot=True,
                save=False, saveformat='png', savename='zmzw_and_am.png'):

    """ Input: LFRicPlanet object holding LFRic UGRID output
        Output: two-panel figure, zonal mean zonal wind on the left and
        relative angular momentum against time on the right

        meaning (default True) time-means the wind over trange
        time_slice (default -1) selects the time if meaning=False
        nbins (default None) sets the number of latitude bands, chosen from
        the mesh size if not given

        Eastward wind is plotted with its physical sign. Venus superrotates
        retrograde, so an established superrotation appears as negative u. """

    if not hasattr(plobject, 'areas'):
        plobject.set_resolution()

    zonal = plobject.data[plobject.key('u')]
    if meaning is True:
        wind = np.mean(zonal[trange[0]:trange[1], :, :].values, axis=0)
    else:
        wind = zonal[time_slice, :, :].values

    lats, zmean = plobject.zonal_mean(wind, plobject.edge_lats, nbins=nbins)
    times, am = plobject.calc_relative_am(trange=trange)
    days = times / 86400.0

    if plot is not True:
        return lats, zmean, days, am

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    levels = ax[0].contourf(lats, plobject.heights, zmean,
                            cmap='RdBu_r', norm=TwoSlopeNorm(0))
    ax[0].set_title('Zonal mean zonal wind')
    ax[0].set_xlabel('Latitude [deg]')
    ax[0].set_ylabel('Height [km]')
    cbar = fig.colorbar(levels, ax=ax[0])
    cbar.ax.set_title('m/s')

    ax[1].plot(days, am, color='k')
    ax[1].axhline(0.0, color='gray', linewidth=0.8, linestyle='--')
    ax[1].set_title('Relative angular momentum')
    ax[1].set_xlabel('Time [Earth days]')
    ax[1].set_ylabel('$M_r$ [kg m$^2$ s$^{-1}$]')

    fig.suptitle(f'{plobject.name}, {plobject.run}')
    fig.tight_layout()

    if save is True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
