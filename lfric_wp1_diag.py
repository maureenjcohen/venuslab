""" Diagnostics for LFRic-Venus variable heat capacity (WP1) runs """

# %%
## Import packages
import numpy as np
import matplotlib.pyplot as plt

# %%
## Parameters of the heat capacity law c_p(T) = cp_law_ref*(T/cp_law_t0)**nu.
## These mirror the venus_cp_law_* constants in LFRic's variable_cp_mod,
## so that derived quantities match the model's variable_cp formulation.
## Reference heat capacity in J/kg/K, reference temperature in K.
cplawdict = {'cp_law_ref': 1000.0, 'cp_law_t0': 460.0,
             'cp_law_exponent': 0.35}


def cp_law(temperature):
    """ Heat capacity at constant pressure (J/kg/K) at temperature (K). """
    return cplawdict['cp_law_ref'] * \
        (temperature / cplawdict['cp_law_t0'])**cplawdict['cp_law_exponent']


def calc_temperature(plobject, time_slice=-1):
    """ Temperature at cell centres on half levels, T = theta*exner.

    Depending on the XIOS file definition, theta is written either on full
    levels (Wtheta) or already mapped to half levels (W3). Full-level theta
    is averaged onto half levels to meet exner. """
    theta = plobject.data[plobject.key('theta')][time_slice].values
    exner = plobject.data[plobject.key('exner')][time_slice].values
    if theta.shape[0] == exner.shape[0] + 1:
        theta = 0.5 * (theta[:-1, :] + theta[1:, :])
    return theta * exner


def calc_static_stability(plobject, time_slice=-1, constant_cp=False):
    """ Static stability S = dT/dz + g(z)/c_p in K/km on the interior full
    levels, one value per column.

    constant_cp=False uses the c_p(T) law, the adiabat a parcel follows
    when variable_cp is on. constant_cp=True uses the namelist cp, the
    adiabat of the constant heat capacity model. S < 0 is convectively
    unstable under the chosen adiabat.

    Gravity falls off with height as g*(a/(a+z))**2, matching the deep
    atmosphere geopotential of the configuration (shallow=.false.). """
    if not hasattr(plobject, 'areas'):
        plobject.set_resolution()

    temperature = calc_temperature(plobject, time_slice)
    dtdz = np.diff(temperature, axis=0) / np.diff(plobject.z_half)[:, None]
    t_interface = 0.5 * (temperature[1:] + temperature[:-1])
    z = plobject.z_full[1:-1]
    gravity = plobject.g * (plobject.radius / (plobject.radius + z))**2
    if constant_cp is True:
        cp = plobject.cp
    else:
        cp = cp_law(t_interface)
    return (dtdz + gravity[:, None] / cp) * 1e3


# %%
def static_stability(plobjects, time_slice=-1, constant_cp=(True, False),
                     plot=True, save=False, saveformat='png',
                     savename='static_stability.png'):

    """ Input: list of LFRicPlanet objects, e.g. the variable_cp off and on
        runs, each holding LFRic UGRID output
        Output: global area-weighted mean static stability profiles of all
        runs on one plot, labelled by run name

        time_slice (default -1) selects the time
        constant_cp gives one flag per run, in the order of plobjects: True
        evaluates S with the namelist cp, False with the c_p(T) law. The
        default (True, False) suits [off, on], judging each run by the
        adiabat its own dynamics follow.

        The mean is taken over columns of S, not S of the mean temperature,
        because S depends nonlinearly on T through c_p(T). """

    if len(constant_cp) != len(plobjects):
        raise ValueError('constant_cp needs one flag per run')

    profiles = []
    for plobject, const in zip(plobjects, constant_cp):
        stability = calc_static_stability(plobject, time_slice, const)
        weights = plobject.areas / np.sum(plobject.areas)
        profiles.append(np.sum(stability * weights[None, :], axis=1))

    heights = plobjects[0].heights_full[1:-1]

    if plot is not True:
        return heights, profiles

    fig, ax = plt.subplots(figsize=(6, 7))
    for plobject, profile, const in zip(plobjects, profiles, constant_cp):
        if const is True:
            adiabat = f'constant $c_p$ = {plobject.cp:.0f} J/kg/K'
        else:
            adiabat = '$c_p(T)$'
        ax.plot(profile, heights, label=f'{plobject.run}, {adiabat}')
    ax.axvline(0.0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title('Global mean static stability')
    ax.set_xlabel('$dT/dz + g/c_p$ [K/km]')
    ax.set_ylabel('Height [km]')
    ax.legend()
    fig.suptitle(f'{plobjects[0].name}, time index {time_slice}')
    fig.tight_layout()

    if save is True:
        plt.savefig(savename, format=saveformat, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
