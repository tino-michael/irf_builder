import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.optimize import minimize

from matplotlib import pyplot as plt

import irf_builder as irf
from irf_builder import spectra


def sigma_lima(n_on, n_off, alpha=0.2):
    """
    Compute the significance according to Eq. (17) of Li & Ma (1983).

    Parameters
    ----------
    n_on : integer/float
        Number of on counts
    n_off : integer/float
        Number of off counts
    alpha : float, optional (default: 0.2)
        Ratio of on-to-off exposure

    Returns
    -------
    sigma : float
        the significance of the given off and on counts
    """

    alpha1 = alpha + 1.0
    n_sum = n_on + n_off
    arg1 = n_on / n_sum
    arg2 = n_off / n_sum
    term1 = n_on
    term2 = n_off
    if n_on > 0:
        term1 *= np.log((alpha1 / alpha) * arg1)
    if n_off > 0:
        term2 *= np.log(alpha1 * arg2)
    sigma = np.sqrt(2.0 * (term1 + term2))

    return sigma


def diff_to_x_sigma(scale, n, alpha, x=5):
    """
    calculates the significance according to `sigma_lima` and returns the squared
    difference to `x`. To be used in a minimiser that determines the necessary source
    intensity for a detection of given significance of `x` sigma
    The square here is only to have a continuously differentiable function with a smooth
    turning point at the minimum -- in contrast to an absolute-function that makes a
    sharp turn at the minimum.

    Parameters
    ----------
    scale : python list with a single float
        this is the variable in the minimisation procedure
        it scales the number of gamma events
    n : shape (2) list
        the signal count in the on- (index 0)
        and the background count int the off-region (index 1)
        the events in the on-region are to be scaled by `scale[0]`
        the background rate in the on-region is estimated as `alpha` times off-count
    alpha : float
        the ratio of the on and off areas
    x : float, optional (default: 5)
        target significance in multiples of "sigma"

    Returns
    -------
    (sigma - x)**2 : float
        squared difference of the significance to `x`
        minimise this function for the number of gamma events needed for your desired
        significance of `x` sigma
    """

    n_on = n[1] * alpha + n[0] * scale[0]
    n_off = n[1]
    sigma = sigma_lima(n_on, n_off, alpha)
    return (sigma - x)**2


def point_source_sensitivity(events, energy_bin_edges,
                             alpha, signal_list=("g"), mode="MC",
                             sensitivity_source_flux=spectra.crab_source_rate,
                             n_draws=1000):
    """
    Calculates the sensitivity to a point-source

    Parameters
    ----------
    alpha : float
        area-ratio of the on- over the off-region
    energy_bin_edges : numpy array
        array of the bin edges for the sensitivity calculation
    signal_list : iterable of strings, optional (default: ("g"))
        list of keys to consider as signal channels
    mode : string ["MC", "Data"] (default: "MC")
        interprete the signal/not-signal channels in all the dictionaries as
        gamma/background ("MC") or as on-region/off-region ("Data")
        - if "MC":
            the signal channel is taken as the part comming from the source and
            the background channels multiplied by `alpha` is used as the background
            part in the on-region; the background channels themselves are taken as
            coming from the off-regions
        - if "Data":
            the signal channel is taken as the counts reconstructed in the on-region
            the counts from the background channels multiplied by `alpha` are taken as
            the background estimate for the on-region
    sensitivity_source_flux : callable, optional (default: `crab_source_rate`)
        function of the flux the sensitivity is calculated with
    n_draws : int, optional (default: 1000)
        number of random draws to calculate uncertainties on the sensitivity

    Returns
    -------
    sensitivities : astropy.table.Table
        the sensitivity for every energy bin of `energy_bin_edges`

    """

    # sensitivities go in here
    sensitivities = Table(
        names=("Energy", "Sensitivity", "Sensitivity_low", "Sensitivity_up"))
    try:
        sensitivities["Energy"].unit = energy_bin_edges.unit
    except AttributeError:
        sensitivities["Energy"].unit = irf.energy_unit
    sensitivities["Sensitivity"].unit = irf.flux_unit
    sensitivities["Sensitivity_up"].unit = irf.flux_unit
    sensitivities["Sensitivity_low"].unit = irf.flux_unit

    try:
        # trying if every channel has a `weight` column
        for ev in events.values():
            ev["weight"]

        # in case we do have event weights, we sum them within the energy bin

        def sum_events(ev, mask):
            return np.sum(ev["weight"][mask]) if len(ev["weight"][mask]) > 1 else 0

    except KeyError:
        # otherwise we simply check the length of the masked energy array
        # since the weights are 1 here, `sum_events` is the same as `len(events)`
        def sum_events(ev, mask):
            return len(ev[e_mask])

    # loop over all energy bins
    # the bins are spaced logarithmically: use the geometric mean as the bin-centre,
    # so when plotted logarithmically, they appear at the middle between
    # the bin-edges
    for elow, ehigh, emid in zip(energy_bin_edges[:-1],
                                 energy_bin_edges[1:],
                                 np.sqrt(energy_bin_edges[:-1] *
                                         energy_bin_edges[1:])):

        S_events = np.zeros(2)  # [on-signal, off-background]
        N_events = np.zeros(2)  # [on-signal, off-background]

        # count the (weights of the) events in the on and off regions for this
        # energy bin
        for ch, ev in events.items():
            # single out the events in this energy bin
            e_mask = (ev[irf.energy_names["reco"]] > elow) & \
                     (ev[irf.energy_names["reco"]] < ehigh)

            # we need the actual number of events to estimate the statistical error
            N_events[0 if ch in signal_list else 1] += len(ev[e_mask])
            S_events[0 if ch in signal_list else 1] += sum_events(ev, e_mask)

        # If we have no counts in the on-region, there is no sensitivity.
        # If on data the background estimate from the off-region is larger than the
        # counts in the on-region, `sigma_lima` will break! Skip those
        # cases, too.
        if N_events[0] <= 0:
            continue

        if mode.lower() == "data":
            # the background estimate for the on-region is `alpha` times the
            # background in the off-region
            # if running on data, the signal estimate for the on-region is the counts
            # in the on-region minus the background estimate for the on-region
            S_events[0] -= S_events[1] * alpha
            N_events[0] -= N_events[1] * alpha

        MC_scale = S_events / N_events

        scales = []
        # to get the proper Poisson fluctuation in MC, draw the events with
        # `N_events` as lambda and then scale the result to the weighted number
        # of expected events
        if n_draws > 0:
            trials = np.random.poisson(N_events, size=(n_draws, 2))
            trials = trials * MC_scale
        else:
            # if `n_draws` is zero or smaller, don't do any draws and just take the
            # numbers that we have
            trials = [S_events]

        for trial_events in trials:
            # find the scaling factor for the gamma events that gives a 5 sigma
            # discovery in this energy bin
            scale = minimize(diff_to_x_sigma, [1e-3],
                             args=(trial_events, alpha),
                             method='L-BFGS-B', bounds=[(1e-4, None)],
                             options={'disp': False}
                             ).x[0]

            scales.append(scale)

        # get the scaling factors for the median and the 1sigma containment region
        scale = np.percentile(scales, (50, 32, 68))

        # get the flux at the bin centre
        flux = sensitivity_source_flux(emid).to(irf.flux_unit)

        # and scale it up by the determined factors
        sensitivity = flux * scale

        # store results in table
        sensitivities.add_row([emid, *sensitivity])

    return sensitivities


def plot_sensitivity(sensitivities):

    for mode, sensitivitiy in sensitivities.items():
        sens_low, sens_up = (
            (sensitivitiy["Sensitivity"] -
             sensitivitiy["Sensitivity_low"]).to(irf.flux_unit) *
            sensitivitiy["Energy"].to(u.erg)**2,
            (sensitivitiy["Sensitivity_up"] -
             sensitivitiy["Sensitivity"]).to(irf.flux_unit) *
            sensitivitiy["Energy"].to(u.erg)**2)

        plt.errorbar(
            sensitivitiy["Energy"] / irf.energy_unit,
            (sensitivitiy["Sensitivity"] *
             sensitivitiy["Energy"].to(u.erg)**2).to(irf.sensitivity_unit).value,
            (sens_low.value, sens_up.value),
            color=irf.plotting.mode_colour_map[mode],
            marker="s",
            label=irf.plotting.mode_map[mode])

    # plt.legend(title=f"Obsetvation Time: {irf.observation_time}", loc=1)
    plt.xlabel(r'$E_\mathrm{reco}$' + ' / {:latex}'.format(irf.energy_unit))
    plt.ylabel(r'$E^2 \Phi /$ {:latex}'.format(irf.sensitivity_unit))
    plt.gca().set_xscale("log")
    plt.grid()
    plt.xlim([1e-2, 2e2])
    plt.ylim([5e-15, 5e-10])
