import numpy as np
from astropy import units as u
from astropy.units.core import UnitTypeError

import scipy.integrate as integrate

from gammapy.spectrum.cosmic_ray import cosmic_ray_flux
from gammapy.spectrum.crab import CrabSpectrum


__all__ = ["crab_source_rate",
           "cr_background_rate",
           "electron_spectrum"]


def crab_source_rate_gammapy(energy, ref='magic_lp'):
    crab = CrabSpectrum(ref).model
    return crab(energy)


def cr_background_rate_gammapy(energy):
    return cosmic_ray_flux(energy, particle="proton")


def electron_spectrum_gammapy(energy):
    return cosmic_ray_flux(energy, particle="electron")


crab_source_rate = crab_source_rate_gammapy
cr_background_rate = cr_background_rate_gammapy
electron_spectrum = electron_spectrum_gammapy


def e_minus_2(energy, unit=u.TeV):
    '''
    boring, old, unnormalised E^-2 spectrum

    Parameters
    ----------
    energy : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    flux : astropy quantity
        differential flux at E

    '''
    return (energy / unit)**(-2) / (unit * u.s * u.m**2)


def make_mock_event_rate(spectrum, bin_edges, log_e=False, norm=None):
    """
    Creates a histogram with a given binning and fills it according to a spectral function

    Parameters
    ----------
    spectrum : function object
        function of the differential spectrum that shall be sampled into the histogram
        ought to take the energy as an astropy quantity as sole argument
    bin_edges : numpy array, optional (default: None)
        bin edges of the histogram that is to be filled
    log_e : bool, optional (default: False)
        tell if the values in `bin_edges` are given in logarithm
    norm : float, optional (default: None)
        normalisation factor for the histogram that's being filled
        sum of all elements in the array will be equal to `norm`

    Returns
    -------
    rates : numpy array
        histogram of the (non-energy-differential) event rate of the proposed spectrum
    """

    def spectrum_value(e):
        """
        `scipy.integrate` does not like units during integration. use this as a quick fix
        """
        try:
            return spectrum(e).value
        except UnitTypeError:
            # in case `spectrum` function insists on unified energy
            return spectrum(e * u.TeV).value

    rates = []
    if log_e:
        bin_edges = 10**bin_edges
    for l_edge, h_edge in zip(bin_edges[:-1], bin_edges[1:]):
        bin_events = integrate.quad(
            spectrum_value, l_edge.value, h_edge.value)[0]
        rates.append(bin_events)

    # units have been strip for the integration. the unit of the result is the unit of the
    # function: spectrum(e) times the unit of the integrant: e -- for the latter use the
    # first entry in `bin_edges`
    rates = np.array(rates) * spectrum(bin_edges[0]).unit * bin_edges[0].unit

    # if `norm` is given renormalise the sum of the `rates`-bins to this value
    if norm:
        rates *= norm / np.sum(rates)

    return rates
