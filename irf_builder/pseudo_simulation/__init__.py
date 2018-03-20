import numpy as np

from scipy import interpolate

import irf_builder as irf


def draw_from_distribution(dist, abscis, n_draws=100, k=1):
    """draws events from a given distribution

    Parameters
    ----------
    dist : 1d array
        the distribution the drawn events are supposed to follow
    abscis : 1d array
        the abscissa values corresponding to `dist`
    n_draws : integer (default: 100)
        number of samples to be drawn
    k : integer (default: 1)
        degree of spline interpolation of the cumulative distribution of `dist`

    Returns
    -------
    randomx : 1d array
        array of samples randomly drawn from the provided distribution

    """
    cdf = np.cumsum(dist) / np.sum(dist)

    try:
        unit = abscis.unit
        abscis = abscis.value
        cdf = cdf.value
    except AttributeError:
        unit = 1

    cdfspline = interpolate.splrep(cdf, abscis, k=1)
    randomx = interpolate.splev(np.random.random(n_draws), cdfspline)

    return randomx * unit
