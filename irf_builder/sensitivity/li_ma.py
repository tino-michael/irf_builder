import numpy as np


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
