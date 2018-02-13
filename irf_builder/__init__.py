import numpy as np
from astropy import units as u

# pull in sub-modules
from . import writer
from . import spectra
from . import plotting
from . import sensitivity
from . import event_selection


# pull in and rename some functions into top-level namespace for easier access
from .meta_data_loader import load_meta_data_from_yml as load_meta_data
from .weighting import unbinned_wrapper as make_weights
from .irfs import effective_areas
from .event_selection import minimise_sensitivity_per_bin as optimise_cuts
from .sensitivity import differential_energy_point_source as calculate_sensitivity


# the header entries for a number of quantities
energy_names = {"mc": "MC_Energy",
                "reco": "reco_Energy"}
reco_error_name = "xi"
offset_angle_name = "off_angle"


# your favourite units here
angle_unit = u.deg
energy_unit = u.TeV
flux_unit = (u.erg * u.cm**2 * u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

observation_time = 50 * u.h


# define edges to sort events in
e_bin_edges = np.logspace(-2, 2.5, 20) * u.TeV
e_bin_centres = np.sqrt(e_bin_edges[:-1] * e_bin_edges[1:])
e_bin_edges_fine = np.logspace(-2, 2.5, 100) * u.TeV
e_bin_centres_fine = np.sqrt(e_bin_edges_fine[:-1] * e_bin_edges_fine[1:])


# use `meta_data_loader` to put information concerning the MC production in here
meta_data = {"units": {}, "gamma": {}, "proton": {}, "electron": {}}


# factor by which the radial "Theta-cut" is
# larger for the off- than for the on-region
r_scale = 1

# the ratio between the area of the off-region
# over the on-region (alpha = r_scale**-2)
alpha = 1


class RegionScaler:
    '''tiny property-wrapper class to simultaneously set `r_scale` and `alpha`
    and ensure consistency
    '''
    @property
    def r_scale(self):
        global r_scale
        return r_scale

    @r_scale.setter
    def r_scale(self, val):
        global r_scale
        global alpha
        r_scale = val
        alpha = val**-2

    @property
    def alpha(self):
        global alpha
        return alpha

    @alpha.setter
    def alpha(self, val):
        global r_scale
        global alpha
        alpha = val
        r_scale = val**-.5
