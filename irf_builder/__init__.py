import numpy as np
from astropy import units as u

# pull in sub-modules
from . import spectra
from . import plotting
from . import sensitivity
from . import event_selection


# pull in and rename some functions into top-level namespace for easier access
from .meta_data_loader import load_meta_data_from_yml as load_meta_data
from .weighting import unbinned_wrapper as make_weights
from .irfs import effective_areas
from .event_selection import minimise_sensitivity_per_bin as optimise_cuts
from .sensitivity import point_source_sensitivity as calculate_sensitivity


# the header entries for the simulated and reconstructed energies
# and offset angle
energy_names = {"mc": "MC_Energy",
                "reco": "reco_Energy"}
offset_angle_name = "off_angle"

# your favourite units here
angle_unit = u.deg
energy_unit = u.TeV
flux_unit = (u.erg * u.cm**2 * u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

observation_time = 50 * u.h

# factor by which the radial "Theta-cut" is larger for the off- than for the on-region
r_scale = 1
# the ratio between the area of the off-region over the on-region (alpha = r_scale**-2)
alpha = 1

# define edges to sort events in
e_bin_edges = np.logspace(-2, 2.5, 20) * u.TeV
e_bin_centres = np.sqrt(e_bin_edges[:-1] * e_bin_edges[1:])
e_bin_edges_fine = np.logspace(-2, 2.5, 100) * u.TeV
e_bin_centres_fine = np.sqrt(e_bin_edges_fine[:-1] * e_bin_edges_fine[1:])


# use `meta_data_loader` to put information concerning the MC production in here
meta_data = {"units": {}, "gamma": {}, "proton": {}, "electron": {}}
