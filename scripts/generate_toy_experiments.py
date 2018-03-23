#!/usr/bin/env python

import sys
from os.path import expandvars, split
import argparse

import numpy as np
from astropy import units as u
from astropy.table import Table

from scipy import interpolate

import irf_builder as irf
from irf_builder.spectra import crab_source_rate, cr_background_rate, electron_spectrum
from irf_builder.pseudo_simulation import draw_from_distribution


from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='')
parser.add_argument('--outdir',
                    default=expandvars("$CTA_SOFT/data/toy_mc/"),
                    help="directory to look up the input files")
parser.add_argument('--outfile', type=str, default="classified_events",
                    help="base of the input files' name before mode and channel")
parser.add_argument('--meta_file', type=str, default="meta_data.yml",
                    help="name of the config file that contains information\n"
                         "(energy range, number of generated events etc.)")
parser.add_argument('-c', '--config', type=str,
                    default=expandvars("$CTA_SOFT/irf_builder/irf_builder/"
                                       "pseudo_simulation/distributions.tex"),
                    help="config file containing a table with energy-dependent"
                         " resolutions for several distributions")

args = parser.parse_args()

# reading the meta data that describes the MC production
irf.meta_data = irf.load_meta_data(f"{args.outdir}/{args.meta_file}")


# TODO
# simulate
# - xi ✔
# - off_angle ✔
# - MC Energy ✔ -> reco Energy ✔
# - gammaness ✔
#
# add
# - energy-dependent selection efficiency


table = Table.read(args.config, format="ascii.latex")

# simulate energy distributions for the three different channels
mc_ener_gam = draw_from_distribution(crab_source_rate(irf.e_bin_centres_fine),
                                     abscis=irf.e_bin_centres_fine,
                                     n_draws=irf.meta_data["gamma"]["n_simulated"])
mc_ener_pro = draw_from_distribution(cr_background_rate(irf.e_bin_centres_fine),
                                     abscis=irf.e_bin_centres_fine,
                                     n_draws=irf.meta_data["proton"]["n_simulated"])
mc_ener_ele = draw_from_distribution(electron_spectrum(irf.e_bin_centres_fine),
                                     abscis=irf.e_bin_centres_fine,
                                     n_draws=irf.meta_data["electron"]["n_simulated"])


# use an energy-dependent energy resolution from the config file
en_res_spline = interpolate.splrep(irf.e_bin_edges.value, table["energy_res_gamma"], k=1)
# gamma reco energy is mc energy smeared by gaussian with std as given in config
reco_ener_gam = mc_ener_gam * np.random.normal(
    1, interpolate.splev(mc_ener_gam, en_res_spline), len(mc_ener_gam))
# electron reco energy is mc energy smeared by gaussian with std as gamma
reco_ener_ele = mc_ener_ele * np.random.normal(
    1, interpolate.splev(mc_ener_ele, en_res_spline), len(mc_ener_ele))
# proton reco energy is flat within the given energy range
reco_ener_pro = np.random.uniform(*irf.e_bin_edges[[0, -1]].value, size=len(mc_ener_pro))


# use an energy-dependent direction resolution from the config file
xi_spline = interpolate.splrep(irf.e_bin_edges.value, table["xi_gamma"], k=1)
del_x_g, del_y_g = np.random.normal(0, interpolate.splev(mc_ener_gam, xi_spline),
                                    (2, len(mc_ener_gam)))
# generate reconstructed proton directions flat within the field of view
del_x_p, del_y_p = np.random.uniform(low=-3, high=3, size=(2, len(mc_ener_pro)))
# generate reconstructed electron directions flat within the field of view
del_x_e, del_y_e = np.random.uniform(low=-3, high=3, size=(2, len(mc_ener_ele)))

# offset angles (i.e. Theta) as the squared sum of the deltas
off_angle_gam = (del_x_g**2 + del_y_g**2)**.5
off_angle_ele = (del_x_e**2 + del_y_e**2)**.5
off_angle_pro = (del_x_p**2 + del_y_p**2)**.5

# electron angular resolution is the same as gammas
xi_x_e, xi_y_e = np.random.normal(0, interpolate.splev(mc_ener_ele, xi_spline),
                                  (2, len(mc_ener_ele)))
# proton angular resolution is five times higher than gammas (just made that up)
xi_x_p, xi_y_p = np.random.normal(0, interpolate.splev(mc_ener_pro, xi_spline) * 5,
                                  (2, len(mc_ener_pro)))
xi_gam = off_angle_gam
xi_ele = (xi_x_e**2 + xi_y_e**2)**.5
xi_pro = (xi_x_p**2 + xi_y_p**2)**.5


# generate gammaness from the distributions in the config file
gamman_absc = np.linspace(0, 1, len(table["gammaness_gamma"]))
gammaness_gam = draw_from_distribution(table["gammaness_gamma"], abscis=gamman_absc,
                                       n_draws=irf.meta_data["gamma"]["n_simulated"])
gammaness_pro = draw_from_distribution(table["gammaness_proton"], abscis=gamman_absc,
                                       n_draws=irf.meta_data["proton"]["n_simulated"])
gammaness_ele = draw_from_distribution(table["gammaness_electron"], abscis=gamman_absc,
                                       n_draws=irf.meta_data["electron"]["n_simulated"])

g = Table([mc_ener_gam, reco_ener_gam, off_angle_gam, xi_gam, gammaness_gam],
          names=["MC_Energy", "reco_Energy", "off_angle", "xi", "gammaness"])
g.write(f"{args.outdir}/{args.outfile}_gamma.h5", path="/reco_events", format="hdf5",
        compression=True)

p = Table([mc_ener_pro, reco_ener_pro, off_angle_pro, xi_pro, gammaness_pro],
          names=["MC_Energy", "reco_Energy", "off_angle", "xi", "gammaness"])
p.write(f"{args.outdir}/{args.outfile}_proton.h5", path="/reco_events", format="hdf5",
        compression=True)

e = Table([mc_ener_ele, reco_ener_ele, off_angle_ele, xi_ele, gammaness_ele],
          names=["MC_Energy", "reco_Energy", "off_angle", "xi", "gammaness"])
e.write(f"{args.outdir}/{args.outfile}_electron.h5", path="/reco_events", format="hdf5",
        compression=True)
