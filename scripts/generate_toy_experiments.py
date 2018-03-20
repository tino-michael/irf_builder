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
from irf_builder.pseudo_simulation import draw_from_distribution, temp_get_gammaness


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


# TODO simulate
# - xi
# - off_angle
# - MC Energy ✔ -> reco Energy ✔
# - gammaness


# path, filename = split(args.config)
# gam_g, gam_p, gam_e = temp_get_gammaness()
# Table([gam_g, gam_p, gam_e],
#       names=["gammaness_gamma", "gammaness_proton",
#              "gammaness_electron"]).write(filename=filename, path=path,
#                                           format="ascii.latex")

table = Table.read(args.config, format="ascii.latex")

mc_ener_gam = draw_from_distribution(crab_source_rate(irf.e_bin_centres_fine),
                                     abscis=irf.e_bin_centres_fine,
                                     n_draws=irf.meta_data["gamma"]["n_simulated"])
mc_ener_pro = draw_from_distribution(cr_background_rate(irf.e_bin_centres_fine),
                                     abscis=irf.e_bin_centres_fine,
                                     n_draws=irf.meta_data["proton"]["n_simulated"])
mc_ener_ele = draw_from_distribution(electron_spectrum(irf.e_bin_centres_fine),
                                     abscis=irf.e_bin_centres_fine,
                                     n_draws=irf.meta_data["electron"]["n_simulated"])

en_res_spline = interpolate.splrep(irf.e_bin_edges.value, table["energy_res_gamma"], k=1)

# gamma reco energy is mc energy smeared by gaussian with std as given in config
reco_ener_gam = mc_ener_gam * np.random.normal(
    1, interpolate.splev(mc_ener_gam, en_res_spline), len(mc_ener_gam))
# electron reco energy is mc energy smeared by gaussian with std as gamma
reco_ener_ele = mc_ener_gam * np.random.normal(
    1, interpolate.splev(mc_ener_gam, en_res_spline), len(mc_ener_gam))
# proton reco energy is flat within the given energy range
reco_ener_pro = np.random.uniform(irf.e_bin_edges[[0, -1]], size=len(mc_ener_pro))
