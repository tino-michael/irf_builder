#!/usr/bin/env python

import sys
from os.path import expandvars
import argparse

import numpy as np
from astropy import units as u
from astropy.table import Table

import pandas as pd

import irf_builder as irf

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


# reading the meta data that describes the MC production
irf.meta_data = irf.load_meta_data(f"{args.indir}/{args.meta_file}")


# TODO simulate
# - xi
# - off_angle
# - MC Energy -> reco Energy
# - gammaness
