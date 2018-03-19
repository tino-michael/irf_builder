#!/usr/bin/env python

import sys
from os.path import expandvars
import argparse

import numpy as np
from astropy import units as u
from astropy.table import Table

import pandas as pd

from scipy import interpolate

from matplotlib import pyplot as plt

import irf_builder as irf
from irf_builder.plotting import save_fig


# def correct_off_angle(data, origin=None):
#     import ctapipe.utils.linalg as linalg
#     origin = origin or linalg.set_phi_theta(90 * u.deg, 20 * u.deg)
#
#     reco_dirs = linalg.set_phi_theta(data["phi"] * u.deg.to(u.rad),
#                                      data["theta"] * u.deg.to(u.rad)).T
#     off_angles = np.arccos(np.clip(np.dot(reco_dirs, origin), -1., 1.)) * u.rad
#     data["off_angle"] = off_angles.to(u.deg)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--indir',
                    default=expandvars("$CTA_SOFT/tino_cta/data/prod3b/paranal_LND"),
                    help="directory to look up the input files")
parser.add_argument('--infile', type=str, default="classified_events",
                    help="base of the input files' name before mode and channel")
parser.add_argument('--meta_file', type=str, default="meta_data.yml",
                    help="name of the config file that contains information\n"
                         "(energy range, number of processed files etc.)")
parser.add_argument('-k', type=int, default=1, help="order of spline interpolation")
parser.add_argument('--r_scale', type=float, default=5.,
                    help="scale by which to increase radius of the off-region")
parser.add_argument('--n_draws', type=int, default=10,
                    help="number of Poisson draws during the sensitivity calculation "
                         "to estimate the statistical uncertainties")
parser.add_argument('--modes', type=str, nargs='*', default=["wave", "tail"],
                    help="list of data processing modes")
parser.add_argument('--write_irfs', default=False, action='store_true',
                    help="write out produced IRFs in an HDF5 file")

cut_store_group = parser.add_mutually_exclusive_group()
cut_store_group.add_argument('--make_cuts', action='store_true', default=False,
                             help="determines optimal bin-wise gammaness and theta cut "
                             "values and stores them to disk in an astropy table")
cut_store_group.add_argument('--load_cuts', dest='make_cuts', action='store_false',
                             help="loads the gammaness and theta cut values from an "
                             "astropy table from disk")
parser.add_argument('--write_cuts', action='store_true', default=False,
                    help="write cuts as a latex table to disk")

show_plots_group = parser.add_argument_group()
show_plots_group.add_argument('--plot_all', default=False, action='store_true',
                              help="display all plots on screen")
show_plots_group.add_argument('--plot_cuts', default=False, action='store_true',
                              help="display cut-values plots on screen")
show_plots_group.add_argument('--plot_energy', default=False, action='store_true',
                              help="display energy related plots on screen")
show_plots_group.add_argument('--plot_rates', default=False, action='store_true',
                              help="display signal and background rates on screen")
show_plots_group.add_argument('--plot_ang_res', default=False, action='store_true',
                              help="display plots showing angular performance on screen")
show_plots_group.add_argument('--plot_selection', default=False, action='store_true',
                              help="display effective areas, selection efficiencies "
                                   "and number of selected events on screen")
show_plots_group.add_argument('--plot_sensitivity', default=False, action='store_true',
                              help="display sensitivity on screen")
show_plots_group.add_argument('--plot_classification', default=False, action='store_true',
                              help="display plots related to event classification "
                                   "on screen")

write_plots_group = parser.add_argument_group()
write_plots_group.add_argument('--write_plots', default=False, action='store_true',
                               help="write plots to disk in a list of file formats")
write_plots_group.add_argument('--picture_formats', type=str, nargs='*',
                               default=["pdf", "png", "tex"],
                               help="list of file formats to write plots out as")
write_plots_group.add_argument('--plots_outdir', type=str, default='plots',
                               help="directory to write the created plots into")

args = parser.parse_args()

irf.RegionScaler().r_scale = args.r_scale
irf.plotting.file_formats = args.picture_formats

# reading the meta data that describes the MC production
irf.meta_data = irf.load_meta_data(f"{args.indir}/{args.meta_file}")

# reading the reconstructed and classified events
all_events = {}
for mode in args.modes:
    all_events[mode] = {}
    for c, channel in irf.plotting.channel_map.items():
        # all_events[mode][c] = \
        these_events = \
            pd.read_hdf(f"{args.indir}/{args.infile}_{mode}_{channel}.h5")

        # make sure to remove not-reconstructed events (contain nan-values)
        all_events[mode][c] = these_events[these_events["gammaness"] ==
                                           these_events["gammaness"]]

# FUCK FUCK FUCK FUCK
# irf.reco_error_name = "off_angle"
# try:
#     for c in irf.plotting.channel_map:
#         correct_off_angle(all_events["wave"][c])
# except KeyError:
#     pass


# adding a "weight" column to the data tables
for mode, events in all_events.items():
    irf.make_weights(events)
    # for ch in events:
    #     events[ch]["weight_unbinned"] = events[ch]["weight"]
    # effective_areas, _, selected_events = irf.irfs.get_effective_areas(events)
    # irf.weighting.binned_wrapper(events, effective_areas, selected_events)
    # for ch, ev in events.items():
    #     print(ch)
    #     print(events[ch].loc[:, ["MC_Energy", "weight", "weight_unbinned"]])

# # # # # #
# determine optimal bin-by-bin cut values and fit splines to them

cut_energies, ga_cuts, th_cuts = {}, {}, {}
if args.make_cuts:
    print("making cut values")
    for mode, events in all_events.items():
        cut_energies[mode], ga_cuts[mode], th_cuts[mode] = \
            irf.optimise_cuts(events)

    if args.write_cuts:
        Table([cut_energies[mode], ga_cuts[mode], th_cuts[mode]],
              names=["Energy", "gammaness", "theta"]) \
            .write(filename=f"cut_values_{mode}.tex",
                   path=args.indir,
                   format="ascii.latex")
else:
    print("loading cut values")
    for mode in all_events:
        cuts = Table.read(f"{args.indir}/cut_values_{mode}.tex", format="ascii.latex")
        cut_energies[mode] = cuts["Energy"]
        ga_cuts[mode] = cuts["gammaness"]
        th_cuts[mode] = cuts["theta"]
irf.writer.add_cuts(ga_cuts, locals())
irf.writer.add_cuts(th_cuts, locals())


spline_ga, spline_th = {}, {}
for mode in cut_energies:
    spline_ga[mode] = interpolate.splrep(cut_energies[mode], ga_cuts[mode], k=args.k)
    spline_th[mode] = interpolate.splrep(cut_energies[mode], th_cuts[mode], k=args.k)

    if args.plot_cuts or args.plot_all:
        fig = plt.figure(figsize=(10, 5))
        plt.suptitle(mode)
        for i, (cut_var, spline, ylabel) in enumerate(zip(
                [ga_cuts[mode], th_cuts[mode]],
                [spline_ga[mode], spline_th[mode]],
                ["gammaness", r"$\Theta_\mathrm{cut} / ^\circ$"])):
            fig.add_subplot(121 + i)
            plt.plot(cut_energies[mode], cut_var,
                     label="crit. values", ls="", marker="^")
            plt.plot(irf.e_bin_centres_fine / u.TeV,
                     interpolate.splev(irf.e_bin_centres_fine, spline),
                     label="spline fit")

            plt.xlabel(r"$E_\mathrm{MC}$ / TeV")
            plt.ylabel(ylabel)
            plt.gca().set_xscale("log")
            plt.legend()

            if i == 0:
                plt.plot(irf.e_bin_centres_fine[[0, -1]], [1, 1],
                         ls="dashed", color="lightgray")
        plt.subplots_adjust(top=0.91, bottom=0.148,
                            left=0.077, right=0.981,
                            hspace=0.2, wspace=0.204)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/cut_values_{mode}")
        plt.pause(.1)


# evaluating cuts and add columns with flags
for mode, events in all_events.items():
    for ch, ev in events.items():
        ev["pass_gammaness"] = \
            ev["gammaness"] > interpolate.splev(
                ev[irf.energy_names['reco']], spline_ga[mode])
        ev["pass_theta"] = \
            ev["off_angle"] < (1 if ch == 'g' else irf.r_scale) * \
            interpolate.splev(ev[irf.energy_names['reco']], spline_th[mode])


# applying the cuts
cut_events = {}
gamma_events = {}
for mode, events in all_events.items():
    gamma_events[mode] = irf.event_selection.apply_cuts(events, ["pass_gammaness"])
    cut_events[mode] = irf.event_selection.apply_cuts(events,
                                                      ["pass_gammaness", "pass_theta"])


# printing selected number events and summed weights
for step, evs in (("reco", all_events),
                  ("gammaness", gamma_events),
                  ("theta", cut_events)):
    print(f"\nselected MC events at step {step}:")
    for ch, ev in evs["wave"].items():
        print(f"{ch}: {len(ev)}")
    print(f"expected weighted events at step {step}:")
    for ch, ev in evs["wave"].items():
        print(f"{ch}: " + str(np.sum(ev["weight"])))


# measure and correct for the energy bias
energy_resolution, energy_bias = {}, {}
for mode in args.modes:
    energy_resolution[mode], xlabel_e_res, ylabel_e_res = \
        irf.irfs.energy.get_energy_resolution(cut_events[mode])
    energy_bias[mode] = irf.irfs.energy.get_energy_bias(cut_events[mode])
    irf.irfs.energy.correct_energy_bias(cut_events[mode], energy_bias[mode]['g'])
irf.writer.add_dist(energy_resolution, locals())
irf.writer.add_dist(energy_bias, locals())

# energy-integrated significance as a function of observation time
sigmas = {}
obs_test_times = np.logspace(-2, 5, 10) * u.s
for mode, events in cut_events.items():
    sigmas[mode] = irf.sensitivity.sigma_vs_time(events, obs_test_times)
irf.writer.add_stuff(sigmas, locals())

# finally, calculate the sensitivity
sensitivities = {}
for mode, events in cut_events.items():
    sensitivities[mode] = irf.calculate_sensitivity(
        events, irf.e_bin_edges, alpha=irf.alpha, n_draws=args.n_draws)
irf.writer.add_sens(sensitivities)


# ########  ##        #######  ########  ######
# ##     ## ##       ##     ##    ##    ##    ##
# ##     ## ##       ##     ##    ##    ##
# ########  ##       ##     ##    ##     ######
# ##        ##       ##     ##    ##          ##
# ##        ##       ##     ##    ##    ##    ##
# ##        ########  #######     ##     ######
args.modes = ['wave']
if args.plot_energy or args.plot_all:
    energy_matrix, rel_delta_e_reco, rel_delta_e_mc = {}, {}, {}
    irf.writer.add_matr(energy_matrix, locals())
    irf.writer.add_matr(rel_delta_e_reco, locals())
    irf.writer.add_matr(rel_delta_e_mc, locals())
    for mode in args.modes:
        plt.figure(figsize=(10, 5))
        energy_matrix[mode] = \
            irf.irfs.energy.get_energy_migration_matrix(cut_events[mode])
        irf.plotting.plot_energy_migration_matrix(energy_matrix[mode])
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/energy_migration_{mode}")

        plt.figure(figsize=(10, 5))
        rel_delta_e_reco[mode], xlabel, ylabel = \
            irf.irfs.energy.get_rel_delta_e(cut_events[mode])
        irf.plotting.plot_rel_delta_e(rel_delta_e_reco[mode], xlabel, ylabel)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/energy_relative_error_reco_{mode}")

        plt.figure(figsize=(10, 5))
        rel_delta_e_mc[mode], xlabel, ylabel = \
            irf.irfs.energy.get_rel_delta_e(cut_events[mode], ref_energy="mc")
        irf.plotting.plot_rel_delta_e(rel_delta_e_mc[mode], xlabel, ylabel)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/energy_relative_error_mc_{mode}")

        plt.figure()
        irf.plotting.plot_energy_resolution(
            energy_resolution[mode], xlabel_e_res, ylabel_e_res)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/energy_resolution_{mode}")

        plt.figure()
        irf.plotting.plot_energy_bias(energy_bias[mode])
        plt.gca().set_ylim((0, 0.2))  # BUG? matplotlib2tikz does not like neg. axes
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/energy_bias_{mode}")

        plt.figure()
        energy_bias_2 = irf.irfs.energy.get_energy_bias(cut_events[mode])
        irf.plotting.plot_energy_bias(energy_bias_2)
        plt.title("Energy Bias (post e-bias correction)")
        plt.gca().set_ylim((0, 0.2))  # BUG? matplotlib2tikz does not like neg. axes
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/energy_bias_corrected_{mode}")

        plt.figure()
        energy_resolution2 = irf.irfs.energy.get_energy_resolution(cut_events[mode])
        irf.plotting.plot_energy_resolution(*energy_resolution2)
        plt.title("Energy Resolution (post e-bias correction)")
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/energy_resolution_corrected_{mode}")


if args.plot_rates or args.plot_all:
    energy_fluxes, energy_rates = {}, {}
    irf.writer.add_dist(energy_fluxes, locals())
    irf.writer.add_dist(energy_rates, locals())
    for mode in args.modes:
        plt.figure()
        energy_fluxes[mode], xlabel = irf.irfs.event_rates.get_energy_event_fluxes(
            cut_events[mode], th_cuts[mode])
        irf.plotting.plot_energy_event_fluxes(energy_fluxes[mode], xlabel=xlabel)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/fluxes_{mode}")

        plt.figure()
        energy_rates[mode], xlabel = \
            irf.irfs.event_rates.get_energy_event_rates(cut_events[mode])
        irf.plotting.plot_energy_event_rates(energy_rates[mode], xlabel=xlabel)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/event_rates_{mode}")


if args.plot_selection or args.plot_all:
    eff_areas, selec_effs, selec_events = {}, {}, {}
    irf.writer.add_dist(eff_areas, locals())
    irf.writer.add_dist(selec_effs, locals())
    irf.writer.add_dist(selec_events, locals())
    generator_energies = {}
    for mode in args.modes:
        eff_areas[mode], selec_effs[mode], selec_events[mode] = \
            irf.irfs.get_effective_areas(cut_events[mode])
        plt.figure()
        irf.plotting.plot_effective_areas(eff_areas[mode])
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/effective_areas_{mode}")
        plt.figure()
        irf.plotting.plot_selection_efficiencies(selec_effs[mode])
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/selection_efficiencies_{mode}")

        plt.figure()
        generator_energies[mode] = \
            irf.irfs.get_simulated_energy_distribution(cut_events[mode])
        irf.plotting.plot_energy_distribution(energies=generator_energies[mode])

        plt.plot(irf.e_bin_centres, selec_events[mode]['p'], label="sel. protons")
        plt.plot(irf.e_bin_centres, selec_events[mode]['g'], label="sel. gammas")
        plt.legend()
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")


if args.plot_ang_res or args.plot_all:
    theta, xi = {}, {}
    for mode in args.modes:
        plt.figure()
        theta[mode], theta_bins = \
            irf.irfs.angular_resolution.get_theta_square(cut_events[mode])
        irf.plotting.plot_theta_square(theta[mode], theta_bins)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/theta_square_{mode}")

        plt.figure()
        xi[mode], xlabel = \
            irf.irfs.angular_resolution.get_angular_resolution(gamma_events[mode])
        irf.plotting.plot_angular_resolution(xi[mode], xlabel)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/angular_resolution_{mode}")

        plt.figure()
        irf.plotting.plot_angular_resolution_violin(gamma_events[mode])
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/angular_violins_{mode}")
    irf.writer.add_dist(xi, locals())
    irf.writer.add_stuff(theta, locals())
    irf.writer.add_stuff(theta_bins, locals())


if args.plot_sensitivity or args.plot_all:
    plt.figure()
    irf.plotting.plot_crab()
    irf.plotting.plot_reference()
    irf.plotting.plot_sensitivity(sensitivities)
    if args.write_plots:
        save_fig(f"{args.plots_outdir}/sensitivity")

    plt.figure()
    irf.plotting.plot_significance_vs_time(sigmas, obs_test_times)
    if args.write_plots:
        save_fig(f"{args.plots_outdir}/significance_vs_time")


if args.plot_classification or args.plot_all:
    false_p_rate, true_p_rate = {}, {}
    irf.writer.add_stuff(false_p_rate, locals())
    irf.writer.add_stuff(true_p_rate, locals())
    for mode in args.modes:
        plt.figure()
        false_p_rate[mode], true_p_rate[mode], roc_area_under_curve = \
            irf.irfs.classification.get_roc_curve(all_events[mode])
        irf.plotting.plot_roc_curve(false_p_rate[mode], true_p_rate[mode],
                                    roc_area_under_curve)
        if args.write_plots:
            save_fig(f"{args.plots_outdir}/ROC_curve_{mode}")


# if desired, put all constructed distributions into an h5 file
if args.write_irfs:
    irf.writer.write_irfs(f"foo.h5")

plt.show()
