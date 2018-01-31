import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

import irf_builder as irf


def get_simulated_energy_distribution(generator_spectra, n_simulated_events):
    generator_energy_hists = {}
    for cl in generator_spectra:
        generator_energy_hists[cl] = irf.spectra.make_mock_event_rate(
            generator_spectra[cl], norm=n_simulated_events[cl],
            bin_edges=irf.e_bin_edges, log_e=False)
    return generator_energy_hists


def get_simulated_energy_distribution_wrapper(events=None):
    """
    Notes
    -----
    solely depends on meta data -- `events` is not actually used -- only here to unify
    interface for the function calls
    """
    return get_simulated_energy_distribution(
        generator_spectra={'g': irf.spectra.e_minus_2,
                           'p': irf.spectra.e_minus_2,
                           'e': irf.spectra.e_minus_2},
        n_simulated_events={'g': irf.meta_data["gamma"]["n_simulated"],
                            'p': irf.meta_data["proton"]["n_simulated"],
                            'e': irf.meta_data["electron"]["n_simulated"]}
    )


def plot_energy_distribution(events=None, energies=None):
    if (events is None) == (energies is None):
        raise ValueError("please provide one of `events` or `energies`, but not both")

    if energies is None:
        energies = dict((c, e[irf.energy_names["mc"]]) for c, e in events.items())

    for ch, energy in energies.items():
        plt.bar(irf.e_bin_edges[:-1], energy, width=np.diff(irf.e_bin_edges),
                align='edge',
                label=irf.plotting.channel_map[ch],
                color=irf.plotting.channel_colour_map[ch],
                alpha=.5)
    plt.legend()
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel("number of generated events")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.grid()


def get_energy_event_rates(events, ref_energy="reco"):
    energy_rates = {}
    for ch, ev in events.items():
        counts = np.histogram(ev["reco_Energy"],
                              bins=irf.e_bin_edges,
                              weights=ev["weight"])[0]

        energy_rates[ch] = counts / irf.observation_time.to(u.s)
        if ch != 'g':
            energy_rates[ch] *= irf.alpha

    if ref_energy == "reco":
        xlabel = r"$E_\mathrm{reco}$ / TeV"
    else:
        xlabel = r"$E_\mathrm{MC}$ / TeV"

    return energy_rates, xlabel


def get_energy_event_fluxes(events, th_cuts, ref_energy="reco"):
    energy_fluxes = {}
    for ch, ev in events.items():
        counts = np.histogram(ev[irf.energy_names[ref_energy]],
                              bins=irf.e_bin_edges,
                              weights=ev["weight"])[0]

        angle = th_cuts * (1 if ch == 'g' else irf.r_scale) * u.deg
        angular_area = 2 * np.pi * (1 - np.cos(angle)) * u.sr
        energy_fluxes[ch] = counts / (angular_area.to(u.deg**2) *
                                      irf.observation_time.to(u.s))
    if ref_energy == "reco":
        xlabel = r"$E_\mathrm{reco}$ / TeV"
    else:
        xlabel = r"$E_\mathrm{MC}$ / TeV"

    return energy_fluxes, xlabel


def plot_energy_event_fluxes(energy_fluxes, xlabel=None):
    irf.plotting.plot_channels_lines(
        energy_fluxes, xlabel=xlabel,
        ylabel=r"event flux $f / (\mathrm{s}^{-1}*\mathrm{deg}^{-2})$")


def plot_energy_event_rates(energy_rates, xlabel=None):
    irf.plotting.plot_channels_lines(
        energy_rates, xlabel=xlabel,
        ylabel=r"on-region event rates $f / (\mathrm{s}^{-1})$")
