import numpy as np
from matplotlib import pyplot as plt

import irf_builder as irf


def get_energy_migration_matrix(events):
    """
    Return
    ------
    energy_matrix : 2D array
        the energy migration matrix in the form of:
        `energy_matrix[mc_energy_bin][reco_energy_bin]`
    """
    energy_matrix = {}
    for i, (ch, ev) in enumerate(events.items()):
        counts, _, _ = np.histogram2d(ev[irf.energy_names["mc"]],
                                      ev[irf.energy_names["reco"]],
                                      bins=(irf.e_bin_edges_fine,
                                            irf.e_bin_edges_fine))
        energy_matrix[ch] = counts
    return energy_matrix


def plot_energy_migration_matrix(energy_matrix, fig=None):
    if fig is None:
        fig = plt.gcf()
    for i, (ch, e_matrix) in enumerate(energy_matrix.items()):
        ax = fig.add_subplot(131 + i)

        ax.pcolormesh(irf.e_bin_edges_fine.value,
                      irf.e_bin_edges_fine.value, e_matrix)
        plt.plot(irf.e_bin_edges_fine.value[[0, -1]],
                 irf.e_bin_edges_fine.value[[0, -1]],
                 color="darkgreen")
        plt.title(irf.plotting.channel_map[ch])
        ax.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
        if i == 0:
            ax.set_ylabel(r"$E_\mathrm{MC}$ / TeV")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.grid()

    plt.tight_layout()


def get_rel_delta_e(events, ref_energy="reco"):
    ref_energy_name = irf.energy_names[ref_energy]
    rel_delta_e = {}
    for ch in events:
        counts, _, _ = np.histogram2d(
            events[ch][ref_energy_name],
            (events[ch][irf.energy_names["reco"]] - events[ch][irf.energy_names["mc"]]) /
            events[ch][ref_energy_name],
            bins=(irf.e_bin_edges_fine, np.linspace(-1, 1, 100)))
        rel_delta_e[ch] = counts

    if ref_energy == "reco":
        xlabel = r"$E_\mathrm{reco}$ / TeV"
        ylabel = r"$(E_\mathrm{reco} - E_\mathrm{MC}) / E_\mathrm{reco}$"
    else:
        xlabel = r"$E_\mathrm{MC}$ / TeV"
        ylabel = r"$(E_\mathrm{reco} - E_\mathrm{MC}) / E_\mathrm{MC}$"

    return rel_delta_e, xlabel, ylabel


def plot_rel_delta_e(rel_delta_e, xlabel=None, ylabel=None, fig=None):
    if fig is None:
        fig = plt.gcf()
    for i, ch in enumerate(rel_delta_e):
        ax = fig.add_subplot(131 + i)
        ax.pcolormesh(irf.e_bin_edges_fine / irf.energy_unit,
                      np.linspace(-1, 1, 100),
                      rel_delta_e[ch].T)
        plt.plot(irf.e_bin_edges_fine.value[[0, -1]], [0, 0],
                 color="darkgreen")
        plt.title(irf.plotting.channel_map[ch])
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        plt.grid()

    plt.tight_layout()


def get_energy_bias(events):
    energy_bias = {}
    for ch, e in events.items():
        median_bias = np.zeros_like(irf.e_bin_centres.value)
        for i, (e_low, e_high) in enumerate(zip(irf.e_bin_edges[:-1] / irf.energy_unit,
                                                irf.e_bin_edges[1:] / irf.energy_unit)):
            bias = (e[irf.energy_names["mc"]] / e[irf.energy_names["reco"]]) - 1

            try:
                median_bias[i] = np.percentile(
                    bias[(e[irf.energy_names["reco"]] > e_low) &
                         (e[irf.energy_names["reco"]] < e_high)],
                    50)
            except IndexError:
                pass
        energy_bias[ch] = median_bias

    return energy_bias


def plot_energy_bias(energy_bias, channels=None):

    channels = channels or ['g']
    irf.plotting.plot_channels_lines(
        data=dict((ch, energy_bias[ch]) for ch in channels),
        ylabel=r"$E_\mathrm{MC}/E_\mathrm{reco} - 1$",
        title="Energy Bias"
    )
    plt.gca().set_yscale("linear")

    return


def correct_energy_bias(events, energy_bias, k=1):
    from scipy import interpolate

    spline = interpolate.splrep(irf.e_bin_centres.value, energy_bias, k=k)

    for ch in events:
        events[ch][irf.energy_names["reco"]] *= \
            (1 + interpolate.splev(events[ch][irf.energy_names["reco"]], spline))
    return events


def get_energy_resolution(events, ref_energy="reco", percentile=68):
    ref_energy_name = irf.energy_names[ref_energy]
    energy_resolution = {}
    for ch, e in events.items():
        resolution = np.zeros_like(irf.e_bin_centres.value)
        for i, (e_low, e_high) in enumerate(zip(irf.e_bin_edges[:-1] / irf.energy_unit,
                                                irf.e_bin_edges[1:] / irf.energy_unit)):
            rel_error = np.abs(e[irf.energy_names["mc"]] -
                               e[irf.energy_names["reco"]]) / e[ref_energy_name]

            try:
                resolution[i] = np.percentile(
                    rel_error[(e[ref_energy_name] > e_low) &
                              (e[ref_energy_name] < e_high)],
                    percentile)
            except IndexError:
                pass
        energy_resolution[ch] = resolution

    return energy_resolution


def plot_energy_resolution(energy_resolution, channels=None):

    channels = channels or ['g']
    irf.plotting.plot_channels_lines(
        data=dict((ch, energy_resolution[ch]) for ch in channels),
        ylabel=r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{reco}$",
        title="Energy Resolution"
    )
    plt.gca().set_yscale("linear")

    return
