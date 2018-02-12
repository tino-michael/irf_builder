import numpy as np
from matplotlib import pyplot as plt

import irf_builder as irf


def get_theta_square(events, bin_edges=None):
    bin_edges = bin_edges or np.linspace(0, .1, 50)
    theta_square = {}
    for channel in events:
        theta_square[channel] = np.histogram(
            events[channel][irf.offset_angle_name]**2,
            weights=events[channel]["weight"],
            bins=bin_edges)[0]
    return theta_square, bin_edges


def plot_theta_square(theta_square, bin_edges):
    for channel in ['g', 'e', 'p']:
        plt.bar(bin_edges[:-1], theta_square[channel], width=np.diff(bin_edges),
                align='edge', color=irf.plotting.channel_colour_map[channel],
                label=irf.plotting.channel_map[channel], alpha=.3)
    plt.xlabel(r"$\Theta^2 / {}^{\circ^2}$")
    plt.ylabel("event counts")
    plt.gca().set_xlim(bin_edges[[0, -1]])

    import matplotlib.ticker
    mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    plt.gca().yaxis.set_major_formatter(mf)

    plt.subplots_adjust(top=0.935, bottom=0.151,
                        left=0.137, right=0.948,
                        hspace=0.2, wspace=0.4)
    plt.legend()


def percentiles(values, bin_values, bin_edges, percentile):
    percentiles_binned = \
        np.squeeze(np.full((len(bin_edges) - 1, len(values.shape)), np.inf))
    for i, (bin_l, bin_h) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        try:
            percentiles_binned[i] = \
                np.percentile(values[(bin_values > bin_l) &
                                     (bin_values < bin_h)], percentile)
        except IndexError:
            pass
    return percentiles_binned.T


def get_angular_resolution(events, percent=68, ref_energy="reco"):
    xi_xx = {}
    for channel, values in events.items():
            xi_xx[channel] = percentiles(values[irf.reco_error_name],
                                         values[irf.energy_names[ref_energy]],
                                         irf.e_bin_edges.value, percent)
    if ref_energy == "reco":
        xlabel = r"$E_\mathrm{reco}$ / TeV"
    else:
        xlabel = r"$E_\mathrm{MC}$ / TeV"

    return xi_xx, xlabel


def plot_angular_resolution(xi, xlabel=None):
    # for cl, a in xi.items():
    for cl, a in [('g', xi['g'])]:
        plt.plot(irf.e_bin_centres, a,
                 label=irf.plotting.channel_map[cl],
                 color=irf.plotting.channel_colour_map[cl],
                 marker=irf.plotting.channel_marker_map[cl])
    plt.legend()
    plt.title("angular resolution")
    plt.xlabel(xlabel)
    plt.ylabel(r"$\xi_{68} / ^\circ$")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.gca().set_ylim((1e-2, 1))
    plt.subplots_adjust(top=0.918, bottom=0.154,
                        left=0.147, right=0.971,
                        hspace=0.2, wspace=0.4)
    plt.grid()


def plot_angular_resolution_violin(events):

    energy = events['g'][irf.energy_names["mc"]]
    off_angle = events['g'][irf.offset_angle_name]

    # to plot the violins, sort the ordinate values into a dictionary
    # the keys are the central values of the bins given by `bin_edges`
    val_vs_dep = {}

    # get the energy-bin-number of every event
    # outliers are put into the first and last bin accordingly
    ibins = np.clip(np.digitize(energy, irf.e_bin_edges) - 1,
                    0, len(irf.e_bin_centres) - 1)

    for e_low, e_high in zip(irf.e_bin_edges[:-1],
                             irf.e_bin_edges[1:]):
        ibin = ibins[(energy > e_low) & (energy < e_high)][0]
        val_vs_dep[irf.e_bin_centres[ibin].value] = \
            np.log10(off_angle[(energy > e_low) & (energy < e_high)])

    keys = [k[0] for k in sorted(val_vs_dep.items())]
    vals = [k[1] for k in sorted(val_vs_dep.items())]

    # calculate the widths of the violins as 90 % of the corresponding bin width
    widths = []
    for cen, wid in zip(irf.e_bin_centres, np.diff(irf.e_bin_edges)):
        if cen.value in keys:
            widths.append(wid.value * .9)

    plt.violinplot(vals, keys, widths=widths,
                   points=60, showextrema=False, showmedians=True)

    plt.gca().set_xscale('log')
    plt.ylim((-3, 2))
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$\log_{10}(\xi / ^\circ$)")
    plt.subplots_adjust(top=0.956, bottom=0.154,
                        left=0.123, right=0.971,
                        hspace=0.2, wspace=0.4)
    plt.grid()
