import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker


import irf_builder as irf


# matplotlib2tikz can conveniently store matplotlib plots as pgf/tikzpictures
# highly recommended
try:
    from matplotlib2tikz import save as tikzsave

    def tikz_save(arg, **kwargs):
        print(arg)
        tikzsave(arg, **kwargs,
                 figureheight='\\figureheight',
                 figurewidth='\\figurewidth')
except ImportError:
    print("matplotlib2tikz is not installed")
    print("install with: \n$ pip install matplotlib2tikz")

    def tikz_save(arg, **kwargs):
        print("matplotlib2tikz is not installed")
        print("no .tex is saved")


# pull in all the plot functions into one `plotting` namespace
from irf_builder.sensitivity.differential_energy_point_source_sensitivity import \
    plot_sensitivity
from irf_builder.sensitivity.significance_vs_time import plot_significance_vs_time
from irf_builder.irfs.classification import plot_roc_curve
from irf_builder.irfs.effective_areas import (plot_effective_areas,
                                              plot_selection_efficiencies)
from irf_builder.irfs.energy import (plot_energy_migration_matrix, plot_rel_delta_e,
                                     plot_energy_bias, plot_energy_resolution)
from irf_builder.irfs.event_rates import (plot_energy_distribution,
                                          plot_energy_event_rates,
                                          plot_energy_event_fluxes)
from irf_builder.irfs.angular_resolution import (plot_theta_square,
                                                 plot_angular_resolution,
                                                 plot_angular_resolution_violin)


# some dictionaries to control the visuals of the different plots in the same figure
channel_map = {'g': "gamma", 'p': "proton", 'e': "electron"}
channel_colour_map = {'g': "orange", 'p': "blue", 'e': "red"}
channel_marker_map = {'g': 's', 'p': '^', 'e': 'v'}

step_linestyle_map = {'theta': '-', 'gammaness': '--',
                      'reco': '-.', 'trigger': ':'}

mode_map = {"wave": "wavelets", "tail": "tailcuts"}
mode_colour_map = {"tail": "darkorange", "wave": "darkred"}


# list of file endings to store all plots as
file_formats = ["pdf"]


def save_fig(outname, endings=None, **kwargs):
    """saves the current matplotlib figure as a given list of file formats

    Parameters
    ----------
    outname : string
        destination path without ending for the figure being saved
    endings : list of strings, optional (default: ["tex", "pdf", "png"])
        list of file name endings to save the figure as; can be "tex" or
        any other ending supported and recognised by matplotlib

    Note
    ----
    uses the package `matplotlib2tikz` to save figures as tikz/pgf plots;
    if not installed, only prints an info message
    """
    try:
        for end in endings or irf.plotting.file_formats:
            if end == "tex":
                tikz_save(f"{outname}.{end}", **kwargs)
            else:
                plt.savefig(f"{outname}.{end}")
    except FileNotFoundError:
        import os
        pathname, filename = os.path.split(outname)
        print(f"path '{pathname}' does not seem to exist -- {filename} not written")


def plot_channels_lines(data, ylabel, bin_centres=None, title=None,
                        xlabel=r"$E_\mathrm{MC}$ / TeV"):
    """generic plotting function that plots all data in a dictionary as a line plot.
    Picks colours and labels assuming gamma/proton/electron channels.

    Parameters
    ----------
    data : dict of 1D arrays
        dictionary of the data to be plotted
    bin_centres : 1D array, optional (default: `irf.e_bin_centres`)
        "x-values" of the data that is being plotted
    title, ylabel, xlabel : strings (defaults: xlabel=r"$E_\mathrm{MC}$ / TeV")
        text to be put around the plot

    """
    bin_centres = bin_centres or irf.e_bin_centres
    for cl, dat in data.items():
        plt.plot(bin_centres, dat,
                 label=irf.plotting.channel_map[cl],
                 color=irf.plotting.channel_colour_map[cl],
                 marker=irf.plotting.channel_marker_map[cl])
    plt.legend()
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.grid()
    plt.tight_layout()


def plot_crab(e_bins=None, fractions=None):
    e_bins = e_bins or irf.e_bin_centres
    fractions = fractions or [1, 10, 100]
    for i, frac in enumerate(fractions):
        plt.loglog(e_bins, (irf.spectra.crab_source_rate(e_bins) / frac * e_bins**2)
                   .to(irf.sensitivity_unit).value,
                   color="red", ls="dashed", alpha=1 - (i / len(fractions)),
                   label="Crab Nebula" + (f" / {frac}" if frac != 1 else ""))


def plot_reference():
    """some pseude-official sensitivity line to compare to
    """
    ref_loge, ref_sens = *(np.array([
        (-1.8, 6.87978e-11), (-1.6, 1.87765e-11),
        (-1.4, 7.00645e-12), (-1.2, 1.77677e-12), (-1.0, 8.19263e-13),
        (-0.8, 4.84879e-13), (-0.6, 3.00256e-13), (-0.4, 2.07787e-13),
        (-0.2, 1.4176e-13), (0.0, 1.06069e-13), (0.2, 8.58209e-14),
        (0.4, 6.94294e-14), (0.6, 6.69301e-14), (0.8, 7.61169e-14),
        (1.0, 7.13895e-14), (1.2, 9.49376e-14), (1.4, 1.25208e-13),
        (1.6, 1.91209e-13), (1.8, 3.11611e-13), (2.0, 4.80354e-13)]).T),
    plt.loglog(10**ref_loge,
               ((ref_sens) * (u.erg * u.cm**2 * u.s)**(-1)).to(irf.flux_unit).value,
               marker="s", color="black", ms=3, linewidth=1,
               label="reference")


class SquaredScale(mscale.ScaleBase):
    """
    ScaleBase class for generating squared scale in matplotlib axes.
    shamelessly copied from
    https://stackoverflow.com/questions/42277989/square-root-scale-using-matplotlib-python?answertab=votes#tab-top
    """
    name = 'squared'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax

    # def get_transform(self):
    #     return self.InvertedSquareRootTransform()
    class SquareTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquaredScale.InvertedSquareTransform()

    class InvertedSquareTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**.5

        def inverted(self):
            return SquaredScale.SquareTransform()

    def get_transform(self):
        return self.SquareTransform()


mscale.register_scale(SquaredScale)
