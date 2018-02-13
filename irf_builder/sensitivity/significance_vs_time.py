import numpy as np
import matplotlib.pyplot as plt

import irf_builder as irf
from irf_builder import spectra

from .li_ma import sigma_lima


def sum_events(ev, mask):
    return np.sum(ev["weight"][mask]) if len(ev["weight"][mask]) > 1 else 0


def sigma_vs_time(events, times, energy_range=None, alpha=None, signal_list=("g")):
    """
    """
    energy_range = energy_range or irf.e_bin_edges
    alpha = alpha or irf.alpha

    S_events = np.zeros(2)  # [on-signal, off-background]

    # count the (weights of the) events in the on and off regions
    # for the whole energy range
    for ch, ev in events.items():
        # only pick events with reconstructed energy in the given energy range
        e_mask = (ev[irf.energy_names["reco"]] > energy_range[0]) & \
                 (ev[irf.energy_names["reco"]] < energy_range[-1])

        S_events[0 if ch in signal_list else 1] += sum_events(ev, e_mask)

    return sigma_lima(*(S_events[:, None] * times / irf.observation_time).si, alpha)


def plot_significance_vs_time(sigmas, times):
    for mode, sigma in sigmas.items():
        plt.loglog(
            times, sigma, marker="s",
            color=irf.plotting.mode_colour_map[mode],
            label=irf.plotting.mode_map[mode])

    for sig in [3, 5]:
        plt.plot(times[[0, -1]], [sig, sig], label=f"{sig} sigma",
                 color="gray", alpha=1 - sig / 10, ls="--")

    plt.legend()
    plt.xlabel(rf'$t_\mathrm{{obs}}$ / {times.unit:latex}')
    plt.ylabel(r'significance / $\sigma$')
    plt.grid()
    plt.xlim(times[[0, -1]].value)
    plt.tight_layout()
