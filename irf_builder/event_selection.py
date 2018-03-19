import numpy as np
from scipy import optimize

import irf_builder as irf


def cut_and_sensitivity(cuts, events, bin_edges, n_draws=10,
                        syst_nsim=False, syst_nphy=False):
    """ throw this into a minimiser """
    ga_cut = cuts[0]
    th_cut = cuts[1]

    cut_events = {}
    for key in events:
        cut_events[key] = events[key][
            (events[key]["gammaness"] > ga_cut) &
            # the background regions are larger to gather more statistics
            (events[key]["off_angle"] < th_cut * (1 if key == 'g' else irf.r_scale))]

    if syst_nsim and (len(events['g']) < 10 or
                      len(events['g']) < irf.alpha * 0.05 *
                      (len(events['p']) + len(events['e']))):
        return 1

    if syst_nphy and (np.sum(events['g']['weight']) < 10 or
                      np.sum(events['g']['weight']) <
                      (np.sum(events['p']['weight']) +
                       np.sum(events['e']['weight'])) * 0.05 * irf.alpha):
        return 1

    sensitivities = irf.calculate_sensitivity(
        cut_events, bin_edges, alpha=irf.alpha, n_draws=n_draws)

    try:
        return sensitivities["Sensitivity"][0]
    except (KeyError, IndexError):
        return 1


def minimise_sensitivity_per_bin(events, n_draws=10,
                                 syst_nsim=False, syst_nphy=False):

    bin_edges = irf.e_bin_edges

    cut_energies, ga_cuts, th_cuts = [], [], []
    for elow, ehigh, emid in zip(bin_edges[:-1], bin_edges[1:],
                                 np.sqrt(bin_edges[:-1] * bin_edges[1:])):

        cut_events = {}
        for key in events:
            cut_events[key] = events[key][
                (events[key][irf.reco_energy_name] > elow) &
                (events[key][irf.reco_energy_name] < ehigh)]

        res = optimize.differential_evolution(
            cut_and_sensitivity,
            bounds=[(.5, 1), (0, 0.5)],
            maxiter=1000, popsize=10,
            args=(cut_events,
                  np.array([elow / irf.energy_unit,
                            ehigh / irf.energy_unit]) * irf.energy_unit,
                  n_draws, syst_nsim, syst_nphy
                  )
        )

        if res.success:
            cut_energies.append(emid.value)
            ga_cuts.append(res.x[0])
            th_cuts.append(res.x[1])

    return cut_energies, ga_cuts, th_cuts


def apply_cuts(events, cuts=None):
    cuts = cuts or []
    cut_events = {}
    for c, t in events.items():
        mask = np.ones(len(t), dtype=bool)
        for cut in cuts:
            mask = mask & t[cut]
        cut_events[c] = t[mask]
    return cut_events
