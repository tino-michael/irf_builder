import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

import irf_builder as irf
from irf_builder.spectra import make_mock_event_rate, e_minus_2


def get_effective_areas(events, generator_areas,
                        n_simulated_events=None,
                        generator_spectra=None,
                        generator_energy_hists=None
                        ):
    """
    calculates the effective areas for the provided channels

    Parameters
    ----------
    generator_areas : dictionary of astropy quantities
        the area for each channel within which the shower impact position was
        generated
    n_simulated_events : dictionary of integers, optional (defaults: None)
        number of events used in the MC simulation for each channel
    generator_spectra : dictionary of functors, optional (default: None)
        function object for the differential generator flux of each channel
    generator_energy_hists : dictionary of numpy arrays, optional (default: None)
        energy histogram of the generated events for each channel binned according to
        `.energy_bin_edges`

    Returns
    -------
    effective_areas : dictionary of numpy arrays
        histograms of the effective areas of the different channels binned according to
        `irf.e_bin_edges`
    selection_efficiencies : dictionary of numpy arrays

    selected_events : dictionary of numpy arrays


    Notes
    -----
    either give the histogram of the energy distributions at MC generator level with
    `generator_energy_hists` or create them on the fly with `n_simulated_events` and
    `generator_spectra`
    """

    if (n_simulated_events is not None and generator_spectra is not None) == \
            (generator_energy_hists is not None):
        raise ValueError("use either (n_simulated_events and generator"
                         "_spectra) or generator_energy_hists to set the MC "
                         "generated energy spectrum -- not both")

    if generator_energy_hists is None:
        generator_energy_hists = \
            irf.irfs.event_rates.get_simulated_energy_distribution_wrapper(events)

    # an energy-binned histogram of the effective areas
    # binning according to irf.e_bin_edges
    effective_areas = {}

    # an energy-binned histogram of the selected events
    # binning according to irf.e_bin_edges
    selected_events = {}

    # an energy-binned histogram of the selection efficiencies
    # binning according to irf.e_bin_edges
    selection_efficiencies = {}

    # generate the histograms for the energy distributions of the selected events
    for ch in events:
        mc_energy = events[ch][irf.energy_names["mc"]]
        selected_events[ch] = np.histogram(mc_energy, bins=irf.e_bin_edges)[0]

        # the effective areas are the selection efficiencies per energy bin multiplied
        # by the area in which the Monte Carlo events have been generated in
        selection_efficiencies[ch] = selected_events[ch] / generator_energy_hists[ch]
        effective_areas[ch] = selection_efficiencies[ch] * generator_areas[ch]

    return effective_areas, selection_efficiencies, selected_events


def get_effective_areas_wrapper(events):
    return get_effective_areas(
        events,
        generator_areas=dict((ch,
                              np.pi * (irf.meta_data[channel]["gen_radius"] * u.m)**2)
                             for ch, channel in irf.plotting.channel_map.items()),
        n_simulated_events=dict((ch, irf.meta_data[channel]["n_simulated"])
                                for ch, channel in irf.plotting.channel_map.items()),
        generator_spectra={'g': e_minus_2,
                           'p': e_minus_2,
                           'e': e_minus_2})


def plot_effective_areas(eff_areas):
    """plots the effective areas of the different channels as a line plot

    Parameter
    ---------
    eff_areas : dict of 1D arrays
        dictionary of the effective areas of the different channels
    """
    irf.plotting.plot_channels_lines(eff_areas, title="Effective Areas",
                                     ylabel=r"$A_\mathrm{eff} / \mathrm{m}^2$")


def plot_selection_efficiencies(selec_effs):
    """plots the selection efficiencies of the different channels as a line plot

    Parameter
    ---------
    eff_areas : dict of 1D arrays
        dictionary of the selection efficiencies of the different channels
    """
    irf.plotting.plot_channels_lines(selec_effs, title="selection Efficiencies",
                                     ylabel="fraction of selected events")
