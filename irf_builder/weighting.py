import numpy as np
from astropy import units as u

import irf_builder as irf


def unbinned(events, n_simulated_events, e_min_max, target_spectra,
             generator_areas, generator_gamma, observation_time,
             diff_angle=None, extensions=None):
    """
    generates a weight for every event

    Parameters
    ----------
    events : dictionary of tables
        dictionary of the DL3 (?) event tables
    n_simulated_events : dictionary
        total number of simulated events for every channel
    e_min_max : dictionary of tuples
        lower and upper energy limits used in the Monte Carlo event generator
    target_spectra : dictionary of callables
        dictionary of the target fluxes the different channels shall be weighted to
    generator_areas : dictionary
        area within which the shower impact point has been distributed by the
        Monte Carlo event generator
    generator_gamma : dictionary
        index of the MC generated energy spectrum
    observation_time : astropy time quantity
        set the exposure time the weighted events should correspond to
    diff_angle : dictionary of angles, optional
        opening angle of the cone the diffuse flux has been generated in;
    extensions : dictionary of solid angles, optional
        solid angle the diffuse flux has been generated in

    Note
    ----
    • for any channel, `diff_angle` superseeds `extensions` if both are given
    • if neither `diff_angle` nor `extensions` is set for a given channel,
      -> assume point source
    """

    # setting defaults because mutable
    diff_angle = diff_angle or {}
    extensions = extensions or {}
    for ch, angle in diff_angle.items():
        extensions[ch] = 2 * np.pi * (1 - np.cos(angle)) * u.rad**2

    for ch in events:
        try:
            # testing if table columns have units
            events[ch][irf.energy_names["mc"]].unit
            mc_energy = events[ch][irf.energy_names["mc"]]
        except AttributeError:
            # if not, add the default energy unit
            mc_energy = events[ch][irf.energy_names["mc"]].values * irf.energy_unit

        # event weights for a flat energy distribution
        e_w = mc_energy**generator_gamma[ch] \
            * (e_min_max[ch][1]**(1 - generator_gamma[ch]) -
               e_min_max[ch][0]**(1 - generator_gamma[ch])) / \
              (1 - generator_gamma[ch]) \
            * generator_areas[ch] \
            * (extensions[ch] if (ch in extensions) else 1) \
            * observation_time / n_simulated_events[ch]

        # multiply these flat event weights by the flux to get weights corresponding
        # to the number of expected events from that flux
        events[ch]["weight"] = (e_w * target_spectra[ch](mc_energy)).si

    return events


def unbinned_wrapper(events):
    return unbinned(
        events,
        n_simulated_events=dict((ch, irf.meta_data[channel]["n_simulated"])
                                for ch, channel in irf.plotting.channel_map.items()),
        generator_areas=dict((ch,
                              np.pi * (irf.meta_data[channel]["gen_radius"] * u.m)**2)
                             for ch, channel in irf.plotting.channel_map.items()),
        observation_time=irf.observation_time,
        target_spectra={'g': irf.spectra.crab_source_rate,
                        'p': irf.spectra.cr_background_rate,
                        'e': irf.spectra.electron_spectrum},
        e_min_max=dict((ch, (irf.meta_data[channel]["e_min"],
                             irf.meta_data[channel]["e_max"]) * u.TeV)
                       for ch, channel in irf.plotting.channel_map.items()),
        diff_angle=dict((ch, irf.meta_data[channel]["diff_cone"] * u.deg)
                        for ch, channel in irf.plotting.channel_map.items()
                        if irf.meta_data[channel]["diff_cone"] > 0),
        generator_gamma=dict((ch, irf.meta_data[channel]["gen_gamma"])
                             for ch, channel in irf.plotting.channel_map.items()))


def binned(events, effective_areas, selected_events, spectra, observation_time,
           diff_angle={}, extensions={}):

    # setting defaults because mutable
    diff_angle = diff_angle or {}
    extensions = extensions or {}
    for ch, angle in diff_angle.items():
        extensions[ch] = 2 * np.pi * (1 - np.cos(angle)) * u.rad**2

    for ch in events:
        event_rate = irf.spectra.make_mock_event_rate(
            spectra[ch], log_e=False, bin_edges=irf.e_bin_edges)
        if ch in extensions:
            event_rate *= extensions[ch]

        exp_events_per_energy_bin = event_rate * observation_time * effective_areas[ch]
        bin_weights = (exp_events_per_energy_bin / selected_events[ch]).si
        event_weights = bin_weights[
            np.clip(np.digitize(events[ch][irf.energy_names["mc"]],
                                irf.e_bin_edges) - 1,
                    0, len(irf.e_bin_edges) - 2)]

        events[ch]["weight"] = event_weights


def binned_wrapper(events, effective_areas, selected_events):
    return binned(
        events=events, effective_areas=effective_areas, selected_events=selected_events,
        observation_time=irf.observation_time,
        spectra={'g': irf.spectra.crab_source_rate,
                 'p': irf.spectra.cr_background_rate,
                 'e': irf.spectra.electron_spectrum},
        diff_angle=dict((ch, irf.meta_data[channel]["diff_cone"] * u.deg)
                        for ch, channel in irf.plotting.channel_map.items()
                        if irf.meta_data[channel]["diff_cone"] > 0)
    )
