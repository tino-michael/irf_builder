import irf_builder

import pandas as pd
from astropy.table import hstack

e_binned_cuts = [
    "th_cuts",
    "ga_cuts"
]

list_of_irfs_e_binned = [
    "energy_resolution",
    "energy_bias",
    "rel_delta_e_mc",
    "rel_delta_e_reco",
    "energy_fluxes",
    "energy_rates",
    "eff_areas",
    "selec_effs",
    "selec_events",
    "xi",
]

list_of_irfs_e_binned_fine = [
    "energy_matrix"
]

list_of_other_stuff = [
    "th_sq",
    "false_p_rate",
    "true_p_rate"
]


def write_irfs(outfile_path, global_names):

    e_binned_names = []
    e_binned_irfs = []

    irf_frame = pd.DataFrame(irf_builder.e_bin_centres[:, None],
                             columns=["Energy Centres / TeV"])

    try:
        sensitivities = global_names["sensitivity"]
        for mode, sens in sensitivities.items():
            del sens["Energy"]
            for col in sens.colnames:
                sens.rename_column(col, '_'.join([col, mode]))
            try:
                merge = merge.join(sens.to_pandas())
            except NameError:
                merge = sens.to_pandas()
        irf_frame = irf_frame.join(merge)
    except KeyError:
        raise

    for irf_name in list_of_irfs_e_binned:
        try:
            irf = global_names[irf_name]
            for mode in irf:
                for ch in irf[mode]:
                    irf_frame['_'.join([irf_name, ch, mode])] = irf[mode][ch]
        except KeyError:
            print(f"irf {irf_name} not found in global namespace")
            pass

    # add the energy-dependent cut values in Theta and gammaness to the data frame
    for cut_name in e_binned_cuts:
        cut = global_names[cut_name]
        for mode in cut:
            irf_frame['_'.join([cut_name, mode])] = cut[mode]

    print(irf_frame)

    # outfile = pd.HDFStore(outfile_path)
