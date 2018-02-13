import irf_builder as irf

import pandas as pd

distributions = {}
e_binned_cuts = {}
energy_matrix = {}

a_other_stuff = [
    "th_sq",
    "false_p_rate",
    "true_p_rate"
]


def stringify(val, locals):
    '''Turns `val` into a string by looking up the local namespace -- i.e. `locals()` --
    and returning the first variable name that `is` identical to `val` itself.

    Parameters
    ----------
    val : python variable
        any variable name
    locals : dict
        local namespace dictionary created by `locals()`

    Returns
    -------
    name : string
        name of `val` as a string in the namespace of the function call

    '''
    for n, v in locals.items():
        if val is v:
            return n


def tuplefy(val, locals):
    '''Returns a tuple of `val` and its stringified version

    Parameters
    ----------
    val : python variable
        any variable name
    locals : dict
        local namespace dictionary created by `locals()`

    Returns
    -------
    name, val
        tuple of `val`'s name and `val` itself

    '''
    return (stringify(val, locals), val)


def add_dist(val, locals):
    '''Enters `val` into the distributions dictionary in the `irf.writer` namespace'''
    irf.writer.distributions.update(dict([tuplefy(val, locals)]))


def add_cuts(val, locals):
    '''Enters `val` into the e_binned_cuts dictionary in the `irf.writer` namespace'''
    irf.writer.e_binned_cuts.update(dict([tuplefy(val, locals)]))


def add_matr(val, locals):
    '''Enters `val` into the energy_matrix dictionary in the `irf.writer` namespace'''
    irf.writer.energy_matrix.update(dict([tuplefy(val, locals)]))


def write_irfs(
        outfile_path,
        sensitivities=None, distributions=None, e_binned_cuts=None, energy_matrix=None):
    '''
    '''
    import irf_builder as irf  # ??? write_irfs forgets about the import at top of file?

    sensitivities = sensitivities or {}
    distributions = distributions or irf.writer.distributions
    e_binned_cuts = e_binned_cuts or irf.writer.e_binned_cuts
    energy_matrix = energy_matrix or irf.writer.energy_matrix

    # the pandas DataFrame to collect all the distributions
    # initialise already containing the energy bin-centres
    irf_frame = pd.DataFrame(irf.e_bin_centres[:, None],
                             columns=["Energy Centres / TeV"])

    # the energy-binned sensitivity
    # three distribution per `mode`:
    # the sensitivity itself and the upper and lower limits
    for mode, sens in sensitivities.items():
        del sens["Energy"]
        for col in sens.colnames:
            sens.rename_column(col, '_'.join([col, mode]))

        irf_frame = irf_frame.join(sens.to_pandas())

    # other distributions that have the same energy binning
    # here, every channel possibly has its own distribution,
    # so the dict nesting is one level deeper
    for irf_name, irf in distributions.items():
        for mode in irf:
            for ch in irf[mode]:
                irf_frame['_'.join([irf_name, ch, mode])] = irf[mode][ch]

    # add the energy-dependent cut values in Theta and gammaness to the data frame
    for cut_name, cut in e_binned_cuts.items():
        for mode in cut:
            irf_frame['_'.join([cut_name, mode])] = cut[mode]

    # print(irf_frame)
    # outfile = pd.HDFStore(outfile_path)
