import irf_builder as irf

import pandas as pd

sensitivities = {}
distributions = {}
e_binned_cuts = {}
energy_matrix = {}
a_other_stuff = {}


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


def add_sens(sens, *args):
    '''Sets the sensitivity variable in `irf.writer`.
    This is here to not confuse users with inconsistent approach compared to other
    containers, e.g. `distributions`, which are nested deeper.
    '''
    irf.writer.sensitivities = sens


def add_dist(val, locals):
    '''Enters `val` into the distributions dictionary in the `irf.writer` namespace'''
    irf.writer.distributions.update(dict([tuplefy(val, locals)]))


def add_cuts(val, locals):
    '''Enters `val` into the e_binned_cuts dictionary in the `irf.writer` namespace'''
    irf.writer.e_binned_cuts.update(dict([tuplefy(val, locals)]))


def add_matr(val, locals):
    '''Enters `val` into the energy_matrix dictionary in the `irf.writer` namespace'''
    irf.writer.energy_matrix.update(dict([tuplefy(val, locals)]))


def add_stuff(val, locals):
    '''Enters `val` into the a_other_stuff dictionary in the `irf.writer` namespace'''
    irf.writer.a_other_stuff.update(dict([tuplefy(val, locals)]))


def write_irfs(
        outfile_path,
        sensitivities=None, distributions=None,
        e_binned_cuts=None, energy_matrix=None,
        a_other_stuff=None):
    '''
    '''
    import irf_builder as irf  # ??? write_irfs forgets about the import at top of file?

    sensitivities = sensitivities or irf.writer.sensitivities
    distributions = distributions or irf.writer.distributions
    e_binned_cuts = e_binned_cuts or irf.writer.e_binned_cuts
    energy_matrix = energy_matrix or irf.writer.energy_matrix
    a_other_stuff = a_other_stuff or irf.writer.a_other_stuff

    # the pandas DataFrame to collect all the distributions
    # initialise already containing the energy bin-centres
    irf_frame = pd.DataFrame(irf.e_bin_centres[:, None],
                             columns=["Energy_Centres"])

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

    outfile = pd.HDFStore(outfile_path)
    outfile.put('distributions', irf_frame, format='table', data_columns=True)

    # put in all the energy-related 2D plots
    for name, dists in energy_matrix.items():
        for mode, dist in dists.items():
            for ch, di in dist.items():
                energy_frame = pd.DataFrame(di)
                outfile.put(f'energy_matrix/{"_".join([name, ch, mode])}',
                            energy_frame, format='table')

    # now brute-force try to add any other distributions that are left
    rercursive_put(a_other_stuff, "other_stuff/", outfile)

    outfile.close()


def rercursive_put(data, name, outfile):
    '''Assumes `data` is a dictionary and tries to iterate over it. Then repeats by
    calling itself with the values of the dictionary.
    In case that fails (due to an `AttributeError`), the deepest dict-level is reached,
    so puts `data` into the HDF5 file given by `outfile`.
    Constructs the output name of the dataframe from the keys of the dictionary during
    the recursive unfolding of the dictionary.
    '''
    try:
        for n, d in data.items():
            new_name = '_'.join([name, n])
            rercursive_put(d, new_name, outfile)
    except AttributeError:
        name = name.replace("/_", "/")
        out_frame = pd.DataFrame(data, columns=[name.split('/')[-1]])
        print(out_frame)
        outfile.put(name, out_frame, format='table', data_columns=True)
