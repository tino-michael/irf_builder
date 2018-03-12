# irf_builder

This is a stand-alone package to compute the instrument response functions (IRFs) for the
Cherenkov Telescope Array (CTA) gamma observatory.

The package provides methods to generate event weights, do the event selection
(based on minimising the sensitivity) and produces all kinds of distributions including
different manifestations of a sensitivity.

## TODO
- [ ] doc-strings in many functions
    - [ ] \_\_init__.py
    - [ ] plotting.py
    - [ ] event_selection.py
    - [ ] spectra.py
    - [ ] sensitivity/li_ma.py
    - [ ] sensitivity/\_\_init__.py
    - [ ] sensitivity/discovery_flux_vs_time.py
    - [ ] sensitivity/differential_energy_point_source_sensitivity.py
    - [ ] sensitivity/significance_vs_time.py
    - [ ] weighting.py
    - [ ] meta_data_loader.py
    - [ ] writer.py
    - [ ] irfs/\_\_init__.py
    - [ ] irfs/angular_resolution.py
    - [ ] irfs/event_rates.py
    - [ ] irfs/effective_areas.py
    - [x] irfs/energy.py
    - [ ] irfs/classification.py
- [ ] write out event tables including the selection flags
- [x] write out the produced IRFs in _any_ format
- [ ] write out the produced IRFs in an agreed upon format
- [ ] add more plots concerning event discrimination
- [x] write out plots
- [ ] add ability to plot requirement curves
- [ ] unit test everything
