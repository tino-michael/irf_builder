import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def load_meta_data_from_yml(meta_data_file):
    # load meta data from disk
    meta_data = yaml.load(open(meta_data_file), Loader=Loader)

    # add a field for number of simulated events
    for meta in [meta_data["gamma"], meta_data["proton"], meta_data["electron"]]:
        meta["n_simulated"] = meta["n_files"] * meta["n_events_per_file"]

    return meta_data
