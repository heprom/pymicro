"""A package to work with material microstructures and 3d data sets

.. moduleauthor:: Henry Proudhon <henry.proudhon@mines-paristech.fr>

"""

__version__ = '0.5.3'

import pathlib as pl 
import os 

DATASET_SOURCE = "https://raw.githubusercontent.com/heprom/pymicro-data/main"

def get_cache_dir(data_home=None):
    if data_home is None:
        data_home = os.environ.get("PYMICRO_DATA", pl.Path.home() / ".pymicro_data")
    data_home = pl.Path(data_home)
    if not data_home.exists():
        data_home.mkdir()
    return data_home

def download_data( fname, cache=True):
    data_home = get_cache_dir()
    data_file = data_home / fname
    if not data_file.exists() or not cache:
        data_file.parent.mkdir(parents=True, exist_ok=True)
        url = f"{DATASET_SOURCE}/{fname}"
        print(f"Downloading {url} to {data_file}")
        import requests # pylint: disable=import-outside-toplevel
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Could not download {url}")

        with open(data_file, "wb") as fid:
            fid.write(response.content)
    else:
        print(f"Use cached file {data_file}")
    return data_file
