"""A package to work with material microstructures and 3d data sets

.. moduleauthor:: Henry Proudhon <henry.proudhon@mines-paristech.fr>

"""

__version__ = '0.6.1'

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

def dowload_datadir():
    """Download the data directory

    Returns:
        pl.Path: the path to the data directory
    """
    data_home = get_cache_dir()
    fpath = download_data("inventory.txt", cache=False)
    with fpath.open() as fid:
        for line in fid:
            fname = line.strip()
            download_data(fname)
    return data_home

PYMICRO_ROOT_DIR = pl.Path(__file__).parent.parent

def get_examples_data_dir(download_if_required=True) -> pl.Path:
    """Get the path to the examples data directory.

    Returns:
        pl.Path: the path to the examples data directory
    """
    data_home = os.environ.get("PYMICRO_DATA", str(pl.Path.home() / ".pymicro_data"))
    if os.path.exists(data_home):
        PYMICRO_EXAMPLES_DATA_DIR = pl.Path(data_home)
    elif download_if_required:
        PYMICRO_EXAMPLES_DATA_DIR = dowload_datadir()
    else:
        raise ValueError("Examples data directory does not exist, please specify environment variable PYMICRO_DATA or download the data directory")
            
    return PYMICRO_EXAMPLES_DATA_DIR

PYMICRO_XRAY_DATA_DIR = os.path.join(PYMICRO_ROOT_DIR, 'pymicro', 'xray', 'data')
