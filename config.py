import os
import pathlib as pl 

PYMICRO_ROOT_DIR = pl.Path(__file__).parent
PYMICRO_EXAMPLES_DATA_DIR = PYMICRO_ROOT_DIR / 'examples' / 'data'

if not PYMICRO_EXAMPLES_DATA_DIR.exists():
    data_home = os.environ.get("PYMICRO_DATA", pl.Path.home() / ".pymicro_data")
    PYMICRO_EXAMPLES_DATA_DIR = pl.Path(data_home)
    
PYMICRO_XRAY_DATA_DIR = os.path.join(PYMICRO_ROOT_DIR, 'pymicro', 'xray', 'data')
