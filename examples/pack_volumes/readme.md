# Compressing X-ray tomography volumes with multiple semantic segmentations (voxel-wise categorical values)

## Install

Install a minimal conda env for the pack_volumes.ipynb example.

My conda version:

`conda -V`

```
conda 4.9.0
```
### Create a conda env

Sequence of commands that I did:

```
conda create --name packenv python=3.8
conda activate packenv
conda install -c conda-forge basictools lxml pytables ipykernel
```
### `pymicro`

`pymicro` is not on conda, so let's install it with pip

```
~/.conda/envs/packenv/bin/pip install git+https://github.com/heprom/pymicro.git
```

You might have to change the path to get the correct pip (in your conda env). 

You should look for the `pip` file at somewhere like `/path/to/my/env/bin/`.

### `jupyter lab` (or `jupyter notebook` if you prefer)

#### option 1

If you want to install jupyter notebook/lab in the same env:

```
conda install -c conda-forge jupyterlab
```

#### option 2

If you already have a jupyter notebook/lab installed, you can create a kernel in env you just created (`packenv`) and use it from there:

```
ipython kernel install --user --name=packenv
```

Restart your jupyter notebook/lab sever (or launch another one with a custom port).
You will see the kernel `packenv` available.

Then:

```
conda deactivate
```

Have fun (:

## Env

See the files:
 - `packenv-hist.yml`: from `conda env export --from-history`
 - `packenv.yml`: from `conda env export`
 
## Data

The files used in this example can be downloaded in the link below. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4597288.svg)](https://doi.org/10.5281/zenodo.4597288)


Extract them to a folder `raws` in the same directory of the tutorial. 

These are mock volumes from a glass fiber-reinforced polyamide 66 (`pa66`) image.

---

author: [joaopcbertoldo](joaopcbertoldo.github.io)

---

# Todos

 - reload data after writing it
 - plot it
 - get the last bit of the tutorial and put it with the published data to show how to open a sample data
