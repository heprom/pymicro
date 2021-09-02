# source me!
# not available for power 9 machines! because of basictools


# source ~/init_conda_power9.sh
ENV_NAME="pymicro-esrf-slurm-intel"


if [[ ! $(command -v conda) ]]; then
    echo "Please load conda in your environment first!"
    exit 1
else
    echo -e "Using conda from (type -a conda): \n$(type -a conda)"
fi;

if [ -z "$ENV_NAME" ]
then
    echo "\$ENV_NAME is empty, please provide an environment name"
else    
    echo "\$ENV_NAME=${ENV_NAME}"
fi

SCRIPTDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
PYMICRO_DIR="$(realpath $SCRIPTDIR/../..)"

echo "creating env ${ENV_NAME}"
conda create -y --name $ENV_NAME

echo "activating env ${ENV_NAME}"
conda activate $ENV_NAME

echo "installing stuff in ${ENV_NAME}"
# from henry
conda install -y -c conda-forge --name $ENV_NAME vtk=9.0.1 basictools ipython pytables lxml scikit-image h5py scipy
# from me
conda install -y --name $ENV_NAME ipykernel pip viztracer

# for compatibility of widgets in esrf's jupyter slurm
echo  -e "installing esrf's jupyter-slurm-compatible stuff with pip \nwhich pip: $(which pip)"
pip install ipympl==0.5.8 matplotlib==3.1.2

echo "installing pymicro in develop mode with pip"
cd $PYMICRO_DIR
pip install -e .

echo -e  "installing ipykernel\nwhich python: $(which python)"
python -m ipykernel install --name ${ENV_NAME} --user

conda deactivate