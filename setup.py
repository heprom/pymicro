import sys
import os
import setuptools
import pymicro

with open('README.rst', 'r') as f:
    long_description = f.read()

try:
    from distutils.command.build_py import build_py2to3 as build_py
    from distutils.command.build import build
except ImportError:
    from distutils.command.build_py import build_py
    from distutils.command.build import build 

## Load requirements.txt and format for setuptools.setup
requirements = []
dependency_links = []
here = os.path.abspath( os.path.dirname(__file__))
with open( os.path.join(here, "requirements.txt")) as fid:
    content = fid.read().split("\n")
    for line in content:
        if line.startswith( "#" ) or line.startswith( " " ) or line=="":
            continue
        elif line.startswith( "-e" ):
            pname = line.split("#egg=")[1]
            req_line = "{} @ {}".format( pname, line[3:] )
            requirements.append( req_line )
            dep_line = line.replace("-e", "").strip()
            dependency_links.append( dep_line )
        else:
            requirements.append( line )

## DISABLE C++ part compilation for basic-tools
os.environ["BASICTOOLS_DISABLE_MKL"] = "1"
os.environ["BASICTOOLS_DISABLE_OPENMP"] = "1"


all_packages = setuptools.find_packages(".", exclude=("examples","examples.*")) + ["pymicro." + x for x in setuptools.find_packages(".", exclude=("pymicro", "pymicro.*"))]

setuptools.setup(
    name="pymicro",
    version=pymicro.__version__,
    author="Henry Proudhon",
    author_email="henry.proudhon@mines-paristech.fr",
    description="An open-source Python package to work with material microstructures and 3d data sets",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/heprom/pymicro",
    packages=all_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    dependency_links=dependency_links,
    include_package_data=False,
    package_dir={'pymicro.examples': 'examples'},
    package_data = {'': ['*.png', '*.gif'],
    'pymicro.examples': ['data/*']},
    license="MIT license",
    cmdclass={'build':build}
)
