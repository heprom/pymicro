import setuptools

with open('README.rst', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="pymicro",
    version="0.4.4",
    author="Henry Proudhon",
    author_email="henry.proudhon@mines-paristech.fr",
    description="An open-source Python package to work with material microstructures and 3d data sets",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/heprom/pymicro",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    python_requires='>=3.5',
    license="MIT license",
)
