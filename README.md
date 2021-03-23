# Machine Learning Tutorial

### Numerical examples for regression and classification of data

  - Author: Alexander Hartmaier
  - Organization: ICAMS, Ruhr University Bochum, Germany
  - Contact: <alexander.hartmaier@rub.de>

Machine learning methods are trained with different data sets to work either as regression functions or to find the delimiter lines between classes of data points with different characteristics.

## Jupyter notebooks on Binder

The tutorial is conveniently used with Jupyter notebooks that can be directly accessed with Binder:  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AHartmaier/ML-Tutorial.git/HEAD)

[https://mybinder.org/v2/gh/AHartmaier/ML-Tutorial.git/HEAD]()

## Installation

To use the tutorial on your own hardware, you need an [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation with a recent Python version. Then follow those steps:

1. Download the contents of the [GitHub repository](https://github.com/AHartmaier/ML-Tutorial.git), e.g. with  
```
$ git clone https://github.com/AHartmaier/ML-Tutorial.git
```  
or download and unpack the ZIP archive directly from GitHub.

2. Change the working directory  
```
$ cd ML-Tutorial
```

3. Create a conda environment  
```
$ conda env create -f environment.yml
```

4. Activate the environment  
```
$ conda activate ml-tutorial
```

5. Start JupyterLab (or juypter notebook)  
```
$ jupyter lab
```
6. Open the jupyter notebooks (.ipynb) to follow the tutorials.

## De-Installation
If you want to remove the tutorial from your hardware, you need to follow those steps:
 
1. Deactivate the conda environment  
```
$ conda deactivate
```

2. Remove the environment  
```
$ conda env remove -n ml-tutorial
```

3. Delete the folder ML-Tutorial  
```
$ cd ..; rm -rf ML-Tutorial
```

## Dependencies

The tutorial uses the following packages, which are automatically installed in the environment when following the instruction above:

 - [NumPy](http://numpy.scipy.org) for array handling and mathematical operations
 - [scikit-learn](https://scikit-learn.org/stable/) for machine learning algorithms
 - [MatPlotLib](https://matplotlib.org/) for graphical output
 - [pandas](https://pandas.pydata.org/) for data import

## License

The software in this tutorial comes with ABSOLUTELY NO WARRANTY. This is free
software, and you are welcome to redistribute it under the conditions of
the GNU General Public License
([GPLv3](http://www.fsf.org/licensing/licenses/gpl.html))

The contents of the notebooks are published under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
([CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/))
