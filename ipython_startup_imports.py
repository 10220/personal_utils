# Make a link named 'imports.py' to this script in ~/.ipython/profile_default/startup/
#   > cd ~/.ipython/profile_default/startup/
#   > ln -s /path/to/this/script.py imports.py

# Always helpful
import numpy as np
import scipy as sp
from scipy import constants

# Tools for HDF5 Analysis
import h5py
import sys
sys.path.insert(0, '/home/dante/Utils/')
from h5pytoolbox import *

# Set up pretty plots
import matplotlib as mpl
import matplotlib.pyplot as plt
# Nice font for text
plt.rcParams.update({
    'font.family' : 'FreeSerif',
    'text.usetex' : False,
    'mathtext.fontset' : 'stix' })
# Nice font for LaTeX equations
mpl.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}"]
