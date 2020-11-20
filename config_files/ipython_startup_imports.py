# Make a link named 'imports.py' to this script in ~/.ipython/profile_default/startup/
#   > cd ~/.ipython/profile_default/startup/
#   > ln -s /path/to/this/script.py imports.py

# Always helpful
import numpy as np
from scipy import constants
from scipy.interpolate import InterpolatedUnivariateSpline, CubicSpline

# -----------------------------
# Tools for HDF5 Analysis
# -----------------------------
import h5py
import sys
sys.path.insert(0, '/home/dante/Utils/')
from h5pytoolbox import *

# For importing files
# This is indexed by scri dataType
# e.g. data_type_file_label[scri.h]
data_type_file_label = {
    6 : 'r2sigmaOverM',
    7 : 'rhOverM',
    5 : 'rMPsi4',
    4 : 'r2Psi3',
    3 : 'r3Psi2OverM',
    2 : 'r4Psi1OverM2',
    1 : 'r5Psi0OverM3',
}

#--------------------------
# Set up pretty plots
#--------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt

# Hi res plots
plt.rcParams['figure.figsize']=(7,3)
plt.rcParams['figure.dpi']=500
# Plot defaults
plt.rcParams['lines.linewidth']=0.75
plt.rcParams['grid.linewidth']=0.25
plt.rcParams['grid.alpha']=0.25
plt.rcParams['axes.grid']=True

# Nice font for text
plt.rcParams.update({
    'font.family' : 'FreeSerif',
    'text.usetex' : False,
    'mathtext.fontset' : 'stix' })
# Nice font for LaTeX equations
mpl.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}"]

# set color cycle variable
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
