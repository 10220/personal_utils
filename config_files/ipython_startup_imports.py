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
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}\usepackage{amssymb}"

# Custom expanded color cycle
color_cycle = [
        (53/255.,  74/255.,  93/255.),   # black
        (59/255.,  153/255., 217/255.),  # blue
        (229/255., 126/255., 49/255.),   # orange
        (53/255.,  206/255., 116/255.),  # green
        (230/255., 78/255.,  67/255.),   # red
        (154/255., 91/255.,  179/255.),  # purple
        (240/255., 195/255., 48/255.),   # gold
        '#e377c2',                       # pink
        '#8c564b',                       # brown
        '#7f7f7f',                       # gray
        '#17becf',                       # teal
        '#bcbd22',                       # lime
    ]
# Set this as the default
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_cycle)

# Default mpl color cycle
original_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
