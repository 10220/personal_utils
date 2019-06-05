#!/usr/bin/env python2
"""
Prepares a waveform file recording a single timestep of data to be used
for extrapolation by the 'scri' package. 

The single timestep is copied many times with ascending values of coordinate
time so that scri will successfully extrapolate the waveform. 
"""

import numpy as np
import h5py
import argparse

import sys
sys.path.append('/home/dante/Documents/spec/Support/Python/')
from UpdateH5DataVersion import get_list_of_h5files

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    'input',
    type=str,
    help='Name of the HDF5 waveform file to edit')
p.add_argument(
    '--recursive',
    '-r',
    dest='recursive',
    action='store_true',
    default=False,
    help='Perform for all H5 files in the directory and subdirectories.')
p.add_argument(
    '--NTimes',
    type=int,
    default=1000,
    help='Number of timesteps desired in output waveform')
p.add_argument(
    '--final-t',
    dest='final_t',
    type=float,
    default=300.0,
    help='Value of the final time')
args = p.parse_args()

list_of_h5files = get_list_of_h5files(args)

for h5file in list_of_h5files:
    with h5py.File(h5file,'r+') as f:
        groups = sorted(f)
        subgroups = sorted(f[groups[0]])
        subgroups.remove('InitialAdmEnergy.dat')
        for group in groups:
            for subgroup in subgroups:
                idx = group + '/' + subgroup
                data = f[idx][0]
                new_data = np.array([data] * args.NTimes)
                new_data[:,0] = np.linspace(0, args.final_t, args.NTimes)
                del f[idx]
                f.create_dataset(idx, data=new_data)
