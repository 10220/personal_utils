#!/usr/bin/env python
"""
Extrapolate SpEC waveforms with scri
"""

import argparse
from os.path import isfile, join

p = argparse.ArgumentParser(description=__doc__, 
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument(
    '--filenames',
    '-f',
    nargs='+',
    type=int,
    default=[6,5,4,3,2,1,0],
    help='Waveforms to extrapolate, using the guide:\n'
         '    6: r2sigma_FiniteRadii_CodeUnits.h5\n'
         '    5: rh_FiniteRadii_CodeUnits.h5\n'
         '    4: rPsi4_FiniteRadii_CodeUnits.h5\n'
         '    3: r2Psi3_FiniteRadii_CodeUnits.h5\n'
         '    2: r3Psi2_FiniteRadii_CodeUnits.h5\n'
         '    1: r4Psi1_FiniteRadii_CodeUnits.h5\n'
         '    0: r5Psi0_FiniteRadii_CodeUnits.h5\n')
p.add_argument(
    '--InDir',
    '-i',
    dest='InDir',
    type=str,
    default='.',
    help='Directory holding waveform files for extrapolation')
p.add_argument(
    '--OutDir',
    '-o',
    dest='OutDir',
    type=str,
    default='.',
    help='Directory for output')
p.add_argument(
    '--diffs',
    '-d',
    action='store_true',
    default=False,
    help='Output convergence data files')
p.add_argument(
    '--plot-diffs',
    '-p',
    dest='plot_diffs',
    action='store_true',
    default=False,
    help='Make convergence plots')
p.add_argument(
    '--ChMass',
    type=float,
    default=1.0,
    help='Christodoulou mass used for scaling')
p.add_argument(
    '--horizons',
    type=str,
    required=False,
    help='The Horizons.h5 file to use for finding the ChMass value. '
         'If given, this overrides --ChMass.')
p.add_argument(
    '--extrap-orders',
    dest='extrap_orders',
    nargs='+',
    type=int,
    default=[-1,2,3,4,5,6],
    help='Terms used in the fitting polynomial. Negative numbers correspond '
         'to extracted data, counting down from the outermost extraction '
         'radius (which is -1).')
p.add_argument(
    '--scri-format',
    dest='scri_format',
    action='store_true',
    default=False,
    help='Use the default scri format instead of the NRAR format for '
         'the outputted files.')
p.add_argument(
    '--custom-filename',
    '-c',
    dest='custom_filename',
    type=str,
    required=False,
    help='Name of the HDF5 waveform file to extrapolate, if not one '
         'of the files listed under the --filenames options. If given, '
         'this overrides --filenames.')
args = p.parse_args()

# import scri takes a while (~10 seconds). We 
# do the import after parsing args so that it
# doesn't take forever if the user only wants
# to read the help text.
from scri.extrapolation import extrapolate

plot_format = ''
diff_files  = ''
if args.plot_diffs:
    plot_format = 'pdf'
    diff_files  = 'ExtrapConvergence_N{N}-N{Nm1}.h5'
elif args.diffs:
    diff_files  = 'ExtrapConvergence_N{N}-N{Nm1}.h5'

if args.horizons:
    ChMass = 0.0
    horizons_file = args.horizons
else:
    ChMass = args.ChMass
    horizons_file = ''

if max(args.filenames) > 6 or min(args.filenames) < 0:
    raise Exception('The --filenames names option was not specified '
                    'correctly. Only the numbers in the help text may '
                    'be supplied as arguments.\n\nE.g. If you only want '
                    'to extrapolate rh and rPsi4, then pass in:\n'
                    '\textrapolate --filenames 5 4')

if args.custom_filename:
    extrapolate(
        InputDirectory = args.InDir,
        OutputDirectory = args.OutDir,
        DataFile = args.custom_filename,
        ChMass = ChMass,
        HorizonsFile = horizons_file,
        ExtrapolationOrders=args.extrap_orders,
        UseStupidNRARFormat = not args.scri_format,
        DifferenceFiles = diff_files,
        PlotFormat = plot_format 
    )

else:
    for psi in args.filenames:
        if psi==6: 
            filename = 'r2sigma_FiniteRadii_CodeUnits.h5'
        elif psi==5: 
            filename = 'rh_FiniteRadii_CodeUnits.h5'
        else:
            r_tag = 'r'+str(5-psi) if psi != 4 else 'r'
            filename = r_tag + 'Psi'+ str(psi) + '_FiniteRadii_CodeUnits.h5'

        if isfile(join(args.InDir,filename)):
            extrapolate(
                InputDirectory = args.InDir,
                OutputDirectory = args.OutDir,
                DataFile = filename,
                ChMass = ChMass,
                HorizonsFile = horizons_file,
                ExtrapolationOrders=args.extrap_orders,
                UseStupidNRARFormat = not args.scri_format,
                DifferenceFiles = diff_files,
                PlotFormat = plot_format 
            )
