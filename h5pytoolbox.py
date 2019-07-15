import numpy as np
import h5py


def idx(first, second, third=None):
    """
    Helps with navigating the finite and extrapolated waveform files.

    *_FiniteRadii_CodeUnits.h5 : idx(R,l,m) with extraction radius R
    *_Asymptotic_GeometricUnits*.h5 : idx(N,l,m) with extrapolation order N
    
    """
    if third == None:
        if second >= 0:
            m = "+" + str(second)
        else:
            m = str(second)
        return "Data/l" + str(first) + "_m" + m
    elif first == -1:
        return "OutermostExtraction.dir/Y_l" + str(second) + "_m" + str(third) + ".dat"
    elif 0 < first < 10:
        return (
            "Extrapolated_N"
            + str(first)
            + ".dir/Y_l"
            + str(second)
            + "_m"
            + str(third)
            + ".dat"
        )
    else:
        return (
            "R"
            + str(first).zfill(4)
            + ".dir/Y_l"
            + str(second)
            + "_m"
            + str(third)
            + ".dat"
        )


def is_open_h5object(h5file):
    """
    Helper function for determining if function argument
    is an open HDF5 file or just the filename.
    """
    if type(h5file) == h5py._hl.files.File:
        return True
    else:
        return False


def identify_h5file(h5file):
    """
    Determines the data format of an HDF5 waveform file.
    """
    if is_open_h5object(h5file):
        if "Data" in sorted(h5file):
            return "Boyle"
        elif "OutermostExtraction.dir" in sorted(h5file):
            return "NRAR"
        else:
            return "FiniteRadii"
    else:
        with h5py.File(h5file, "r") as W:
            if "Data" in sorted(W):
                return "Boyle"
            elif "OutermostExtraction.dir" in sorted(W):
                return "NRAR"
            else:
                return "FiniteRadii"


def get_radii(h5file):
    """
    Returns the extraction radii from a FiniteRadii waveform file.
    """
    import re

    if identify_h5file(h5file) != "FiniteRadii":
        raise Exception("Can only use get_radii on FiniteRadii waveform files.")
    if is_open_h5object(h5file):
        radii = sorted(h5file)
        try:
            radii.remove("VersionHist.ver")
        except ValueError:
            pass
        radii = np.array(
            [int(re.search(r"""R(\d+).dir""", group)[1]) for group in sorted(radii)]
        )
        return radii
    else:
        with h5py.File(h5file, "r") as W:
            radii = sorted(W)
            try:
                radii.remove("VersionHist.ver")
            except ValueError:
                pass
            radii = np.array(
                [int(re.search(r"""R(\d+).dir""", group)[1]) for group in sorted(radii)]
            )
        return radii


def get_waveform(h5file, R, l, m):
    """
    Returns a complex numpy array of the waveform of a particular mode, and another numpy
    array of the corresponding time column. 
    
    The option h5file can either be an open HDF5 file or the path to an HDF5 file.
    
    The option R can either be the extraction radius of the waveform for a FiniteRadii file
    or the extrapolation order (-1 for OutermostExtraction.dir) for an NRAR file.
    """
    if is_open_h5object(h5file):
        return (
            h5file[idx(R, l, m)][:, 1] + 1j * h5file[idx(R, l, m)][:, 2],
            h5file[idx(R, l, m)][:, 0],
        )
    else:
        with h5py.File(h5file, "r") as W:
            return (
                W[idx(R, l, m)][:, 1] + 1j * W[idx(R, l, m)][:, 2],
                W[idx(R, l, m)][:, 0],
            )


def get_modes(h5file):
    """
    Returns a list of waveform modes available in an open HDF5 file.
    """
    from re import compile as re_compile

    if is_open_h5object(h5file):
        if (
            identify_h5file(h5file) == "FiniteRadii"
            or identify_h5file(h5file) == "NRAR"
        ):
            mode_regex = re_compile(r"""Y_l(?P<L>[0-9]+)_m(?P<M>[-+0-9]+)\.dat""")
            Wfs = sorted(h5file)
            try:
                Wfs.remove("VersionHist.ver")
            except ValueError:
                pass
            matches = [
                mode_regex.search(Wf)
                for Wf in sorted(h5file[Wfs[0]])
                if mode_regex.search(Wf) is not None
            ]
            modes = []
            for match in matches:
                modes.append((int(match.group(1)), int(match.group(2))))
            return modes

        elif identify_h5file(h5file) == "Boyle":
            mode_regex = re_compile(r"""l(?P<L>[0-9]+)_m(?P<M>[-+0-9]+)""")
            matches = [
                mode_regex.search(Wf)
                for Wf in sorted(h5file["Data"])
                if mode_regex.search(Wf) is not None
            ]
            modes = []
            for match in matches:
                modes.append((int(match.group(1)), int(match.group(2))))
            return modes

    else:
        with h5py.File(h5file, "r") as W:
            if identify_h5file(W) == "FiniteRadii" or identify_h5file(W) == "NRAR":
                mode_regex = re_compile(r"""Y_l(?P<L>[0-9]+)_m(?P<M>[-+0-9]+)\.dat""")
                Wfs = sorted(W)
                try:
                    Wfs.remove("VersionHist.ver")
                except ValueError:
                    pass
                matches = [
                    mode_regex.search(Wf)
                    for Wf in sorted(W[Wfs[0]])
                    if mode_regex.search(Wf) is not None
                ]
                modes = []
                for match in matches:
                    modes.append((int(match.group(1)), int(match.group(2))))
                return modes

            elif identify_h5file(W) == "Boyle":
                mode_regex = re_compile(r"""l(?P<L>[0-9]+)_m(?P<M>[-+0-9]+)""")
                matches = [
                    mode_regex.search(Wf)
                    for Wf in sorted(W["Data"])
                    if mode_regex.search(Wf) is not None
                ]
                modes = []
                for match in matches:
                    modes.append((int(match.group(1)), int(match.group(2))))
                return modes


def swsh(s, modes, theta, phi, psi=0):
    """
    Return a value of a spin-weighted spherical harmonic of spin-weight s. 
    If passed a list of several modes, then a numpy array is returned with 
    SWSH values of each mode for the given point.
    
    For one mode:       swsh(s,[(l,m)],theta,phi,psi=0)
    For several modes:  swsh(s,[(l1,m1),(l2,m2),(l3,m3),...],theta,phi,psi=0)
    """
    import spherical_functions as sf
    import quaternion as qt

    return sf.SWSH(qt.from_spherical_coords(theta, phi), s, modes) * np.exp(
        1j * s * psi
    )


def get_at_points(h5file, times, points, s=-2, R=None, eth=0):
    """
    Computes the value of a waveform quantity from an open HDF5 file (or a
    path to an HDF5 file) storing the spin-weighted spherical harmonic (SWSH)
    mode weights. Given a list of timestep indices and a list of points on the
    sphere, this function returns an array of waveform values of shape 
    (N_points, N_times).
    
    OPTIONS:
    
    - h5file: Either an open HDF5 file or path to an HDF5 file.
    
    - times:  A list of the timestep indices at which to compute the waveform quantity. 
              You can use slice(t0,t1) to pick all times from t0 to t1. 
    
    - points: Can either be a list of (theta,phi,psi) triples where psi is the 
              orientation of the SWSH at (theta,phi), or it can be a list of (theta,phi) 
              pairs assuming psi=0. 
              
    - s: Spin-weight of the waveform quantity. (Default: -2)
    
    - R: If the file is a FiniteRadii file, then R is the radius at which to compute 
         the waveform quantity. If the file is an NRAR file, then R is the extrapolation
         order (-1 for OutmostExtraction.dir) at which to compute the waveform quantity.
         If the file is in the default scri extrapolation format, then this argument is 
         not required.

    - eth: Apply the spin-weight raising (if eth > 0) or lower (if eth < 0) operator.
           This argument specifies how many times to apply the operator.

    NOTE: The first time this function is run, it may take some time to import the 
    spherical_functions and quaternion python modules.
    """

    def get_at_points_work(h5, times, points, s=-2, R=None, eth=0):
        if np.array(points).shape[1] == 2:
            swshes = np.array(
                [swsh(s + eth, get_modes(h5), theta, phi, 0) for (theta, phi) in points]
            )
        elif np.array(points).shape[1] == 3:
            swshes = np.array(
                [
                    swsh(s + eth, get_modes(h5), theta, phi, psi)
                    for (theta, phi, psi) in points
                ]
            )
        else:
            raise Exception(
                "'points' must be a list of (theta,phi) pairs assuming psi=0 or a list "
                "of (theta,phi,psi) triples."
            )
        if R is not None:
            if eth == 0:
                weights = np.array(
                    [
                        h5[idx(R, *mode)][times, 1] + 1j * h5[idx(R, *mode)][times, 2]
                        for mode in get_modes(h5)
                    ]
                )
            elif eth == 1:
                weights = np.array(
                    [
                        np.sqrt((mode[0] - s) * (mode[0] + s + 1))
                        * (
                            h5[idx(R, *mode)][times, 1]
                            + 1j * h5[idx(R, *mode)][times, 2]
                        )
                        for mode in get_modes(h5)
                    ]
                )
            elif eth == 2:
                weights = np.array(
                    [
                        np.sqrt(
                            (mode[0] - s)
                            * (mode[0] + s + 1)
                            * (mode[0] - s - 1)
                            * (mode[0] + s + 2)
                        )
                        * (
                            h5[idx(R, *mode)][times, 1]
                            + 1j * h5[idx(R, *mode)][times, 2]
                        )
                        for mode in get_modes(h5)
                    ]
                )
            elif eth == -1:
                weights = np.array(
                    [
                        (-np.sqrt((mode[0] + s) * (mode[0] - s + 1)))
                        * (
                            h5[idx(R, *mode)][times, 1]
                            + 1j * h5[idx(R, *mode)][times, 2]
                        )
                        for mode in get_modes(h5)
                    ]
                )
            else:
                raise Exception(
                    "This successive eth derivative has not been coded up yet"
                )
        else:
            if eth == 0:
                weights = np.array([h5[idx(*mode)][times] for mode in get_modes(h5)])
            elif eth == 1:
                weights = np.array(
                    [
                        np.sqrt((mode[0] - s) * (mode[0] + s + 1))
                        * h5[idx(*mode)][times]
                        for mode in get_modes(h5)
                    ]
                )
            elif eth == 2:
                weights = np.array(
                    [
                        np.sqrt(
                            (mode[0] - s)
                            * (mode[0] + s + 1)
                            * (mode[0] - s - 1)
                            * (mode[0] + s + 2)
                        )
                        * h5[idx(*mode)][times]
                        for mode in get_modes(h5)
                    ]
                )
            elif eth == -1:
                weights = np.array(
                    [
                        (-np.sqrt((mode[0] + s) * (mode[0] - s + 1)))
                        * h5[idx(*mode)][times]
                        for mode in get_modes(h5)
                    ]
                )
            else:
                raise Exception(
                    "This successive eth derivative has not been coded up yet"
                )

        val = swshes.dot(weights)
        return val

    if identify_h5file(h5file) == "Boyle":
        if is_open_h5object(h5file):
            return get_at_points_work(h5file, times, points, s=s, eth=eth)
        else:
            with h5py.File(h5file, "r") as W:
                return get_at_points_work(W, times, points, s=s, eth=eth)
    else:
        if R is None:
            raise Exception(
                "The 'R' argument is required for a FiniteRadii or NRAR file."
            )
        if is_open_h5object(h5file):
            return get_at_points_work(h5file, times, points, s=s, R=R, eth=eth)
        else:
            with h5py.File(h5file, "r") as W:
                return get_at_points_work(W, times, points, s=s, R=R, eth=eth)


def format_group_names(h5file, dry_run=False):
    """
    Removes the prefix of groups in a FiniteRadii waveform file. If there is
    no prefix then this does nothing. Ex:
        "Psi2_R0100.dir" would be renamed to the standard "R0100.dir"

    If the argument is an already opened HDF5 file, then it must have write
    permission! Set 'dry_run=True' for it to print out what the changes would
    be without actually overwriting the group names in the file.
    """
    import re

    if is_open_h5object(h5file):
        grps = sorted(h5file)
        pattern = r"""R\d{4}.dir"""
        for grp in grps:
            p = re.search(pattern, grp)
            if p:
                print("Changing {} to {}".format(grp, p[0]))
                if not dry_run:
                    h5file.move(grp, p[0])

    else:
        with h5py.File(h5file, "r+") as F:
            grps = sorted(F)
            pattern = r"""R\d{4}.dir"""
            for grp in grps:
                p = re.search(pattern, grp)
                if p:
                    print("Changing {} to {}".format(grp, p[0]))
                    if not dry_run:
                        F.move(grp, p[0])
    return


def quick_plot(h5file, part, R=0, l=2, m=2):
    """
    Plot a sample waveform from a datafile.
    """
    import matplotlib.pyplot as plt
    import re

    if part == np.real:
        ylabel = r"$ \Re \left ( "
    elif part == np.imag:
        ylabel = r"$ \Im \left ( "
    elif part == np.abs or abs:
        ylabel = r"$ \left | "
    else:
        raise Exception("'part' must be np.real, np.imag, or np.abs")

    if type(h5file) is str:
        fname = h5file
        f = h5py.File(fname, "r")
    else:
        fname = h5file.filename
        f = h5file
    # Finte radii waveforms
    try:
        radius = int(re.search(r"""R(\d+).dir""", sorted(f)[R])[1])
        data, time = get_waveform(f, radius, l, m)
        data = part(data)
    except KeyError:
        pass
    # Extrapolated waveforms (NRAR)
    try:
        tag = "Y_l" + str(l) + "_m" + str(m) + ".dat"
        time = f[tag][:, 0]
        data = part(f[tag][:, 1] + 1j * f[tag][:, 2])
    except KeyError:
        pass
    # Extrapolated waveforms (SXS Catalog)
    try:
        tag = "Extrapolated_N4.dir/Y_l" + str(l) + "_m" + str(m) + ".dat"
        time = f[tag][:, 0]
        data = part(f[tag][:, 1] + 1j * f[tag][:, 2])
    except KeyError:
        pass
    # Extrapolated waveforms (GWFrames/scri default)
    try:
        time = f["Time"][:]
        data = part(f[idx(l, m)][:])
    except KeyError:
        pass

    match = re.search(r"""r(\d{0,1})(Psi|h|Phi)(\d{0,1})""", fname)
    if match is not None:
        if match[2] == 'Psi':
            if not match[1] == '':
                ylabel += r"^{"+match[1]+"}"
            ylabel += r"r\Psi_"
            ylabel += r"{"+match[3]+r"}"
        elif match[2] == 'h':
            ylabel += r"rh"
        elif match[2] == 'Phi':
            ylabel += r"\Phi"
            if 'PhiMinus' in fname:
                ylabel += r"_{-}"
            if 'PhiPlus' in fname:
                ylabel += r"_{+}"
    else:
        ylabel += r"F"
    ylabel += r"^{("+str(l)+r","+str(m)+r")}"
    if part == np.real or part == np.imag:
        ylabel += r" \right ) $"
    elif part == np.abs or abs:
        ylabel += r" \right | $"

    plt.figure(dpi=400)
    plt.plot(time, data)
    plt.xlabel('Coordinate Time (M)')
    plt.ylabel(ylabel)
    plt.show()
    if type(h5file) is str:
        f.close()


# tet_psi_data = {}
# tet_psi_slm = {}
# for i in range(5):
#     tet_psi_data[i] = np.array([np.loadtxt(path1+"TetradPsi"+str(i)+"_R0"+str(R)+".dat") for R in radii])
#
#     psi_map = {}
#     spin_weight = 2-i
#     for R in range(len(radii)):
#         psi_map[i] = tet_psi_data[i][R][:,1:]
#         n_points_on_sphere = psi_map[i].shape[1] // 2
#         two_ell_plus_1 = int(math.sqrt(n_points_on_sphere))  # Assuming n_theta==n_phi
#         ell_max = (two_ell_plus_1 - 1) // 2
#         psi_map[i] = psi_map[i].flatten().view(dtype=complex).reshape((tet_psi_data[i][R].shape[0], two_ell_plus_1, two_ell_plus_1))
#         tet_psi_slm[i,R] = np.array([sp.map2salm(psi_map[i][j], spin_weight, ell_max) for j in range(psi_map[i].shape[0])])
