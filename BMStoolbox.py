import numpy as np
import h5py
import sys
from scipy.interpolate import InterpolatedUnivariateSpline

sys.path.append("/home/dante/Utils/")
from h5pytoolbox import get_modes, idx, identify_h5file, is_open_h5object
from bag_of_tricks import update_progress


def get_derivative(f, t, n=1, out_t=None):
    """
    Take the (anti)derivative of a function. If out_t is specified
    then the derivative will be evaulated at those times instead.
    """
    if out_t is None:
        out_t = t
    # derivative
    if n >= 1:
        f_re = InterpolatedUnivariateSpline(t, f.real, k=3 + n)
        f_im = InterpolatedUnivariateSpline(t, f.imag, k=3 + n)
        return f_re.derivative(n)(out_t) + 1j * f_im.derivative(n)(out_t)
    # antiderivative
    elif n <= -1:
        f_re = InterpolatedUnivariateSpline(t, f.real, k=3)
        f_im = InterpolatedUnivariateSpline(t, f.imag, k=3)
        return f_re.antiderivative(-n)(out_t) + 1j * f_im.antiderivative(-n)(out_t)
    else:
        raise Exception("n must be >= 1 or <= -1")


def get_waveform_derivative(W, n=1, out_t=None):
    """
    Take the derivative of every modes in a waveform file and return a dictionary
    that is accessed by the same H5 groups as the original file.
    """
    if out_t is None:
        out_t = W["Time"][:]
    Wdot = {}
    for (l, m) in get_modes(W):
        Wdot[idx(l, m)] = get_derivative(W[idx(l, m)][:], W["Time"][:], n, out_t=out_t)
    return Wdot


def map_to_new_domain(f, t, t_new):
    """
    Interpolates a complex function to a new set of times.
    """
    f_re = InterpolatedUnivariateSpline(t, f.real)
    f_im = InterpolatedUnivariateSpline(t, f.imag)
    return f_re(t_new) + 1j * f_im(t_new)


def find_t_merger(h):
    """
    Find the time of merger for scri.WaveformModes object of type h. The 
    merger time is defined as the peak of the L2 Norm of all the modes of rh. 
    """
    return h.t[(np.abs(h.data)**2).sum(axis=1).argmax()]


def compute_supermomentum_vector(Plm):
    """
    Computes the 4-component Bondi supermomentum vector.
    """
    try:
        return np.array(
            [
                Plm[0, 0],
                (Plm[1, -1] - Plm[1, 1]) / np.sqrt(6),
                1j * (Plm[1, -1] + Plm[1, 1]) / np.sqrt(6),
                Plm[1, 0] / np.sqrt(3),
            ]
        )
    except AttributeError:
        return np.array(
            [
                Plm[idx(0, 0)][:],
                (Plm[idx(1, -1)][:] - Plm[idx(1, 1)][:]) / np.sqrt(6),
                1j * (Plm[idx(1, -1)][:] + Plm[idx(1, 1)][:]) / np.sqrt(6),
                Plm[idx(1, 0)][:] / np.sqrt(3),
            ]
        )


def compute_super_rest_mass(P):
    """
    Computes the super rest mass from 4-component Bondi supermomentum vector.
    """
    return np.sqrt(
        np.array([-P[i] * P[i] * np.array([-1, 1, 1, 1])[i] for i in range(4)]).sum(
            axis=0
        )
    )


def compute_Plm(Psi2, h):
    """
    Computes the supermomentum Plm.
    """

    def compute_I2(l, m, h, hdot):
        from spherical_functions import Wigner3j

        L1max = int(np.array(get_modes(h))[:, 0].max())
        L1min = int(np.array(get_modes(h))[:, 0].min())
        L1_loop = 0
        for L1 in range(L1min, L1max + 1):
            L2max = int(l + L1)
            L2min = int(np.abs(l - L1))
            L2_loop = 0
            for L2 in range(L2min, L2max + 1):
                if L2 < L1min or L2 > L1max:
                    continue
                M_loop = 0
                for M in range(-L2, L2 + 1):
                    if np.abs(M + m) > L1:
                        continue
                    M_loop += (
                        (-1) ** (m + M)
                        * h[idx(L1, m + M)][:].conjugate()
                        * hdot[idx(L2, M)]
                        * Wigner3j(l, L1, L2, m, -(m + M), M)
                    )
                L2_loop += np.sqrt(2 * L2 + 1) * Wigner3j(l, L1, L2, 0, 2, -2) * M_loop
            L1_loop += np.sqrt(2 * L1 + 1) * L2_loop
        return 0.5 * np.sqrt((2 * l + 1) / np.pi) * L1_loop

    hdot = get_waveform_derivative(h)
    lmax = min(
        [np.array(get_modes(h))[:, 0].max(), np.array(get_modes(Psi2))[:, 0].max()]
    )
    Plm = {}
    it = 0
    tot = (lmax + 1) ** 2
    update_progress(it)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            I1 = (-1) ** m * map_to_new_domain(
                Psi2[idx(l, -m)][:], Psi2["Time"][:], h["Time"][:]
            )
            I2 = compute_I2(l, m, h, hdot)
            I3 = (
                0
                if l < np.array(get_modes(h))[:, 0].min()
                else np.sqrt((l + 2) * (l + 1) * l * (l - 1))
                * (-1) ** m
                * h[idx(l, -m)][:]
            )
            Plm[l, m] = -0.125 * (4 * I1 + I2 + I3) / np.sqrt(np.pi)
            it += 1
            update_progress(it, tot)
    return Plm


def compute_Psi2_from_Bianchi(h, Psi4, Psi3, Psi2):
    """
    Computes Psi2 from h, Psi4, and Psi3 by integrating the bianchi identity, then 
    writes output to a new file.
    """
    from spinsfast import map2salm

    lmax = min(
        [np.array(get_modes(h))[:, 0].max(), np.array(get_modes(Psi2))[:, 0].max()]
    )
    theta_phi = np.array(
        [
            [
                [theta, phi]
                for phi in np.linspace(0.0, 2 * np.pi, num=2 * lmax + 1, endpoint=False)
            ]
            for theta in np.linspace(0.0, np.pi, num=2 * lmax + 1, endpoint=True)
        ]
    )
    points = theta_phi.reshape(((2 * lmax + 1) ** 2, 2))

    times = slice(None, None)
    h_at_pts = get_at_points(h, times, points)
    ψ4_at_pts = get_at_points(Psi4, times, points)
    ðψ3_at_pts = get_at_points(Psi3, times, points, s=-1, eth=1)
    ψ2_at_pts = get_at_points(Psi2, times, points, s=0)

    new_ψ2 = np.empty((2 * lmax + 1, 2 * lmax + 1, len(h["Time"][:])), dtype=complex)

    for pt in range(len(points)):
        h0 = h_at_pts[pt]
        ψ4 = map_to_new_domain(ψ4_at_pts[pt], Psi4["Time"][:], h["Time"][:])
        ðψ3 = map_to_new_domain(ðψ3_at_pts[pt], Psi3["Time"][:], h["Time"][:])
        dtψ2 = 0.25 * (-2 * ðψ3 + ψ4 * h0.conjugate())

        ψ2 = get_derivative(dtψ2, h["Time"][:], n=-1)

        ψ2_extract = map_to_new_domain(ψ2_at_pts[pt], Psi2["Time"][:], h["Time"][:])
        integration_const = ψ2_extract[-1] - ψ2[-1]

        new_ψ2[int(pt / (2 * lmax + 1)), pt % (2 * lmax + 1)] = ψ2 + integration_const

    spin_weight = 0
    psi2_slm = np.array(
        [map2salm(new_ψ2[:, :, i], spin_weight, lmax) for i in range(new_ψ2.shape[2])]
    )

    from shutil import copyfile

    fname = Psi2.filename.replace("Psi2OverM", "Psi2OverM_Bianchi")
    copyfile(Psi2.filename, fname)

    with h5py.File(fname, "r+") as new_Psi2:
        modes = [(ell, m) for ell in range(lmax + 1) for m in range(-ell, ell + 1)]
        file_modes = get_modes(Psi2)
        for i in range(len(modes)):
            mode = modes[i]
            if mode in file_modes:
                new_Psi2[idx(mode[0], mode[1])][:] = psi2_slm[:, i]
    print("Outputted to {}".format(fname))
    return


def output_Plm(h_file, Psi2_file):
    """
    Computes the supermomentum Plm and writes to file.
    """
    with h5py.File(h_file, "r") as h, h5py.File(Psi2_file, "r") as Psi2:
        print("Computing Plm...")
        Plm = compute_Plm(Psi2, h)

        from shutil import copyfile

        fname = Psi2.filename.replace("r3Psi2OverM", "SupermomentumPlm")
        copyfile(Psi2.filename, fname)
        print("\nWriting to file...")
        with h5py.File(fname, "r+") as new_Plm:
            if h["Time"][0] < Psi2["Time"][0]:
                t0_idx = (np.abs(h["Time"][:] - Psi2["Time"][0])).argmin()
            else:
                t0_idx = 0
            del new_Plm["Time"]
            new_Plm.create_dataset("Time", data=h["Time"][t0_idx:])
            modes = get_modes(Psi2)
            it = 0
            tot = len(modes)
            update_progress(0)
            for (l, m) in get_modes(Psi2):
                del new_Plm[idx(l, m)]
                new_Plm.create_dataset(idx(l, m), data=Plm[l, m][t0_idx:])
                it += 1
                update_progress(it, tot)
        print("\nOutput to {}".format(fname))
