import numpy as np
import h5py
import sys

sys.path.append("/home/dante/Utils/")
from h5pytoolbox import get_modes, idx
from bag_of_tricks import update_progress


def compute_hdot(h, acc=8):
    """
    Computes the derivative of h with respect to Bondi time u.
    """
    from findiff import FinDiff

    d_dt = FinDiff(0, h["Time"][:], 1, acc=acc)
    hdot = {}
    modes = get_modes(h)
    NModes = len(modes)
    update_progress(0)
    for i in range(NModes):
        mode = modes[i]
        hdot[mode] = d_dt(h[idx(mode[0], mode[1])][:])
        update_progress(i, NModes)
    return hdot


def compute_supermomentum_vector(Plm):
    """
    Computes the 4-component Bondi supermomentum vector.
    """
    try:
        return np.array(
            [
                Plm[0, 0],
                (Plm[1, 1] - Plm[1, -1]) / np.sqrt(6),
                1j * (Plm[1, 1] + Plm[1, -1]) / np.sqrt(6),
                Plm[1, 0] / np.sqrt(3),
            ]
        )
    except AttributeError:
        return np.array(
            [
                Plm[idx(0, 0)][:],
                (Plm[idx(1, 1)][:] - Plm[idx(1, -1)][:]) / np.sqrt(6),
                1j * (Plm[idx(1, 1)][:] + Plm[idx(1, -1)][:]) / np.sqrt(6),
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


def compute_Plm(Psi2, h, cached_hdot=None):
    """
    Computes the supermomentum Plm.
    """

    def compute_I2(l, m, h, hdot):
        import spherical_functions as sf

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
                        * hdot[(L2, M)]
                        * sf.Wigner3j(l, L1, L2, m, -(m + M), M)
                    )
                L2_loop += (
                    np.sqrt(2 * L2 + 1) * sf.Wigner3j(l, L1, L2, 0, 2, -2) * M_loop
                )
            L1_loop += np.sqrt(2 * L1 + 1) * L2_loop
        return 0.5 * np.sqrt((2 * l + 1) / np.pi) * L1_loop

    if not cached_hdot:
        print("Preparing hdot. This might take a few minutes.")
        hdot = compute_hdot(h)
        print("Done preparing hdot")
    else:
        hdot = cached_hdot

    lmax = min(
        [np.array(get_modes(h))[:, 0].max(), np.array(get_modes(Psi2))[:, 0].max()]
    )
    Plm = {}
    it = 0
    tot = (lmax + 1) ** 2
    update_progress(it)
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            I1 = (-1) ** m * Psi2[idx(l, -m)][:]
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
    Computes Psi2 from h, Psi4, and Psi3 by integrating the bianchi identity, then writes output to a new file.
    """
    from spinsfast import map2salm
    from scipy.interpolate import InterpolatedUnivariateSpline

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
        ψ4 = ψ4_at_pts[pt]
        ðψ3 = ðψ3_at_pts[pt]
        dtψ2 = 0.25 * (-2 * ðψ3 + ψ4 * h0.conjugate())
        dtψ2_re = InterpolatedUnivariateSpline(h["Time"][:], dtψ2.real)
        dtψ2_im = InterpolatedUnivariateSpline(h["Time"][:], dtψ2.imag)
        ψ2_re = dtψ2_re.antiderivative()
        ψ2_im = dtψ2_im.antiderivative()
        ψ2_extract = ψ2_at_pts[pt]
        integration_const = (ψ2_extract[-1].real - ψ2_re(h["Time"][-1])) + 1j * (
            ψ2_extract[-1].imag - ψ2_im(h["Time"][-1])
        )
        new_ψ2[int(pt / (2 * lmax + 1)), pt % (2 * lmax + 1)] = (
            ψ2_re(h["Time"][:]) + 1j * ψ2_im(h["Time"][:]) + integration_const
        )

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


def output_Plm(h_file, Psi2_file, acc=8):
    """
    Computes the supermomentum Plm and writes to file.
    """
    with h5py.File(h_file, "r") as h, h5py.File(Psi2_file, "r") as Psi2:
        print("Computing hdot...")
        hdot = compute_hdot(h, acc=acc)
        print("\nComputing Plm...")
        Plm = compute_Plm(Psi2, h, cached_hdot=hdot)

        from shutil import copyfile

        fname = Psi2.filename.replace("r3Psi2OverM", "SupermomentumPlm")
        copyfile(Psi2.filename, fname)
        print("\nWriting to file...")
        with h5py.File(fname, "r+") as new_Plm:
            modes = get_modes(Psi2)
            it = 0
            tot = len(modes)
            update_progress(0)
            for (l, m) in get_modes(Psi2):
                new_Plm[idx(l, m)][:] = Plm[l, m]
                it += 1
                update_progress(it, tot)
        print("\nOutput to {}".format(fname))
