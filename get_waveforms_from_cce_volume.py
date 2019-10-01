#!/usr/bin/env python
"""
Takes the raw data from a CCE volume HDF5 file and writes NRAR-formatted
HDF5 waveform files for the Weyl scalars and the strain.

"""

import numpy as np
import h5py
import argparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "filename",
    type=str,
    help="Name of the CCE volume data HDF5 file"
)
p.add_argument(
    "--lmax",
    type=int,
    default=8,
    help="Max spherical harmonic ell value in the output waveforms"
)

args = p.parse_args()


def get_waveforms_from_cce_volume(filename, lmax=8):
    # scri takes a while to import, so only import it when needed
    import scri
    from scri import h, psi4, psi3, psi2, psi1, psi0

    output_extension = "CCE_" + filename.split("CceVolume")[-1]

    def real_idx(l, m):
        return 2 * (l ** 2 + l + m)

    def imag_idx(l, m):
        return 2 * (l ** 2 + l + m) + 1

    with h5py.File(filename, "r") as cce_volume_file:
        print("Preparing to extract waveforms from {}".format(filename))
        scri_data = cce_volume_file.get("cce_scri_data.vol")
        time_ids_and_values = [
            (x, scri_data.get(x).attrs["observation_value"]) for x in scri_data.keys()
        ]
        time_ids_and_values = sorted(time_ids_and_values, key=lambda x: x[1])

        for data_type in [h, psi4, psi3, psi2, psi1, psi0]:
            if data_type is h:
                raw_data = []
                time_set = []
                for (time_id, time) in time_ids_and_values:
                    time_set.append(time)
                    raw_data.append(scri_data[time_id]["Strain"][()])
                raw_data = np.array(raw_data)
                time_set = np.array(time_set)

                modes = [(l, m) for l in range(0, lmax + 1) for m in range(-l, l + 1)]
                mode_data = []
                for (l, m) in modes:
                    mode_data.append(
                        raw_data[:, real_idx(l, m)] + 1j * raw_data[:, imag_idx(l, m)]
                    )
                mode_data = np.array(mode_data).T

                WM = scri.WaveformModes(
                    t=np.array(time_set),
                    data=mode_data,
                    ell_min=0,
                    ell_max=lmax,
                    dataType=data_type,
                )
            else:
                raw_data = []
                time_set = []
                for (time_id, time) in time_ids_and_values:
                    time_set.append(time)
                    raw_data.append(scri_data[time_id][scri.DataNames[data_type]][()])
                raw_data = np.array(raw_data)
                time_set = np.array(time_set)

                modes = [(l, m) for l in range(0, lmax + 1) for m in range(-l, l + 1)]
                mode_data = []
                for (l, m) in modes:
                    mode_data.append(
                        raw_data[:, real_idx(l, m)] + 1j * raw_data[:, imag_idx(l, m)]
                    )
                mode_data = np.array(mode_data).T

                WM = scri.WaveformModes(
                    t=np.array(time_set),
                    data=mode_data,
                    ell_min=0,
                    ell_max=lmax,
                    dataType=data_type,
                )
            print("Writing {}...".format(WM.data_type_string), end='')
            scri.SpEC.write_to_h5(WM, output_extension)
            print("Done")

if __name__ == "__main__":
    get_waveforms_from_cce_volume(args.filename, lmax=args.lmax)
