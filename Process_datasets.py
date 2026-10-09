import numpy as np
import torch
from PIL import Image
import h5py
import os
import pyms
from GradDS import reconstruct_Smatrix_from_datacube,check_alignment,depth_section_reconstruction
from py4DSTEM.process.dpc import get_phase_from_CoM


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle (in radians) around a given
    origin.

    Sourced from https://bit.ly/2GhWOvg
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def bin_array(array, factor=2, func=np.mean):
    """Bin the final two dimensions of an array by a given factor"""
    y, x = array.shape[-2:]
    biny, binx = [y // factor, x // factor]
    return func(
        array.reshape(*array.shape[:-2], biny, factor, binx, factor), axis=(-3, -1)
    )


def wavev(E):
    """Calculates the relativistically corrected wavenumber kprint(ny,nx)0 (reciprocal of
    the wavelength) for an electron of energy eV. See Eq. (2.5) in Kirkland's
    Advanced Computing in electron microscopy"""
    # Planck's constant times speed of light in eV Angstrom
    hc = 1.23984193e4
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return np.sqrt(E * (E + 2 * m0c2)) / hc


def aberration(q, lam, df=0, cs=0, c5=0):
    """calculates the aberration function chi as a function of
    reciprocal space extent q for an electron with wavelength lam.

    Parameters
    ----------
    q : number
        reciprocal space extent (Inverse angstroms).
    lam : number
        wavelength of electron (Inverse angstroms).
    df : number
        Probe defocus (angstroms).
    cs : number
        Probe spherical aberration (mm).
    c5 : number
        Probe c5 coefficient (mm)."""
    p = lam * q
    chi = df * np.square(p) / 2.0 + cs * 1e7 * np.power(p, 4) / 4.0
    chi += c5 * 1e7 * np.power(p, 6) / 6.0
    return 2 * np.pi * chi / lam


def construct_illum(
    pix_dim,
    real_dim,
    eV,
    app,
    beam_tilt=[0, 0],
    aperture_shift=[0, 0],
    optic_axis=[0, 0],
    df=0,
    cs=0,
    c5=0,
    app_units="mrad",
    qspace=False,
):
    """Makes a probe wave function with pixel dimensions given in pix_dim
    and real_dimensions given by real_dim
    ---------
    pix_dim --- The pixel size of the grid
    real_dim --- The size of the grid in Angstrom
    eV --- The energy of the probe electrons in eV
    app --- The apperture in units specified by app_units
    df --- Probe defocus in A, a negative value indicate overfocus
    cs --- The 3rd order spherical aberration coefficient
    c5 --- The 5rd order spherical aberration coefficient
    app_units --- The units of the aperture size (A^-1 or mrad)
    """
    npiy, npix = pix_dim
    y, x = real_dim
    qsize = (float(npiy) / y, float(npix) / x)
    qx, qy = q_space_array(pix_dim, qsize)
    k = wavev(eV)

    if app_units == "mrad":
        app_ = np.tan(app / 1000.0) * k
    else:
        app_ = app
    probe = np.zeros(pix_dim, dtype=np.complex)

    qarray1 = np.sqrt(
        np.square(qy - beam_tilt[0] - optic_axis[0])
        + np.square(qx - beam_tilt[1] - optic_axis[1])
    )
    qarray2 = np.sqrt(
        np.square(qy - beam_tilt[0] - aperture_shift[0])
        + np.square(qx - beam_tilt[1] - aperture_shift[1])
    )
    probe[qarray2 < app_] = np.exp(
        -1j * aberration(qarray1[qarray2 < app_], 1.0 / k, df, cs, c5)
    )
    if qspace:
        return probe
    probe = np.fft.ifft2(probe)
    probe /= np.sqrt(np.sum(np.square(np.abs(probe))))
    return probe


def q_space_array(size, qsize):
    """Creates a reciprocal space array of pixel size given by
    tuple size with qspace dimensions given by qsize"""
    nopiy, nopix = size
    qpiy, qpix = qsize
    y = np.fft.fftfreq(nopiy) * qpiy
    x = np.fft.fftfreq(nopix) * qpix
    return np.meshgrid(x, y)


def reconstruction_from_h5_set(
    tags, scanno, defocii, roll_amount, diff_dim, binning=1, synthesize_images=True,reconstruct_Smatrix=True,specimen_tilt=[0,0]
):
    print("processing Scan {0}".format(scanno))
    # Data file names
    datasets = ["Scan_{0}/{1}.h5".format(scanno, x) for x in tags]

    f = h5py.File(datasets[0], "r")

    datacube_shape = f["/4DSTEM_experiment/data/datacubes/datacube_0/datacube"].shape

    calibrations = f["4D-STEM_data/metadata/calibration"].attrs
    eV = calibrations["accelerating_voltage"][0]
    app = calibrations["convergence_semiangle_mrad"][0]
    R_pix = calibrations["R_pix_size"]
    scan_rotation = np.deg2rad(float(calibrations["R_to_K_rotation_degrees"]))
    scan_rotation = np.deg2rad(0)
    scan_dim = np.asarray(datacube_shape[-4:-2]) * R_pix
    datacube_shape = list(datacube_shape[:2]) + [
        x // binning for x in datacube_shape[2:]
    ]

    # Initialize datacube array
    datacubes = np.zeros((len(datasets), *datacube_shape), dtype=np.float32)
    for idset, dset in enumerate(datasets):
        f = h5py.File(dset, "r")
        ext = os.path.splitext(dset)[1]
        if ext == ".mat":
            dcube = f["cbed"]
        elif ext == ".hdf5" or ext == ".h5":
            dcube = f["/4DSTEM_experiment/data/datacubes/datacube_0/datacube"]

        if binning > 1:
            datacubes[idset, ...] = bin_array(np.clip(dcube, 0, 1e7), binning)
        else:
            datacubes[idset, ...] = np.clip(dcube, 0, 1e7)
        f.close()

    # Make probe to check alignment of dataset
    diffshape = np.asarray(datacubes.shape[-2:])
    illum = construct_illum(
        diffshape, diffshape / np.asarray(diff_dim), eV, app, qspace=True
    )
    illum = np.fft.fftshift(illum)

    if synthesize_images:
        # Synthesise images (BF, ABF and DF)
        if not os.path.exists("Outputs/Synthesised_images"):
            os.mkdir("Outputs/Synthesised_images")

        detector_ranges = [[0, app / 2], [app / 2, app], [app, 100], [0, app]]
        modes = ["BF", "ABF", "DF", "ICBF","DPCy", "DPCx"]
        from pyms import make_detector

        D = np.stack(
            [
                make_detector(
                    diffshape,
                    diffshape / np.asarray(diff_dim),
                    eV,
                    drange[1],
                    drange[0],
                )
                for drange in detector_ranges
            ]
            + pyms.q_space_array(diffshape, diffshape / np.asarray(diff_dim))
        )
        D = np.fft.fftshift(D, axes=(-2, -1))
        for idset, dset in enumerate(datacubes):
            for det, mode in zip(D, modes):
                syn_image = np.sum(det * dset, axis=(-1, -2))
                fnam = "Outputs/Synthesised_images/Scan_{0}_{1}_{2}.tif".format(
                    scanno, tags[idset], mode
                )
                Image.fromarray(syn_image).save(fnam)

            # Get Centre of mass images back and calculate DPC images
            COMy = np.asarray(Image.open("Outputs/Synthesised_images/Scan_{0}_{1}_{2}.tif".format(
                    scanno, tags[idset], 'DPCy'
                )))
            COMx = np.asarray(Image.open("Outputs/Synthesised_images/Scan_{0}_{1}_{2}.tif".format(
                scanno, tags[idset], 'DPCx'
            )))
            DPC = get_phase_from_CoM(COMy,COMx,np.rad2deg(180-scan_rotation),flip=False)[0]
            fnam = "Outputs/Synthesised_images/Scan_{0}_{1}_{2}.tif".format(
                    scanno, tags[idset], 'DPC'
            )
            Image.fromarray(DPC).save(fnam)
    # Scan positions
    scan_files = [
        "Scan_grids/Scan_grid_Scan_{0}_{1}.dm4_4D_concurrent.h5".format(scanno, x)
        for x in tags
    ]
    scan_coordinates = np.zeros((len(scan_files), *datacube_shape[:2], 2))
    for iscan, scan in enumerate(scan_files):
        scan_coordinates[iscan, ...] = h5py.File(scan, "r")["Scan_grid"]

    # Rotate the scan coordinates to match the orientation of the diffraction patterns
    origin = np.mean(scan_coordinates, axis=(0, 1, 2))
    scan_coordinates = np.moveaxis(
        rotate(origin, np.moveaxis(scan_coordinates, -1, 0), -scan_rotation), 0, -1
    )

    if synthesize_images:
        alignment_fnam = "Outputs/diff_alignment/Scan_{0}.pdf".format(scanno)
        for dcube, tag in zip(datacubes, tags):
            PACBED = np.sum(dcube, axis=(0, 1))
            Image.fromarray(PACBED).save("Outputs/PACBEDS/" + tag + "_PACBED.tif")
            fig, segimg, colovlapimg = check_alignment(
                PACBED, diff_dim, eV, app, diffraction_shift=roll_amount
            )
            alignment_fnam = "Outputs/diff_alignment/{0}".format(tag)
            Image.fromarray(segimg).save(alignment_fnam + "_segimg.tif")
            #Image.fromarray(colovlapimg).save(alignment_fnam + "_colovlapimg.tif")
            fig.savefig(alignment_fnam + ".pdf")

    fnam = "Outputs/Scan_{0}_Smatrix_{1}.h5".format(scanno, tags[0])

    if not os.path.exists(fnam) or reconstruct_Smatrix:
        Smatrix, beams_index, smatrix_dimensions, Loss = reconstruct_Smatrix_from_datacube(
            datacubes,
            np.concatenate((scan_dim, diff_dim)),
            defocii,
            app,
            eV,
            mu=1.0,
            nchunks=5,
            niterations=10,
            dtype=np.complex64,
            scan_coordinates=scan_coordinates,
            stream_datacube=True,
            diffraction_shift=roll_amount,
            deviceNum=0,
        )
    
        f = h5py.File(fnam, "w")
        f.create_dataset("Smatrix", shape=Smatrix.shape, data=Smatrix, dtype=Smatrix.dtype)
        for i, bdex in enumerate(beams_index):
            f.create_dataset(
                "beams_index_{0}".format(i), shape=bdex.shape, data=bdex, dtype=bdex.dtype
            )
        smatrix_dimensions = np.asarray(smatrix_dimensions)
        f.create_dataset(
            "Dimensions",
            shape=smatrix_dimensions.shape,
            data=smatrix_dimensions,
            dtype=smatrix_dimensions.dtype,
        )
        Loss = np.asarray(Loss)
        f.create_dataset("Loss", shape=Loss.shape, data=Loss, dtype=Loss.dtype)
        f.close()
    else:
        f = h5py.File(fnam,'r')
        Smatrix = f['Smatrix'][:]
        beams_index = [f["beams_index_{0}".format(i)][:] for i in range(2)]
        smatrix_dimensions = f['Dimensions'][:]
        f.close()


    t = np.arange(-600, 600, 10)

    EW = depth_section_reconstruction(
        t, np.asarray(beams_index).T, 1 / pyms.wavev(eV), Smatrix, smatrix_dimensions,diffraction_shift=roll_amount,device=torch.device('cpu'),optic_axis=specimen_tilt
    )

    phase = np.angle(EW)
    fnam = "Outputs/Optical_section_{0}_Smatrix_{1}".format(scanno, tags[0])
    pyms.utils.stack_to_animated_gif(phase, fnam+'.gif')
    f = h5py.File(
       fnam+'.h5' , "w"
    )
    f.create_dataset("Phase", shape=phase.shape, dtype=phase.dtype, data=phase)
    f.close()



if __name__ == "__main__":
    if not os.path.exists("Outputs"):
        os.mkdir("Outputs")
    if not os.path.exists("Outputs/diff_alignment"):
        os.mkdir("Outputs/diff_alignment")

    # Experimental parameters
    # Defocus correction (both for microscope and to convert nm to Angstrom)
    df_corr = 1.24 * 10

    # Binning to speed up reconstructions
    binning = 1

    #Synthesise images etc from 4D-STEM datasets
    synthesize_images = True
    reconstruct_Smatrix = False


    # Scan 10
    nn = [0, 1, 2]
    scanno = 10
    defocii = np.asarray([135.26, 141.81, 146.87, 151.13])[nn] - 135
    center = np.asarray([63, 63])
    diff_dim = [3.61, 3.61]
    specimen_tilt = [1.66,0.55]
    tags = np.asarray(["135", "141", "146", "151"])[nn]
    roll_amount = [2,2]
    reconstruction_from_h5_set(
        tags, scanno, defocii * df_corr, roll_amount,diff_dim, binning,synthesize_images=synthesize_images,specimen_tilt=specimen_tilt,reconstruct_Smatrix=reconstruct_Smatrix
    )
