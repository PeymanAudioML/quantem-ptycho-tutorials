import numpy as np
from tqdm import tqdm
import cupy as cp
import h5py


def compare_images(d1, d2, arcs=8, block=False, renorm=True):
    """Superpose two images using wedges of each image and color."""
    import matplotlib.pyplot as plt

    def renormalize(img):
        min_ = np.amin(img)
        max_ = np.amax(img)
        return (img - min_) / (max_ - min_)

    shape = d1.shape
    fig, ax = plt.subplots(nrows=2)
    img1 = np.zeros(shape, dtype=np.uint8)

    if renorm:
        d1_ = renormalize(d1)*255
        d2_ = renormalize(d2)*255
    else:
        d1_ = d1
        d2_ = d2

    yy = np.broadcast_to(np.fft.fftfreq(d1.shape[0])[:, None], d1.shape[-2:])
    xx = np.broadcast_to(np.fft.fftfreq(d1.shape[1])[None, :], d1.shape[-2:])
    angle = np.fft.fftshift(np.arctan2(yy, xx) - np.pi / arcs / 3 * 2)
    mask = np.sin(angle / 2 * arcs) > 0

    img1[mask] = d1_[mask].astype(np.uint8)
    img1[np.logical_not(mask)] = d2_[np.logical_not(mask)].astype(np.uint8)
    ax[0].imshow(img1)
    img2 = np.zeros(shape + (3,), dtype=np.uint8)
    img2[..., 0] = (renormalize(d1_)*255).astype(np.uint8)
    img2[..., 2] = (renormalize(d2_)*255).astype(np.uint8)
    ax[1].imshow(img2)
    return fig,img1,img2

def check_alignment(datacube,Smatrix_dimensions,eV,app,diffraction_shift=[0,0]):
    """
    Generate alignment checking figure.

    To check that the provided datacube, diffraction dimensions (last two entries
    of Smatrix_dimensions), probe information (ev and app) and diffractions shift
    are consistent with the datacube this function will generate a figure that
    will overlap the reciprocal space illumination with the scan position averaged
    diffraction convergent beam electron diffraction (PACBED) pattern.
    """
    pix = datacube.shape
    PACBED = np.sum(datacube,axis=tuple(np.arange(datacube.ndim-2,dtype=np.int)))
    aperture_function = construct_illum(
        pix[-2:],
        np.asarray(pix[-2:]) / np.asarray(Smatrix_dimensions[-2:]),
        eV,
        app,
        qspace=True,
        optic_axis=diffraction_shift,
        tilt_units='pixels'
    )
    return compare_images(PACBED,np.abs(np.fft.ifftshift(aperture_function)))

def read_in_scattering_matrix(fnam):
    """Read in a scattering matrix with necessary auxiliary information in hdf5 format"""
    f = h5py.File(fnam, "r")
    scattering_matrix = f["Smatrix"]
    beams_index = [f["beams_index_{0}".format(i)] for i in range(2)]
    dimensions = f["Dimensions"]
    return scattering_matrix, beams_index, dimensions


def output_scattering_matrix(fnam, Smatrix, beams_index, smatrix_dimensions, Loss=None):
    """Output a scattering matrix with necessary auxiliary information in hdf5 format"""
    f = h5py.File(fnam, "w")
    f.create_dataset("Smatrix", shape=Smatrix.shape, data=Smatrix, dtype=Smatrix.dtype)
    for i, bdex in enumerate(beams_index):
        f.create_dataset(
            "beams_index_{0}".format(i), shape=bdex.shape, data=bdex, dtype=bdex.dtype
        )
    smatrix_dimensions_ = np.asarray(smatrix_dimensions)
    f.create_dataset(
        "Dimensions",
        shape=smatrix_dimensions_.shape,
        data=smatrix_dimensions_,
        dtype=smatrix_dimensions_.dtype,
    )
    if Loss is not None:
        Loss_ = np.asarray(Loss)
        f.create_dataset("Loss", shape=Loss_.shape, data=Loss_, dtype=Loss_.dtype)
    f.close()


def std_clip(array, nsigma, new=None):
    """Clip or remove array values more than nsigma standard deviations from the
    mean from the array"""
    mean = np.mean(array)
    sigma = np.std(array)
    if new is None:
        return np.clip(array, mean - nsigma * sigma, mean + nsigma * sigma)
    else:
        array[array > mean + nsigma * sigma] = new[1]
        array[array < mean - nsigma * sigma] = new[0]
        return array


class aberration:
    """A class describing electron lens aberrations."""

    def __init__(self, Krivanek, Haider, Description, amplitude, angle, n, m):
        """
        Intialize the lens aberration object.

        Parameters
        ----------
        Krivanek : string
            A string describing the aberration coefficient in Krivanek notation
            (C_mn)
        Haider : string
            A string describing the aberration coefficient in Haider notation
            (ie. A1, A2, B2)
        Description :
            A string describing the colloqiual name of the aberration ie. 2-fold
            astig.
        amplitude :
            The amplitude of the aberration in Angstrom
        angle :
            The angle of the aberration in radians
        n :
            The principle aberration order
        m :
            The rotational order of the aberration.
        """
        self.Krivanek = Krivanek
        self.Haider = Haider
        self.Description = Description
        self.amplitude = amplitude
        self.m = m
        self.n = n
        if m > 0:
            self.angle = angle
        else:
            self.angle = 0

    def __str__(self):
        """Return a string describing the aberration."""
        if self.m > 0:
            return (
                "{0:17s} ({1:2s}) -- {2:3s} = {3:9.2e} \u00E5 \u03B8 = "
                + "{4:4d}\u00B0 "
            ).format(
                self.Description,
                self.Haider,
                self.Krivanek,
                self.amplitude,
                int(np.rad2deg(self.angle)),
            )
        else:
            return " {0:17s} ({1:2s}) -- {2:3s} = {3:9.2e} \u00E5".format(
                self.Description, self.Haider, self.Krivanek, self.amplitude
            )


def wavev(E):
    """Calculates the relativistically corrected wavenumber k0 (reciprocal of
    the wavelength) for an electron of energy eV. See Eq. (2.5) in Kirkland's
    Advanced Computing in electron microscopy"""
    # Planck's constant times speed of light in eV Angstrom
    hc = 1.23984193e4
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return np.sqrt(E * (E + 2 * m0c2)) / hc


def chi(q, qphi, lam, df=0.0, aberrations=[]):
    """
    Calculate the aberration function, chi.

    Parameters
    ----------
    q : float or array_like
        Reciprocal space extent (Inverse angstroms).
    qphi : float or array_like
        Azimuth of grid in radians
    lam : float
        Wavelength of electron (Inverse angstroms).
    Keyword arguments
    -----------------
    df : float, optional
        Defocus in Angstrom
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated contrast transfer function.
    """
    qlam = q * lam
    chi_ = qlam ** 2 / 2 * df
    for ab in aberrations:
        chi_ += (
            qlam ** (ab.n + 1)
            * float(ab.amplitude)
            / (ab.n + 1)
            * np.cos(ab.m * (qphi - float(ab.angle)))
        )
    return 2 * np.pi * chi_ / lam


def convert_tilt_angles(tilt, tilt_units, rsize, eV, invA_out=False):
    """
    Convert  tilt to pixel or inverse Angstroms units regardless of input units.

    Input units can be mrad, pixels or inverse Angstrom

    Parameters
    ----------
    tilt : array_like
        Tilt in units of mrad, pixels or inverse Angstrom
    tilt_units : string
        Units of specimen and beam tilt, can be 'mrad','pixels' or 'invA'
    rsize : (2,) array_like
        The size of the grid in Angstrom
    eV : float
        Probe energy in electron volts
    Keyword arguments
    -----------------
    invA_out : bool
        Pass True if inverse Angstrom units are desired.
    """
    # If units of the tilt are given in mrad or pixels, convert to inverse Angstrom
    if tilt_units == "mrad":
        k = wavev(eV)
        tilt_ = np.asarray(tilt) * 1e-3 * k
    elif tilt_units == 'pixels':
        tilt_ = np.asarray(tilt) / np.asarray(rsize)
    else:
        tilt_ = tilt

    # If inverse Angstroms are requested our work here is done
    if invA_out:
        return tilt_

    # Convert inverse Angstrom to pixel coordinates, this will be rounded
    # to the nearest pixel
    if tilt_units != "pixels":
        tilt_ = np.round(tilt_ * rsize[:2]).astype(int)
    return tilt_


def construct_illum(
    pix_dim,
    real_dim,
    eV,
    app,
    qspace=False,
    optic_axis=[0, 0],
    aperture_shift=[0, 0],
    tilt_units="mrad",
    df=0,
    aberrations=[],
    q=None,
    app_units="mrad",
):
    """
    Make an electron lens contrast transfer function.

    Parameters
    ---------
    pix_dim --- The pixel size of the grid
    real_dim --- The size of the grid in Angstrom
    eV --- The energy of the probe electrons in eV
    app --- The aperture in units specified by app_units, pass app = None for
            no aperture
    qspace ---- Set to True to get probe forming aperture in diffraction space
    optic_axis --- allows the user to specify a different optic
                    axis
    aperture_shift --- Shift of the objective aperture relative
                        to the center of the array
    tilt_units : string
        Units of the optic_axis or aperture_shift values
    df --- Probe defocus in A, a negative value indicate overfocus
    aberrations --- List containing instances of class aberration
    q --- reciprocal space array, allows the user to reduce computation
        time somewhat
    app_units --- The units of the aperture size (A^-1 or mrad)
    """
    # Make reciprocal space array
    if q is None:
        q = q_space_array(pix_dim, np.asarray(pix_dim)/np.asarray(real_dim[:2]))

    # Get  electron wave number (inverse of wavelength)
    k = wavev(eV)

    # Convert tilts to units of inverse Angstrom
    optic_axis_ = convert_tilt_angles(
        optic_axis, tilt_units, real_dim, eV, invA_out=True
    )
    aperture_shift_ = convert_tilt_angles(
        aperture_shift, tilt_units, real_dim, eV, invA_out=True
    )

    if app is None:
        app_ = np.amax(np.abs(q))
    else:
        # Get aperture size in units of inverse Angstrom
        app_ = convert_tilt_angles(app, app_units, real_dim, eV, invA_out=True)

    # Initialize the array to contain the CTF
    CTF = np.zeros(pix_dim, dtype=np.complex)

    # Calculate the magnitude of the reciprocal lattice grid
    # qarray1 accounts for a shift of the optic axis
    qarray1 = np.sqrt(
        np.square(q[0] - optic_axis_[0]) + np.square(q[1] - optic_axis_[1])
    )

    # qarray2 accounts for a shift of both the optic axis and the aperture
    qarray2 = np.square(q[0] - optic_axis_[0] - aperture_shift_[0]) + np.square(
        q[1] - optic_axis_[1] - aperture_shift_[1]
    )

    # Calculate azimuth of reciprocal space array in case it is required for
    # aberrations
    qphi = np.arctan2(q[0] - optic_axis_[0], q[1] - optic_axis_[1])

    # Only calculate CTF for region within the aperture
    mask = qarray2 <= app_ ** 2
    CTF[mask] = np.exp(-1j * chi(qarray1[mask], qphi[mask], 1.0 / k, df, aberrations))

    # Normalize the STEM probe so that its sum-squared intensity is unity
    probe = (
        CTF * np.sqrt(np.prod(pix_dim)) / np.sqrt(np.sum(np.square(np.abs(CTF))))
    )

    # Return real or diffraction space probe depending on user preference
    if not qspace:
        return np.fft.ifft2(probe)
    return probe


def q_space_array(size, qsize):
    """Creates a reciprocal space array of pixel size given by
    tuple size with qspace dimensions given by qsize"""
    nopiy, nopix = size
    qpiy, qpix = qsize
    y = np.fft.fftfreq(nopiy) * qpiy
    x = np.fft.fftfreq(nopix) * qpix
    return np.meshgrid(x, y)[::-1]


def reconstruct_Smatrix_from_datacube(
    datacube,
    dimensions,
    defocii,
    app,
    eV,
    mu=1.0,
    niterations=5,
    scan_offsets=None,
    nchunks=30,
    padding=1.0,
    dtype=np.complex64,
    diffraction_shift=[0,0],
    stream_datacube=None,
    scan_coordinates=None,
    report_memory=True,
    report_loss=False,
    deviceNum=0,
):
    """reconstruct an Smatrix from a series of datacubes.

    Parameters
    ----------
    datacube : array_like (ndf x ny x nx x ky x kx,) 
        Input datacubes to reconstruction
    dimensions : (4,)
        First two entries are the dimensions of the scan array in Angstrom and
        the second two entries are the dimensions of the diffraction pattern in
        inverse Angstrom
    defocii : array_like (ndf,)
        The defocii at which the datacubes are recorded
    app : float
        The probe-forming aperture in mrad
    eV : float
        Probe accelerating voltage in electron volts

    Keyword arguments
    -----------------
    mu : float
        Amplitude flow update step size
    niterations : int
        Number of iterations of the amplitude flow algorithm
    scan_offsets : (ndf,) float
        constant "offsets" for each scan, can be used to for a through-focal
        series with specimen drift
    nchunks : int
        Number of diffraction patterns to process in parrallel in the algorithm
    padding : float
        Zero padding of the reconstruction array to avoid wrap-around errors
        due to periodic boundary conditions. Typically not necessary.
    dtype : torch.dtype object
        Datatype of the reconstructed scattering matrix. complex64 is usually
        best to preserve memory
    diffraction_shift : (2,) int
        Shift of the diffraction pattern coordinates typically due to misalignment
        in the experiment
    stream_datacube : bool
        Option to store the datacubes in CPU RAM and stream them to the GPU as
        required
    scan_coordinates : (ndf,ny,nx,2)
        User provided scan coordinates, units of pixels ie. should run from
        0 - ny and 0-nx for an undistorted scan.
    report_memory : bool
        Reports memory usage of scattering matrix reconstruction
    report_loss : bool
        Reports the L2 norm for difference between the input data and the current
        model S-matrix
    deviceNum : int
        For multi-GPU systems choose which GPU to use

    Returns
    -------
    Smatrix : complex (b,Y,X) array_like
        The scattering matrix
    beam_index : (2,) tuple 
        Tuple containing the diffraction space pixel coordinates of each of the
        beams correcpsonding to the scattering matrix entries
    smatrix_dimensions : (2,) tuple
        Real space size (in Angstrom) for each scatering matrix
        entry
    Loss : (niterations,) array_like or None
        The L2 error for the difference between the input diffraction patterns
        and the diffraction patterns predicted by the reconstruction scattering
        matrix if report_loss = True or None otherwise
    """

    # Inititialize the scattering matrix to be the identity matrix
    Smatrix, beams_index, smatrix_dimensions, M = initialize_Smatrix(
        app,
        eV,
        datacube.shape[1:],
        dimensions,
        beam_shift=diffraction_shift,
        padding=padding,
        dtype=dtype,
    )
    if report_memory:
        print(
            "Smatrix is on a {0} x {1} x {2} pixel grid measuring {3} x {4} Angstrom, for a total size of {5} GB,".format(
                *Smatrix.shape, *smatrix_dimensions, Smatrix.nbytes * 1e-9
            )
        )
    nbeams = beams_index[0].shape[0]

    # Total number of scan positions for all focal values
    # If scan_coordinates is passed, values of this array less than 0 signify
    # that that coordinate is not to be used in the reconstruction, so calculate
    # the number of scan coordinates that will actually be used
    # If scan_coordinates is passed, values of this array less than 0 signify
    # that that coordinate is not to be used in the reconstruction, so calculate
    # the number of scan coordinates that will actually be used
    # If no scan_coordinates are passed then generate them
    if scan_coordinates is None:
        scan_shape = datacube.shape[1:3]
        scan_coordinates = np.broadcast_to(
            np.moveaxis(np.mgrid[0 : scan_shape[0], 0 : scan_shape[1]], 0, 2),
            (len(defocii), *scan_shape, 2),
        )
        if scan_offsets is not None:
            scan_coordinates = scan_coordinates - scan_offsets[:, None, None, :]

    nscan = np.sum(np.all(scan_coordinates >= 0, axis=3))

    # Now construct the illumination vector
    illumination = np.zeros((nscan, nbeams), dtype=dtype)
    j = 0
    scan_coord = np.zeros((nscan, 2), dtype=np.float)

    lam = 1 / wavev(eV)

    k = [
        (np.fft.fftfreq(Smatrix.shape[i - 2], 1 / Smatrix.shape[i - 2]) - diffraction_shift[i])
        / smatrix_dimensions[i]
        for i in range(2)
    ]
    
    ksqr = (k[0] ** 2)[:, np.newaxis] + (k[1] ** 2)[np.newaxis, :]

    print("Setting up illumination vector...")
    norm = np.sqrt(1 / nbeams)
    for i in tqdm(range(np.prod(datacube.shape[:3]))):

        # Get defocus, and scan indices for this diffraction pattern
        idf, y, x = np.unravel_index(i, datacube.shape[:3])

        # Get scan coordinate as fraction of grid
        scan_coord_ = (
            scan_coordinates[idf, y, x, :]
            / datacube.shape[1:3]
            * dimensions[:2]
            / smatrix_dimensions
        )

        if np.any(scan_coord_ < 0):
            continue

        # Adjust scan coordinates to fractions of the scan window rather than
        # the diffraction coordiantes
        scan_coord[j, :] = scan_coord_

        # Calculate probe phase for this defocus and scan coordinate
        # k = np.asarray(ky[beams_index[0]//M[0]],kx[beams_index[1]//M[1]])
        chi = -ksqr[beams_index[0], beams_index[1]] * defocii[idf] * lam * np.pi
        shift = (
            -2
            * np.pi
            * (
                k[0][beams_index[0]] * scan_coord[j, 0] * smatrix_dimensions[0]
                + k[1][beams_index[1]] * scan_coord[j, 1] * smatrix_dimensions[1]
            )
        )
        illumination[j, :] = np.exp(1j * (shift + chi)) * norm
        j += 1

    # Convert scan coordinates to pixel units and cast to integer data type to
    # do cropping on the array
    scan_coord = np.round(scan_coord * np.asarray(Smatrix.shape[-2:])[None, :]).astype(
        np.int
    )

    # Get mask of coordinates that will contribute to the reconstruction
    mask = np.all(scan_coordinates >= 0, axis=3)

    # zero values < 0 in datacube
    datacube[datacube < 0] = 0

    # Mean filter removes values more than 3 times
    mean_counts = np.amax(
        std_clip(
            np.sum(np.nan_to_num(datacube[mask, ...]), axis=(-2, -1)), 3, new=[0.0, 0.0]
        )
    )

    # Get amplitude of diffraction pattern
    datacube = datacube[mask, ...]
    datacube = np.fft.fftshift(np.sqrt(datacube), axes=[-2, -1])

    # Normalize each diffraction pattern to have approximately unitary
    # total amplitude
    datacube /= np.sqrt(mean_counts)

    # Clear GPU memory and try to make an assessment of whether to stream the
    # datacube or not
    cp.cuda.Device(deviceNum).use()
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    if stream_datacube is None:
        stream_datacube = (
            mempool.total_bytes() - mempool.used_bytes() > 2 * datacube.nbytes
        )

    Smatrix = cp.array(Smatrix)
    illumination = cp.array(illumination)
    scan_coord = cp.array(scan_coord)

    

    # Option to stream the datacube to the GPU as diffraction patterns are used
    # in the reconstruction (necessary if dataset size is > GPU memory) or
    # perform a single transfer of data to the GPU.
    if not stream_datacube:
        datacube = cp.array(datacube)

    if report_memory:
        print("Device has {0} MB total".format(int(mempool.total_bytes() * 1e-6)))
        print(
            "A single copy of the S-matrix requires {0} MB total".format(
                int(Smatrix.nbytes * 1e-6)
            )
        )
        print(
            "The illumination vector requires {0} MB total".format(
                int(illumination.nbytes * 1e-6)
            )
        )
        print(
            "The chunks of datacube streamed to GPU will be {0} MB each".format(
                int(nchunks * (datacube[0].nbytes) * 1e-6)
            )
        )

    # Indices of the cropping window of the scattering matrix
    crop_window = [
        [-Smatrix.shape[i - 2] // M[i] // 2, +Smatrix.shape[i - 2] // M[i] // 2]
        for i in range(2)
    ]

    if report_loss:
        # Initialize Loss metric
        Loss = np.zeros((niterations+1,))
        # Calculate initial loss
        aTZ = cp.asnumpy(AT(
            Smatrix,
            illumination[:1, :],
            scan_coord[:1, :],
            crop_window,
        ))
        Loss[0] = np.sum((np.abs(datacube)**2-np.abs(aTZ)**2)**2)
    else:
        Loss = None

    # Smatrix reconstruction via truncated amplitude flow
    for i in tqdm(range(niterations), desc="Iteration"):

        for ichunk in tqdm(
            range(int(np.ceil(nscan / nchunks))), desc="Diffraction pattern"
        ):

            # Choose
            chunki = ichunk * nchunks
            chunkf = min((ichunk + 1) * nchunks, nscan)

            # Forward projection step (illumination to diffraction patterns)
            aTZ = AT(
                Smatrix,
                illumination[chunki:chunkf, :],
                scan_coord[chunki:chunkf, :],
                crop_window,
            )

            # Transfer requisite diffraction patterns to the GPU if streaming
            # or just choose subset of data if a single transfer of data was used
            if stream_datacube:
                amp = cp.array(datacube[chunki:chunkf, :])
            else:
                amp = datacube[chunki:chunkf, :]

            if report_loss:
                Loss[i+1] += cp.sum((cp.abs(amp)**2 - cp.abs(aTZ) ** 2)**2)
            
            aTZ -= cp.nan_to_num(amp * aTZ / cp.abs(aTZ))

            amp = None

            AinvT(
                aTZ,
                illumination[chunki:chunkf, :],
                scan_coord[chunki:chunkf, :],
                crop_window,
                Smatrix,
                mu / nscan,
                qspace_in=True,
            )
            
    if report_loss:
        Loss /= np.sum(datacube**4)

    return cp.asnumpy(Smatrix), beams_index, smatrix_dimensions, Loss


def crop_window_to_flattened_indices(indices, shape):
    # initialize array to hold flattened index in

    return (
        indices[-1][None, :] % shape[-1]
        + (indices[-2][:, None] % shape[-2]) * shape[-1]
    ).ravel()


def AT(Smatrix, illum, scan_posn, crop_window, beam_index=None, qspace_out=True):
    """Forward Smatrix product to calculate exit surface wave, requires the complex
    smatrix, the beam_index, which maps the beams of the Smatrix to coordinates
    in reciprocal space and the illumination. illumination_indexed tells the
    function the form of the illumination, if True then illumination is a 1D
    array containing the complex coefficients of the illumination for the different
    beams. """

    nbeams = illum.shape[0]

    # Calculate the indices for blocks of the S-matrix that will
    # contribute to the forward projection
    # crop_wind = [Smatrix.shape[i - 2] // crop_window[i] for i in range(2)]

    crop_ = [cp.arange(crop_window[i][0], crop_window[i][1]) for i in range(2)]
    cropshape = [crop_window[i][1] - crop_window[i][0] for i in range(2)]

    exit_wave = cp.zeros((illum.shape[0], np.prod(cropshape)), dtype=Smatrix.dtype)

    # Save Smatrix shape for later recall
    Smatrix_shape = Smatrix.shape

    # Flatten final two dimensions of S-matrix
    nn = np.prod(Smatrix.shape[-2:])
    Smatrix = Smatrix.reshape((Smatrix.shape[0], nn))

    # Calculate matrix-vector product of Smatrix with illumination vector
    # Smatrix format is nbeams (input) x Ry x Rx (real space coordinates)
    # Illumination format is nscan x nbeams
    # Smatrix must be broadcast over the scan postions of the illumination
    # and the illumination vector must be broadcast over the diffraction
    # coordinates
    for i in range(illum.shape[0]):
        # Numpy indexing doesn't work quite the way I want it to so the cropping
        # has to be done using the indices of the flattened array
        crop = (
            (crop_[0] + scan_posn[i, 0]) % Smatrix_shape[1],
            (crop_[1] + scan_posn[i, 1]) % Smatrix_shape[2],
        )

        ind = crop_window_to_flattened_indices(crop, Smatrix_shape[-2:])
        exit_wave[i, :] = cp.matmul(illum[i], Smatrix[:, ind])

    # Reshape into square grid
    exit_wave = exit_wave.reshape((nbeams, *cropshape))

    # Return Smatrix to original shape
    Smatrix = Smatrix.reshape(Smatrix_shape)

    # Fourier transform to the diffraction plane if requested
    if qspace_out:
        exit_wave = cp.fft.fft2(exit_wave, norm="ortho")

    return exit_wave


def AinvT(
    exit_wave,
    illum,
    scan_posn,
    crop_window,
    Smatrix,
    weighting,
    beam_index=None,
    qspace_in=True,
):
    """Back project the exit wave to the scattering matrix"""
    exit_wave = cp.asarray(exit_wave, dtype=Smatrix.dtype)
    Smatrix_shape = Smatrix.shape
    if qspace_in:
        # We require the exit-wave to be in real space so if it's not, inverse
        # Fourier transform it
        exit_wave = cp.fft.ifft2(exit_wave, norm="ortho")

    # Calculate the indices for blocks of the S-matrix that will
    # contribute to the forward projection
    # crop_wind = [Smatrix_shape[i - 2] // crop_window[i] for i in range(2)]
    crop_ = [cp.arange(crop_window[i][0], crop_window[i][1]) for i in range(2)]
    # cropshape = [crop_window[i][1]-crop_window[i][0] for i in range(2)]

    # crop_ = [cp.arange(-crop_wind[i] // 2, crop_wind[i] // 2) for i in range(2)]

    # Flatten final two indices of exit_wave
    exwave = exit_wave.reshape((exit_wave.shape[0], np.prod(exit_wave.shape[-2:])))
    Smatrix = Smatrix.reshape((Smatrix_shape[0], np.prod(Smatrix_shape[-2:])))

    # Perform cropping procedure if requested
    # Loop over probe positions
    for i in range(illum.shape[0]):

        # Numpy indexing doesn't work quite the way I want it to so the cropping
        # has to be done using the indices of the flattened array
        crop = (
            (crop_[0] + scan_posn[i, 0]) % Smatrix_shape[-2],
            (crop_[1] + scan_posn[i, 1]) % Smatrix_shape[-1],
        )
        ind = crop_window_to_flattened_indices(crop, Smatrix_shape[-2:])

        Smatrix[:, ind] -= exwave[i, None, :] / illum[i, :, None] * weighting

    Smatrix.reshape(Smatrix_shape)


def initialize_Smatrix(
    app, eV, pix, dimensions, beam_shift=[0, 0], padding=1.0, dtype=np.complex64
):
    """
    Set up a diagonal Scattering matrix as a starting point for iteration.

    Parameters
    ----------
    app : float
        Maximum input angle for the scattering matrix (ie. a probe forming
        aperture)
    eV : float
        Probe accelerating voltage
    pix : int (4,) array_like
        pixel dimensions of the input datacube (ny,nx,nky,nkx)
    dimensions :
        datacube physical dimensions. The first two array entries are the scan
        field of view in Angstrom and the second two entries are the diffraction
        pattern size in inverse Angstrom.
    beam_shift : array_like (2,) int, optional
        Shift (in pixels) of the illumination, usually to match diffraction
        shift due to experimental misalignment
    padding : float, optional
        Padding of the reconstruction grid, can usually just be 1.
    dtype : np.dtype object or string, optional
        Datatype of the scattering matrix
    Returns
    -------
    Smatrix : (n,Y,X) complex array_like
        The initailized scattering matrix
    beam_index : (2,n) int array_like
        Pixel coordinates of each of the n input beams for the scattering matrix
    Smatrix_dimensions : (2,) array_like
        Real space size of the S-matrix grid
    M : int
        Pixel dimensions of the Smatrix will be M times larger than the diffraction
        pattern size of the 4D-dataset, ie. (Y,X) = (M*nky,M*nkx)
    """

    scan_dim = dimensions[:2]

    # Work out the pixel size of the scattering matrix.
    # Sampling of the Smatrix grid set by the maximum spatial frequency
    # in the diffraction pattern
    sampling = 1 / np.asarray(dimensions[-2:])

    # Work out how large the pixel real space dimensions of the scattering
    # matrix will be, this is so the framework is general with respect to scan
    # size and diffraction space of the datacube.
    # Total number of pixels in the reconstructed S-matrix is the total
    # size of the scan divided by the sampling, rounded up so that the
    # plane waves will be periodic
    Smatrix_size = (
        np.ceil(scan_dim / sampling * padding / pix[-2:]) * pix[-2:]
    ).astype(np.int)

    # M is the integer ratio between the full S-matrix grid and the
    # diffraction pattern pixel grid
    M = Smatrix_size // pix[-2:]

    # Get real space size of scattering matrix
    Smatrix_dimensions = Smatrix_size * sampling

    # Create an aperture function that will be used to determine which input
    # beams of the S-matrix will be used
    aperture_function = construct_illum(
        Smatrix_size,
        np.asarray(Smatrix_size) / np.asarray(dimensions[-2:]),
        eV,
        app,
        qspace=True,
        optic_axis=beam_shift,
        tilt_units='pixels'
    )

    # Construct the diffraction space coordinate mesh
    kx, ky = np.meshgrid(
        *[
            np.asarray(
                np.fft.fftfreq(Smatrix_size[i], 1 / Smatrix_size[i]), dtype=np.int
            )
            for i in [-1, -2]
        ]
    )

    # Determine which beams will be included in the input to the scattering matrix
    beams = np.logical_and(
        np.logical_and(np.abs(aperture_function) > 0, kx % M[0] == 0), (ky % M[1] == 0)
    )

    # Get the indices of those beams, using the numpy nonzero function which will
    # return a tuple containing two arrays
    beam_index = np.nonzero(beams)

    # Since the beam index will used to index an array of size rpixsize, make
    # any indices > #pixels/2 negative so that the indices will correctly carry
    # over using Python's handy indexing syntax
    for i in range(2):
        beam_index[i][beam_index[i] > Smatrix_size[i] // 2] -= Smatrix_size[i]

    # Get number of beams
    nbeams = beam_index[0].shape[0]

    # Initialize scattering matrix. It will be number of beams x scan dimensions
    Smatrix = np.zeros((nbeams, *Smatrix_size), dtype=dtype)

    # Set all included beams to be unity the inverse fourier transform
    # operation will divide the sum square intensity by the number of pixels
    #  so prepare for this
    val = np.sqrt(np.prod(M * Smatrix.shape[-2:]))
    for i in range(nbeams):
        Smatrix[i, beam_index[0][i], beam_index[1][i]] = val

    # Store Smatrix in real space
    Smatrix = np.fft.ifftn(Smatrix, axes=(-2, -1))

    # Return Smatrix and complimentary beam index
    return Smatrix.astype(dtype), beam_index, Smatrix_dimensions, M


if __name__ == "__main__":
    pass
