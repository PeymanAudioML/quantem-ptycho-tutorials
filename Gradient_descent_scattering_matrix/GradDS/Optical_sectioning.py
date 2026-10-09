"""A module to perform the optical sectioning reconstruction of the scattering matrix."""
from copy import deepcopy
import torch as th
import numpy as np
import tqdm

re = np.s_[..., 0]
im = np.s_[..., 1]



def complex_mul_conj(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2 or b.shape[-1] != 2:
        raise RuntimeWarning(
            'taking complex_mul of non-complex tensor! a.shape ' + str(a.shape) + 'b.shape ' + str(b.shape))
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = -b[im]
    real = are * bre - aim * bim
    imag = are * bim + aim * bre
    return th.stack([real, imag], -1)

def complex_abs(a: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2:
        raise RuntimeWarning('taking complex_abs of non-complex tensor!')
    return th.sqrt(a[re] ** 2 + a[im] ** 2)

def roll_n(X, axis, n):
    a = axis % X.ndim
    f_idx = tuple(
        slice(None, None, None) if i != a else slice(0, n, None) for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != a else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return th.cat([back, front], a)


def fourier_coordinates_2D(N, dx=[1.0, 1.0], centered=True):
    qxx = fftfreq(N[1], dx[1])
    qyy = fftfreq(N[0], dx[0])
    if centered:
        qxx += 0.5 / N[1] / dx[1]
        qyy += 0.5 / N[0] / dx[0]
    qx, qy = np.meshgrid(qxx, qyy)
    q = np.array([qy, qx]).astype(np.float32)
    return q


def complex_expi(x: th.Tensor) -> th.Tensor:
    real = th.cos(x)
    imag = th.sin(x)
    return th.stack([real, imag], -1)


def cangle(x: th.Tensor, deg=False) -> th.Tensor:
    real = th.atan2(x[im], x[re])
    if deg:
        real *= 180 / np.pi
    return real


def circshift(x, axes, shifts):
    real, imag = th.unbind(x, -1)
    for ax, sh in zip(axes, shifts):
        real = roll_n(real, axis=ax, n=sh)
        imag = roll_n(imag, axis=ax, n=sh)
    return th.stack((real, imag), -1)


def complex_mul(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2 or b.shape[-1] != 2:
        raise RuntimeWarning(
            "taking complex_mul of non-complex tensor! a.shape "
            + str(a.shape)
            + "b.shape "
            + str(b.shape)
        )
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = b[im]
    real = are * bre - aim * bim
    imag = are * bim + aim * bre
    return th.stack([real, imag], -1)


def cx_from_numpy(x: np.array) -> th.Tensor:
    if "complex" in str(x.dtype):
        out = th.zeros(x.shape + (2,))
        out[re] = th.from_numpy(x.real)
        out[im] = th.from_numpy(x.imag)
    else:
        if x.shape[-1] != 2:
            out = th.zeros(x.shape + (2,))
            out[re] = th.from_numpy(x.real)
        else:
            out = th.zeros(x.shape + (2,))
            out[re] = th.from_numpy(x[re])
            out[re] = th.from_numpy(x[im])
    return out


def cx_to_numpy(x: th.Tensor) -> np.ndarray:
    """Convert a complex pytorch tensor to a complex numpy array."""

    return x[re].cpu().numpy() + 1j * x[im].cpu().numpy()


def get_device(device_type=None):
    """Initialize device cuda if available, CPU if no cuda is available."""
    if device_type is None and th.cuda.is_available():
        device = th.device("cuda")
    elif device_type is None:
        device = th.device("cpu")
    else:
        device = th.device(device_type)
    return device


def ensure_torch_array(array, dtype=th.float, device=None):
    """
    Ensure that the input array is a pytorch tensor.

    Converts to a pytorch array if input is a numpy array and do nothing if the
    input is a pytorch tensor
    """
    if device is None:
        device = get_device(device)
    if not isinstance(array, th.Tensor):
        if np.iscomplexobj(array):
            return cx_from_numpy(array).type(dtype).to(device)
        else:
            return th.from_numpy(array).type(dtype).to(device)
    else:
        return array.to(device)


def depth_section_reconstruction(
    t, beams, lam, S, Smatrixdimensions, beam_mask=None, phase_mask=None, qspacein=False, device=None,diffraction_shift=[0,0],optic_axis=[0,0]
):
    """
    Optical sectioning reconstruction from scattering matrix.

    Parameters:
    -----------
    t : float, array_like
        Optical depths at which the reconstruction will be created
    beams : integer, array_like (B x 2)
        List of beam coordinates
    lam : float
        Electron wavelength
    S : float, array_like (B x Y x X x 2)
        The scattering matrix input each component of the scattering matrix by
        default should be in real space
    Smatrixdimensions : float (2,) array_like
        Real space dimensions of the scattering matrix

    keyword arguements
    -----------------
    phase_mask : float, array_like, optional
        Phase mask to apply to the beams (for example a vortex probe)
    qspace_in : bool, optional
        If True then the each component of the scattering matrix is taken to
        be in reciprocal space.
    device : optional
        Allows the user to choose the reconstruction device, if None is pass
        then a cuda capable GPU device is chosen if available but cpu if not
    diffraction_shift : (2,) array_like, optional
        Diffraction shift of the 4D-STEM acquisition from which the S-matrix was
        reconstructed. This is the pixel coordinate shift from the center of
        the array for the origin of the diffraction coordinates.
    optic_axis : (2,) array_like, optional
        Optic axis assumed in reconstruction. This can be used to correct 
        somewhat for specimen tilt of crystalline samples
    Returns
    -------
    EW :  float, array_like
        Set of reconstructions at depths t
    """
    device = get_device(device)
    S = ensure_torch_array(S, device=device)
    dtype = S.dtype

    # Get array shape of scattering matrix
    B, Y, X = S.shape[:-1]

    # Fourier transform S-matrix coordinates if necessary
    if qspacein:
        S = S
    else:
        S = th.fft(S, 2, True)

    # Generate Fourier grids
    qy = (
        (th.from_numpy(np.fft.fftfreq(Y, 1 / Y))/Smatrixdimensions[0])
        .type(dtype)
        .to(device)
    )
    qx = (
        (th.from_numpy(np.fft.fftfreq(X , 1 / X))/Smatrixdimensions[1])
        .type(dtype)
        .to(device)
    )

    ksqr = (qy[:,None]**2 + qx[None,:]**2).cpu().numpy()
    import matplotlib.pyplot as plt
    # plt.imshow(ksqr)
    # plt.show()

    usebeammask = beam_mask is not None
    if usebeammask:
        beam_mask_ = np.roll(beam_mask,diffraction_shift,axis=[0,1])
        bmask = np.zeros(B,dtype=np.bool)
    else:
        bmask = np.ones(B,dtype=np.bool)

    # Shift all beams to origin for each component of the scattering
    # matrix and generate phase ramps for each beam
    phasey = th.zeros(B, Y, 1, device=device, dtype=dtype)
    phasex = th.zeros(B, 1, X, device=device, dtype=dtype)
    for ib, beam in enumerate(beams):
        # by, bx = [x.item() - y for x,y in zip(beam,optic_axis)]
        by, bx = [x.item() for x in beam]

        if usebeammask:
            bmask[ib] = beam_mask_[by,bx]
            if not bmask[ib]:
                continue
        # fig,ax = plt.subplots(ncols=2)
        # ax[0].imshow(S[ib,...,0].cpu().numpy())
        # Shift beam to origin
        S[ib][:] = circshift(S[ib][:], [0, 1], [by, bx])
        # ax[1].imshow(S[ib,...,0].cpu().numpy())
        # plt.show()
        
        # Fresnel propagation and paraxial shift of smatrix component
        dxy = [th.tan(qy[by-diffraction_shift[0]] * lam-optic_axis[0]*1e-3), th.tan(qx[bx-diffraction_shift[1]] * lam-optic_axis[1]*1e-3)]
        phasey[ib, ..., 0] = np.pi * lam * qy ** 2 + 2 * np.pi * qy * dxy[0]
        phasex[ib] = np.pi * lam * qx  ** 2 + 2 * np.pi * qx * dxy[1]
    del qy, qx

    EW = th.zeros(len(t), Y, X, 2, dtype=dtype, device=device)
    for i, T in enumerate(tqdm.tqdm(t, desc="Optical section")):
        EW[i] = th.sum(
                complex_mul(
                    complex_mul(S[bmask], complex_expi(T * phasey[bmask])), complex_expi(T * phasex[bmask])
                ),
                axis=0,
            )

    EW = th.ifft(EW,2,True)
    # EW = th.ifft(circshift(EW, [1, 2], optic_axis),2,True)

    return EW[..., 0].cpu().numpy() + 1j * EW[..., 1].cpu().numpy()

def laplacian(array,rsize = [1,1],op=None):
    """Calculate Laplacian of an array using Fourier space laplacian operator."""

    if op is None:
        # Get reciprocal space grid
        qy,qx = [np.fft.fftfreq(x,d=r/x) for x,r in zip(array.shape[-3:-1],rsize) ]

        # Calculate reciprocal space Laplacian operator -4pi^2k^2
        op = -4*np.pi**2*((qy**2)[:,None] + (qx**2)[None,:])
        op = th.from_numpy(op).view(*array.shape[-3:-1],1).to(array.device).type(array.dtype)

    # Transform to reciprocal space, apply operator and then inverse Fourier
    # transform
    return th.ifft(op.view(*op.shape,1)*th.fft(array,2),2)

def unwrap_FFT_method(cmplx,rsize=[1,1],reg=1e-5,op = None):
    """Unwrap a modulo 2pi image using the FFT Laplacian approach."""
    
    if op is None:
        # Get reciprocal space grid
        qy,qx = [np.fft.fftfreq(x,d=r/x) for x,r in zip(cmplx.shape[-3:-1],rsize) ]

        # Calculate reciprocal space Laplacian operator 4pi^2k^2
        op = -4*np.pi**2*((qy**2)[:,None] + (qx**2)[None,:]) #+ reg
        op = th.from_numpy(op).to(cmplx.device).type(cmplx.dtype)
    
    invop = deepcopy(op)
    invop[0,0] = reg
    
    shape = cmplx.shape[-3:-1]
    invop = invop.view(*shape,1)
    cmplx = complex_mul_conj(laplacian(cmplx,rsize,op=op),cmplx)
    # Take only imaginary part
    cmplx[...,0] = 0

    return th.ifft(th.fft(cmplx,2)/invop,2)[...,1]

def depth_section_reconstruction_2(
    t, beams, lam, S, Smatrixdimensions, phase_mask=None, qspacein=False, device=None,diffraction_shift=[0,0],optic_axis=[0,0]
):
    """
    Optical sectioning reconstruction from scattering matrix.

    Parameters:
    -----------
    t : float, array_like
        Optical depths at which the reconstruction will be created
    beams : integer, array_like (B x 2)
        List of beam coordinates
    lam : float
        Electron wavelength
    S : float, array_like (B x Y x X x 2)
        The scattering matrix input each component of the scattering matrix by
        default should be in real space
    Smatrixdimensions : float (2,) array_like
        Real space dimensions of the scattering matrix

    keyword arguements
    -----------------
    phase_mask : float, array_like, optional
        Phase mask to apply to the beams (for example a vortex probe)
    qspace_in : bool, optional
        If True then the each component of the scattering matrix is taken to
        be in reciprocal space.
    device : optional
        Allows the user to choose the reconstruction device, if None is pass
        then a cuda capable GPU device is chosen if available but cpu if not
    diffraction_shift : (2,) array_like, optional
        Diffraction shift of the 4D-STEM acquisition from which the S-matrix was
        reconstructed. This is the pixel coordinate shift from the center of
        the array for the origin of the diffraction coordinates.
    Returns
    -------
    EW :  float, array_like
        Set of reconstructions at depths t
    """
    device = get_device(device)
    S = ensure_torch_array(S, device=device)
    dtype = S.dtype

    # Get array shape of scattering matrix
    B, Y, X = S.shape[:-1]

    # Fourier transform S-matrix coordinates if necessary
    if qspacein:
        S = S
    else:
        S = th.fft(S, 2, True)

    # Generate Fourier grids
    qy = (
        th.from_numpy(np.fft.fftfreq(Y, Smatrixdimensions[0] / Y))
        .type(dtype)
        .to(device)
    )
    qx = (
        th.from_numpy(np.fft.fftfreq(X, Smatrixdimensions[1] / X))
        .type(dtype)
        .to(device)
    )

    # Shift all beams to origin for each component of the scattering
    # matrix and generate phase ramps for each beam
    phasey = th.zeros(B, Y, 1, device=device, dtype=dtype)
    phasex = th.zeros(B, 1, X, device=device, dtype=dtype)
    for ib, beam in enumerate(beams):
        by, bx = [x.item() for x in beam]

        # Shift beam to origin
        S[ib] = circshift(S[ib].clone(), [0, 1], [by, bx])

        # Fresnel propagation and paraxial shift of smatrix component
        dxy = [th.tan(qy[by-diffraction_shift[0]] * lam), th.tan(qx[bx-diffraction_shift[1]] * lam)]
        phasey[ib, ..., 0] = np.pi * lam * qy ** 2 + 2 * np.pi * qy * dxy[0]
        phasex[ib] = np.pi * lam * qx ** 2 + 2 * np.pi * qx * dxy[1]
    del qy, qx

    # Calculate reciprocal space Laplacian operator -4pi^2k^2 for unwrapping 
    # routine
    qy,qx = [np.fft.fftfreq(x) for x in [Y,X] ]
    op = th.from_numpy(-4*np.pi**2*((qy**2)[:,None] + (qx**2)[None,:])).to(device).type(dtype)

    EW = th.zeros(len(t), Y, X, dtype=dtype, device=device)

    for i, T in enumerate(tqdm.tqdm(t, desc="Optical section")):
        cmplx = th.ifft(
                    complex_mul(
                        complex_mul(S, complex_expi(T * phasey)),
                        complex_expi(T * phasex),
                    ),
                    2,
                )
        cmplx = cmplx/complex_abs(cmplx).view(*cmplx.shape[:-1],1)
        EW[i] = th.sum(unwrap_FFT_method(cmplx,op=op),axis=0)
            

    return EW.cpu().numpy()