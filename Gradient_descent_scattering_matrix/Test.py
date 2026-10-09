import torch as th
import matplotlib.pyplot as plt
import GradDS
import numpy as np
import h5py
from PIL import Image


def Gaussian(grid, A, alpha, beta, gamma):
    """
    Calculate a 2D Gaussian function.

    Functional form A*exp(-alpha*x**2-beta*x*y-gamma*y**2), grid[0] contains
    y gridpoints and grid[1] contains x gridpoints.
    """
    return A * np.exp(
        -(alpha * grid[1] ** 2 + beta * grid[0] * grid[1] + gamma * grid[0] ** 2)
    )


def Gaussian_test_object(gridshape, minphase=0, maxphase=10 * np.pi):
    """Creates a test object to attempt the phase unwrapping on (a 2D Gaussian)"""
    alpha = 1/(gridshape[1] // 4)**2
    gamma = 1/(gridshape[0] // 4)**2
    beta = 0
    A = maxphase

    npiy,npix = gridshape
    xx = np.broadcast_to(
        np.fft.fftshift(np.fft.fftfreq(npix, 1 / npix)).reshape((1, npix)), (npiy, npix)
    )
    yy = np.broadcast_to(
        np.fft.fftshift(np.fft.fftfreq(npiy, 1 / npiy)).reshape((npiy, 1)), (npiy, npix)
    )
    
    return Gaussian(np.stack([yy,xx],axis=0), A, alpha, beta, gamma)

def test(testobject):
    """Test the unwrapping routine on an array testobject and plot the results."""    
    fig, ax = plt.subplots(ncols=4)
    for a in ax.ravel():
        a.set_axis_off()
    ax[0].imshow(testobject)
    ax[0].set_title("Test object")
    phase = np.angle(np.exp(1j * testobject))
    ax[1].imshow(phase)
    ax[1].set_title("Wrapped")
    complexobj = np.exp(1j * testobject)
    complexobj = np.stack([complexobj,complexobj],axis=0)
    unwrapped = GradDS.unwrap_FFT_method(GradDS.cx_from_numpy(complexobj[0]))
    unwrapped = unwrapped.cpu().numpy()
    # unwrapped = AmpflowS.cx_to_numpy(unwrapped)
    ax[2].imshow(unwrapped)
    ax[2].set_title("Unwrapped")
    ax[3].imshow(unwrapped - np.min(unwrapped) - (testobject - testobject.min()))
    ax[3].set_title("Difference")
    return fig

def renormalize(array,newmin,newmax):
    """Renormalize an array to have new minimum and maximum values"""
    min_,max_ = [array.min(),array.max()]
    return (array-min_)/(max_-min_)*(newmax-newmin) + newmin


if __name__ == "__main__":

    # gridshape = [512, 256]
    # testobject = renormalize(Gaussian_test_object(gridshape),-np.pi,5*np.pi)
    
    # test(testobject).savefig('Gaussian_test.png')
    # plt.show(block=True)

    # from skimage.data import astronaut
    # testobject = renormalize(np.sum(astronaut(),axis=2),0,np.pi*3)

    # test(testobject).savefig('Astronaut_test.png')
    # sys.exit()
    mu = 1.0
    defocii= [-200, -100, 0]
    files = [ "Outputs/Datacube_{0:04d}_Adf.h5".format(int(df)) for df in defocii]
    f = h5py.File(files[0],'r')
    K_pix_units = f['/4D-STEM_data/metadata/calibration'].attrs['K_pix_size'][0]
    R_pix_units = f['/4D-STEM_data/metadata/calibration'].attrs['R_pix_size'][0]
    eV = f['/4D-STEM_data/metadata/calibration'].attrs['accelerating_voltage'][0]
    app = f['/4D-STEM_data/metadata/calibration'].attrs['convergence_semiangle_mrad'][0]
    dcube = f['/4DSTEM_experiment/data/datacubes/datacube_0/datacube'][:]
    Ry,Rx,Ky,Kx = dcube.shape

    Diffraction_shift = [5,7]
    specimen_tilt = [5,2]

    dimensions = [Ry*R_pix_units,Rx*R_pix_units,Ky*K_pix_units,Kx*K_pix_units]

    fig,segimg,ovlapimg = GradDS.check_alignment(dcube,dimensions,eV,app,diffraction_shift=Diffraction_shift)

    fig.savefig('Outputs/Alignment_check.pdf')
    Image.fromarray(segimg).save('Outputs/Alignment_check_segimg.tif')
    Image.fromarray(ovlapimg).save('Outputs/Alignment_check_ovlapimg.tif')
    
    datacube = np.stack([h5py.File(f,'r')['/4DSTEM_experiment/data/datacubes/datacube_0/datacube'] for f in files])
    Image.fromarray(np.sum(datacube[0],axis=(0,1))).save('Outputs/PACBED.tif')

    Smatrix, beams_index, smatrix_dimensions, Loss = GradDS.reconstruct_Smatrix_from_datacube(
        datacube,
        dimensions,
        defocii,
        app,
        eV,
        mu=1.0,
        niterations=4,
        nchunks=30,
        # nstreams=2,
        stream_datacube=True,
        report_memory=True,
        diffraction_shift=Diffraction_shift,
        report_loss=True
        )

    f = h5py.File('Outputs/Smatrix.h5','w')
    f.create_dataset('Smatrix',data=np.abs(np.fft.fft2(Smatrix)))
    f.close()

    if Loss is not None:
        fig,ax = plt.subplots()
        ax.plot(Loss)
        ax.set_title('Loss')
        fig.savefig('Outputs/Loss.pdf')

    import pyms
    beammask=None
    # beammask = pyms.make_detector(Smatrix.shape[-2:],smatrix_dimensions[-2:],eV,app,app/2) > 0
    
    # fig,ax = plt.subplots()
    # ax.imshow(np.where(beammask,1,0))
    # plt.show(block=True)

    t = np.arange(-400,100,10)
    EW = GradDS.depth_section_reconstruction(t, np.asarray(beams_index).T, 1/pyms.wavev(eV), Smatrix, smatrix_dimensions,beam_mask=beammask,diffraction_shift=Diffraction_shift,optic_axis=specimen_tilt)
    
    phase = np.angle(EW)
    pyms.utils.stack_to_animated_gif(phase,'Outputs/depth_section_no_correction.gif')
    f = h5py.File('Outputs/Optical_section.h5','w')
    f.create_dataset('Phase',shape = phase.shape,dtype = phase.dtype,data=phase)
    f.close()