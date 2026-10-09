import numpy as np
import sys # noqa
import pyms
import os

if __name__ == "__main__":

    eV = 3e5
    app = 20

    # Probe defocii
    defocii = [-200, -100, 0]

    # Number of frozen phonon iterations - 1 is fine for rough testing
    nfph= 1

    # Thicknesses
    thicknesses = [300]

    # Subslicing of unit cell for multislice
    subslices = [0.48, 0.7]

    # Multislice grid
    gridshape = [512, 512]
    tiling = [4, 4]

    # Datacube output shape
    # For a low 
    DPshape = [64, 64]
    DPsize = [7.0, 7.0]

    # Input some misalignment in the datacube
    Diffraction_shift = [5,7]
    outputDP = [32,32]
    outputDPsize = [x/y*z for x,y,z in zip(DPsize,DPshape,outputDP)]

    # Add some specimen tilt
    specimen_tilt = [5,2]

    # Scan pixel size
    Rpix = pyms.nyquist_sampling(eV=eV, alpha=app)

    structure = pyms.structure.fromfile("1000048.p1", temperature_factor_units="B")
    # structure.generate_slicing_figure(subslices)
    # print(
    #     pyms.max_grid_resolution(
    #         gridshape, np.asarray(tiling) * structure.unitcell[:2], eV=eV
    #     )
    # )
    if not os.path.exists('Outputs'):
        os.mkdir('Outputs')

    for df in tqdm.tqdm.(defocii,desc='Defocii'):
        h5_filename = "Outputs/Datacube_{0:04d}_Adf.h5".format(int(df))
        result = pyms.STEM_multislice(
            structure,
            gridshape,
            eV,
            app,
            thicknesses,
            subslices=subslices,
            df=df,
            nfph=nfph,
            FourD_STEM=[DPshape, DPsize],
            specimen_tilt = specimen_tilt,
            # h5_filename=h5_filename,
            tiling=tiling,
        )

        DP = pyms.utils.crop(np.roll(result['datacube'],Diffraction_shift,axis=(-2,-1)),outputDP)
        pyms.utils.datacube_to_py4DSTEM_viewable(DP,h5_filename,diffsize = outputDPsize,Rpix = Rpix,eV=eV,alpha=app)
