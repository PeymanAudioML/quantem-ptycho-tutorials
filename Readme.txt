All code that was used in the publication "A three-dimensional reconstruction algorithm for scanning transmission electron microscopy data"

Directories
-----------

Gradient_descent_scattering_matrix

The Python-based gradient descent reconstruction algorithm, folder also contains scripts to produce simulated pseudo-data for testing (requires py_multislice https://github.com/HamishGBrown/py_multislice).


Scripts
-------

Aligner.py

A handly little auxillary script that finds the diffraction shift (deviation of diffraction pattern from center) and specimen tilt (which can be corrected in mild cases) from the PACBED. An example PACBED is provided in ./Inputs/PACBED.tif

Distortion_correct.py

Fits peaks to the atomic columns in the HAADF and then creates a map of the probe positions in between the columns, this is used for in the reconstruction method to account for drift and probe distoritions. Requires python modules scipy and shapely and Magnus Nord's atommap package (https://atomap.org/)

Process_datasets.py

The script to process the datasets, this requires the raw experimental data which can be requested from the authors but is not hosted with the submission due to space constraints.




