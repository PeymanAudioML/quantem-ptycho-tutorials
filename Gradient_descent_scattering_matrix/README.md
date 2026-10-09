# Gradient_descent_scattering_matrix
Gradient descent reconstruction of the scattering matrix from 4D-STEM data

![](depth_section.gif)

To install use this command in the directory

`pip install -e .`

To test (requires py_multislice https://github.com/HamishGBrown/py_multislice and smatrix, included in the ancillary files) 

`ipython Generate_test_data.py`

Use the py4DSTEM viewer (https://github.com/HamishGBrown/py_multislice) or the FIJI (https://imagej.net/Fiji) hdf5 plugin to view the 4D-STEM datacubes 

`ipython Test.py`
