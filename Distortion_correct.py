import sys, os
import numpy as np
from PIL import Image
import hyperspy.api as hs
import matplotlib.pyplot as plt
import atomap.api as am
from scipy.spatial.distance import cdist
from copy import deepcopy


def closest_node(node, nodes):
    indx = cdist([node], nodes).argmin()
    return nodes[indx], indx

fitnewdatapoints = False
# Prepare directory and title for figure output
outputsdir = 'Outputs'
plotsdir = "Outputs/Plots"
fittedpeaksdir = 'Outputs/fitted_peaks'
scangriddir = 'Outputs/Scan_grids'
correctedimagesdir = 'Outputs/Corrected_images'

for dir_ in [outputsdir,plotsdir,fittedpeaksdir,scangriddir,correctedimagesdir]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)

indir = os.path.split(sys.argv[1])[0]
title = os.path.splitext(os.path.split(sys.argv[1])[1])[0]

# Open image
img = np.asarray(Image.open(sys.argv[1]))
# Invert contrast if requested by command line - good for fitting ABF images
invert = False
if len(sys.argv) > 2:
    invert = sys.argv[2] == "I"
if invert:
    img = np.amax(img) - img

# Make hyperspy signal object to use with hyperspy
s = hs.signals.Signal2D(img)

fnam = os.path.join(fittedpeaksdir,'{0}.npy'.format(title))

if fitnewdatapoints or not os.path.exists(fnam):
    # Fit atom positions using atomap
    atom_positions = am.get_atom_positions(s, separation= 7)
    atom_positions_new = am.add_atoms_with_gui(s, atom_positions)
    plt.show(block=True)

    sublattice_A = am.Sublattice(atom_positions_new, image=s.data, color="r")
    sublattice_A.find_nearest_neighbors()
    sublattice_A.refine_atom_positions_using_center_of_mass()
    sublattice_A.refine_atom_positions_using_2d_gaussian()
    
    datapoints = np.asarray(deepcopy(sublattice_A.atom_positions))
    np.save(fnam,datapoints)
else:
    datapoints = np.load(fnam)

fig,ax = plt.subplots()
ax.imshow(s.data)
ax.plot(datapoints[:,0],datapoints[:,1],'ro',markersize=3)
fig.savefig(os.path.join(plotsdir,"Fitted_peaks_{0}_{1}_new.pdf".format(indir,title)))

# Find upper left, upper right etc points in the grid
nopiy,nopix = img.shape
from scipy.spatial.distance import cdist
edges = np.asarray([[22,0],[nopix,0],[0,nopiy],[nopix,nopiy]])
edges = datapoints[np.argmin(cdist(edges,datapoints),axis=1),:]

# Estimate unitcell vectors from edges of grid
gridvec = np.asarray([edges[1]-edges[0],edges[2]-edges[0]])
gridvec += np.asarray([edges[3]-edges[2],edges[3]-edges[1]])
gridvec /= 2
dps= deepcopy(datapoints)
datapoints = datapoints.tolist()

# A note on conventions: usually with numpy first index is y position, second is x
# atomap uses the opposite, so we'll stick with that but, as is common
# when going between different conventions, mistakes are easy to make. Look
# for book-keeping comments

# Number of unit cells in x and y directions
nucx = 6
nucy = 7
avec = np.asarray([gridvec[0]/nucx,gridvec[1]/nucy])

# sys.exit()

# Angle of rotation of lattice in radians
theta = np.arccos(gridvec[0,0]/np.linalg.norm(gridvec[0]))*np.sign(gridvec[0,1])
print(np.rad2deg(theta))
theta = 8.5
theta = np.deg2rad(-5)
theta = np.deg2rad(13.8)

# Seperation in pixels of unit cell
rvec = int(np.mean(np.sqrt(avec[:,0]**2+avec[:,1]**2)))
print(rvec)
shearedpoints = []

for y in range(nucy+1):
    for x in range(nucx+1):
        vec = edges[0]+avec[0]*x+avec[1]*y
        shearedpoints.append(vec)
shearedpoints = np.asarray(shearedpoints)
rvec = 14

# Vertices of "ideal" unit cell lattice that distorted, real lattice will be mapped to
# these arrays are returned flattened by ravel()
xx, yy = [
    x.ravel()
    for x in np.meshgrid(
        np.arange(nucx+1, dtype=np.float), np.arange(nucy+1, dtype=np.float)
    )
]
# idealpoints dimensions are N x 2 (x,y)
idealpoints = (
    xx[:, np.newaxis] * np.asarray([np.cos(theta), np.sin(theta)]) * rvec
    + yy[:, np.newaxis] * np.asarray([-np.sin(theta), np.cos(theta)]) * rvec
)

# Align mean points of ideal and data arrays
idealpoints -= np.mean(idealpoints, axis=0) - np.mean(datapoints, axis=0)

# Define a function that, for a point, will find the index of the
# closest point in an array of points nodes

# Index to sort arrays by
indx = np.argsort(cdist([[64, 64]], idealpoints)[0, :])

# Make a list of the corresponding closest point in the data to each point
# in the ideal lattice
corrpoints = [None] * indx.shape[0]

corrpoints[0] = edges[0]
for i in range(1,indx.shape[0]):
    if i==1 : 
        add = avec[0]
        start = corrpoints[0]
    elif i%(nucx+1) == 0:
        add = avec[1]
        start = corrpoints[i-nucx-1]
    else:
        add = avec[0]
        start = corrpoints[i-1]
    nxtpnt = (start + add).reshape((1,2))
    indx = np.argmin(cdist(nxtpnt,datapoints),axis=1)[0]
    corrpoints[indx]
    corrpoints[i] = datapoints.pop(indx)

corrpoints = np.asarray(corrpoints)

# Align new array to common point
common_origin = np.asarray([64, 64]) - np.mean(idealpoints, axis=0)

# Plot the ideal lattice and the corresponding points
fig, ax = plt.subplots()
ax.imshow(img)
cmap = plt.get_cmap("gist_rainbow")
ax.scatter(
    idealpoints[:, 0], idealpoints[:, 1], c=np.arange(idealpoints.shape[0]), cmap=cmap
)
ax.scatter(
    corrpoints[:, 0], corrpoints[:, 1], c=np.arange(idealpoints.shape[0]), cmap=cmap
)
for i in range(idealpoints.shape[0]):
    ax.plot(
        [idealpoints[i, 0], corrpoints[i, 0]],
        [idealpoints[i, 1], corrpoints[i, 1]],
        color=cmap(i / idealpoints.shape[0]),
    )
ax.set_title("Mapping of actual lattice points onto ideal lattice")
fig.savefig(os.path.join(plotsdir,"Lattice_mapping_{0}_{1}.pdf".format(indir,title)))
# plt.show(block=True)

cmap = plt.get_cmap("gist_rainbow")
fig, ax = plt.subplots()
ax.imshow(img)

from shapely.geometry import Polygon

def points_to_cells(points_,nucx,nucy):
    # Construct vertices
    cells_ = []
    for i in range((nucx+1) * nucy):
        # Top left hand corner
        if i % (nucx+1) != nucx:
            ps = [points_[i, :], points_[i + 1, :], points_[i + nucx+2, :], points_[i + nucx+1, :], points_[i, :]]
            cells_.append(Polygon(tuple([tuple(x) for x in ps])))
    return cells_


cells = points_to_cells(idealpoints,nucx,nucy)

corrcells = points_to_cells(corrpoints,nucx,nucy)
shearcells = points_to_cells(shearedpoints,nucx,nucy)

for icell, cell in enumerate(shearcells):
    xs, ys = cell.exterior.xy
    c = cmap(icell / len(cells))
    ax.plot(xs, ys, linestyle="-", color=c, linewidth=1)


for icell, cell in enumerate(corrcells):
    xs, ys = cell.exterior.xy
    c = cmap(icell / len(cells))
    # ax.plot(xs, ys, linestyle="-", color=c, linewidth=1)
for icell, cell in enumerate(cells):
    xs, ys = cell.exterior.xy
    c = cmap(icell / len(cells))
    ax.plot(xs, ys, linestyle=":", color=c, linewidth=1)
ax.set_title("Mapping of ideal and actual lattices")
datapoints = np.load(fnam)
ax.plot(datapoints[:,0],datapoints[:,1],'wo')
ax.scatter(shearedpoints[:,0], shearedpoints[:,1], c=np.arange(shearedpoints.shape[0]),cmap=cmap)
ax.plot(edges[:,0],edges[:,1],'ro')
for i in range(2):
    ax.plot([edges[0,0],edges[0,0]+avec[i,0]],[edges[0,1],edges[0,1]+avec[i,1]],'m-')
plt.show(block=False)
fig.savefig(os.path.join(plotsdir,"Lattice_mapping_{0}_new.pdf".format(title)))
transforms = np.zeros((len(cells), 2, 2))
Origins = np.zeros((len(cells), 2, 2))
for icell, cell in enumerate(corrcells):
    ps = np.stack(cell.exterior.xy, axis=1)
    qs = np.stack(cells[icell].exterior.xy, axis=1)
    Origins[icell, 0, :] = ps[0, :]
    Origins[icell, 1, :] = qs[0, :]
    transforms[icell] = np.vstack((ps[1] - ps[0], ps[2] - ps[0])).T
    transforms[icell] = np.linalg.inv(transforms[icell])
    transforms[icell] = (
        np.vstack((qs[1, :] - qs[0, :], qs[2, :] - qs[0, :])).T @ transforms[icell, ...]
    )


from shapely.geometry import Point, MultiPoint, MultiPolygon

gridpoints = MultiPoint(
    np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    .reshape((2, np.prod(img.shape)))
    .T.tolist()
)


cell_list = -np.ones((np.prod(img.shape),), np.int)

# import IPython

# IPython.embed()

# TODO This bit is really slow, speed up
bounds = MultiPolygon(corrcells).convex_hull
for ipoint, point in enumerate(gridpoints):
    if bounds.contains(point):
        cell_list[ipoint] = -2

for icell, cell in enumerate(corrcells):
    for ipoint, point in enumerate(gridpoints):
        if cell_list[ipoint] > -2:
            continue
        if cell.contains(point):
            cell_list[ipoint] = icell
cell_list[cell_list == -2] = -1

fig, ax = plt.subplots()
ax.imshow(img, alpha=1.0)
# Book-keeping: Since cell_list has been made using atomic coordinates
# it needs to be plotted transposed to match the numpy row, column format
# of the image
cell_list= cell_list.reshape(img.shape)
from matplotlib.colors import Normalize
cmap = plt.get_cmap("gist_rainbow")
colors = Normalize(cell_list.min(),cell_list.max(),clip=True)(cell_list.T)
colors = cmap(colors)
colors[...,-1]  = 0.5
colors[...,-1][cell_list.T<0] = 0
ax.imshow(colors)
ax.set_axis_off()
for icell, cell in enumerate(corrcells):
    xs, ys = cell.exterior.xy
    ax.plot(xs, ys, "k-", linewidth=1)
ax.set_title("Cell mappings")
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plotname = os.path.join(plotsdir,"Cell_mappings_{0}_{1}_new".format(indir,title))
fig.savefig(plotname + ".pdf")
fig.savefig(plotname + ".png")


from scipy.spatial.distance import cdist

warped_grid = -np.ones((*img.T.shape, 2))
for i in range(img.shape[1]):
    for j in range(img.shape[0]):
        if cell_list[j, i] == -1:
            continue
        # Get cell
        icell = cell_list[j, i]
        warped_grid[j, i, :] = (
            (transforms[icell, ...] @ (np.asarray([j, i]) - Origins[icell, 0, :]))
            + Origins[icell, 1, :]
            + common_origin
        )

# warped_grid needs to be transposed in its first two dimensions to match
# the usual numpy convention
warped_grid = np.rollaxis(warped_grid, 1, 0)

# To match the usual numpy convention the order of the final dimension
# must also be reversed
warped_grid = warped_grid[...,::-1]

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(warped_grid[..., 1], cmap=plt.get_cmap("gist_rainbow"))
ax[1].imshow(warped_grid[..., 0], cmap=plt.get_cmap("gist_rainbow"))

# Save warped grid to HDF5 file
fnam = os.path.join(scangriddir,"Scan_grid_{0}_{1}.h5".format(indir,title))
import h5py

f = h5py.File(fnam, "w")
f.create_dataset(
    "Scan_grid", shape=warped_grid.shape, data=warped_grid, dtype=warped_grid.dtype
)
f.close()


for icell, cell in enumerate(corrcells):
    xs, ys = cell.exterior.xy
    ax[0].plot(xs, ys, linestyle="--", color="k", linewidth=1)
    ax[1].plot(xs, ys, linestyle="--", color="k", linewidth=1)
ax[0].set_title("Grid x")
ax[1].set_title("Grid y")
fig.savefig(os.path.join(plotsdir,"Fitted_grid_{0}_{1}_new.pdf".format(indir,title)))
plt.show(block=False)


from scipy.interpolate import griddata


def renomarlize(array):
    return (array - np.amin(array)) / (np.amax(array) - np.amin(array))

img_shape = img.shape

# Make pixel grid, mgrid result is 2 x Y x X, flatten the final two dimensions
# and swap so that result is (Y*X) x 2
xi = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].reshape((2, np.prod(img.shape))).T

if len(sys.argv) > 3:
    img = np.asarray(Image.open(sys.argv[3]))
    if invert:
        img = np.amax(img) - img
    title = os.path.splitext(sys.argv[3])[0]

points_ = warped_grid.reshape((np.prod(img.shape), 2))
mask = np.logical_not(np.all(points_ == -1, axis=1))

A = (
    griddata(points_[mask, :], img.ravel()[mask], xi, fill_value=np.amin(img))
    .reshape(img.shape)
)

fig, ax = plt.subplots()
ax.imshow(A)
ax.set_title("Corrected image")
plt.show(block=False)
outdir = os.path.join(correctedimagesdir,indir)
if not os.path.exists(outdir):
    os.mkdir(outdir)
Image.fromarray(A).save(os.path.join(outdir,"corrected_{0}_{1}.tif".format(indir,title)))
