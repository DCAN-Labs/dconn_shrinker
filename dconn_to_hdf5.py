
import numpy as np
import time
import os
import h5py
import nibabel as nb
from nilearn import connectome

# Author: Martin Gell 
# Purpose: transform dconn file into hdf5 file (shrinks dconn from ~32 GB to ~8 GB)
# Last Updated: 4 Feb 2025


def dconn_to_hdf5(dconn_in):
    """
    Converts a dense connectome (.dconn.nii) file to HDF5 format.
    Importantly, we only save the upper triangle of the conn matrix without the diagonal.

    Parameters:
    dconn_in (str): The base name of the dense connectome file.

    The output HDF5 file will have the following datasets:
    - 'data': The upper triangle of the connectivity matrix.
    - 'diagonal_value': The value of the diagonal element of the connectivity matrix.
    - 'n_grayordinates': The number of grayordinates (nodes) in the connectivity matrix.
    """

    # start timer
    start = time.time()

    # load data and get upper triangle
    print('loading...')
    nii = nb.load(dconn_in)
    mat = nii.get_fdata()

    print(f'{dconn_in}')
    
    # turn to float16 to reduce size
    mat = mat.astype(np.float16)

    # extract only upper triangle of conn mat
    upper = connectome.sym_matrix_to_vec(mat, discard_diagonal = True)
    diagonal_value = mat[0,0]

    # save out as hdf5
    # -should take about 25 seconds
    # strip dconn_in of any file extensions
    file_out = os.path.splitext(dconn_in)[0]
    hdf5_out = f'{file_out}.h5'

    print('Saving...')
    h = h5py.File(hdf5_out, 'w')
    dset = h.create_dataset('data', data=upper)
    dset = h.create_dataset('diagonal_value', data=np.atleast_1d(diagonal_value))
    dset = h.create_dataset('n_grayordinates', data=np.atleast_1d(mat.shape[0]))
    h.close() # important to close the file!

    # print out file path and end timer
    print(f'{hdf5_out}')

    end = time.time()
    print(f'Exporting dconn as HDF5 took {end - start} seconds')


