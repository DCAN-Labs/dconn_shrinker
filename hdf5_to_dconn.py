
import numpy as np
import os
import time
import h5py
import nibabel as nb
from nilearn import connectome

# Author: Martin Gell 
# Purpose: Load hdf5 file into dconn format
# Last Updated: 4 Feb 2025

def hdf5_to_dconn(hdf5_in, dconn_ref):
    """
    Converts an HDF5 file containing connectome data to a CIFTI-2 dense connectivity (dconn.nii) file,
    using the header information from the reference dconn file.

    Parameters:
    hdf5_in (str): Path to the input HDF5 file.
    dconn_ref (str): Path to a reference dconn file to copy header information from.
            This should be ideally the exact type of dconn that you want reconstructed.
    """

    # Open the HDF5 file for reading
    print('loading...')
    hdf5_file = h5py.File(hdf5_in,'r')

    print(f'{hdf5_in}')

    # Read the upper triangular part of the connectome matrix
    upper = hdf5_file["data"][0:len(hdf5_file["data"])]

    # Get the info about diagonal
    diag_len =   hdf5_file["n_grayordinates"][0]
    diag_value = hdf5_file["diagonal_value"][0]

    # Reconstruct the full symmetric matrix
    print('reconstructing matrix...')
    mat = connectome.vec_to_sym_matrix(upper, diagonal=np.repeat(np.nan,diag_len))
    np.fill_diagonal(mat, diag_value)

    # Convert the matrix to float32
    mat = mat.astype(np.float32)

    # Load the reference dconn file to copy header information
    nii = nb.load(f'{dconn_ref}')

    # Create a new CIFTI-2 image with the reconstructed matrix and the copied header information
    file_out = os.path.splitext(hdf5_in)[0]
    file_out = os.path.splitext(file_out)[0]
    dconn_out = f'{file_out}_NEW.dconn.nii'

    print('Saving...')
    new_img = nb.Cifti2Image(mat, header=nii.header,
                        nifti_header=nii.nifti_header)
    new_img.to_filename(f'{dconn_out}')

    print(f'{dconn_out}')

    # Close the HDF5 file
    hdf5_file.close() # important to close the file!

