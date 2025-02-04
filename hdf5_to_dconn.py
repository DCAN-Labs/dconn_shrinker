import numpy as np
import pandas as pd
import time
import h5py
import nibabel as nb
from nilearn import connectome

# Author: Martin Gell 
# Purpose: Load hdf5 file into dconn format
# Last Updated: 4 Feb 2025

dconn_dir = '/home/faird/shared/code/internal/utilities/dconn_shrinker/'
hdf5_in =   f'{dconn_dir}sub-1003001_ses-2_task-restMENORDICtrimmed_space-fsLR_den-91k_desc-denoised_bold_FD_02_smoothed_2mm.h5'
dconn_out = f'{dconn_dir}sub-1003001_ses-2_task-restMENORDICtrimmed_space-fsLR_den-91k_desc-denoised_bold_FD_02_smoothed_2mm.dconn.nii'
dconn_ref = f'{dconn_dir}sub-1003001_ses-2_task-restMENORDICtrimmed_space-fsLR_den-91k_desc-denoised_bold_FD_02_smoothed_2mm_new.dconn.nii'

def hdf5_to_dconn(hdf5_in, dconn_out, dconn_ref):
    hdf5_file = h5py.File(hdf5_in,'r')

    upper = hdf5_file["data"][0:len(hdf5_file["data"])]

    diag_len = hdf5_file["n_grayordinates"][0]

    mat = connectome.vec_to_sym_matrix(upper, diagonal=np.repeat(hdf5_file["diagonal_value"][0],diag_len))

    mat = mat.astype(np.float32)

    nii = nb.load(f'{dconn_ref}')

    new_img = nb.Cifti2Image(mat, header=nii.header,
                        nifti_header=nii.nifti_header)
    new_img.to_filename(f'{dconn_out}')

    hdf5_file.close()


# exported dconn should look the same a the original
