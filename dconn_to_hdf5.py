import numpy as np
import pandas as pd
import time
import h5py
import nibabel as nb
from nilearn import connectome

# Author: Martin Gell 
# Purpose: Load dconn file into hdf5 file (shrinks dconn from ~35 GB to ~8 GB)
# Last Updated: 4 Feb 2025

dconn = '/home/btervocl/shared/projects/martin_SNR/input/subpop/sub-1003001/ses-2/sub-1003001_ses-2_task-restMENORDICtrimmed_space-fsLR_den-91k_desc-denoised_bold_FD_02_smoothed_2mm'

def dconn_to_hdf5(dconn):
    start = time.time()
    # load data and get upper triangle
    print('loading...')
    #add
    nii = nb.load(f'{dconn}.dconn.nii')
    dat = nii.get_fdata()

    # turn to float16 to reduce size
    dat = dat.astype(np.float16)

    # extract only upper triangle of conn mat
    upper = connectome.sym_matrix_to_vec(dat, discard_diagonal = True)
    diagonal_value = dat[0,0]

    # save out as hdf5
    # should take about 25 seconds
    file_out = f'{dconn}.h5'
    h = h5py.File(file_out, 'w')
    dset = h.create_dataset('data', data=upper)
    dset = h.create_dataset('diagonal_value', data=np.atleast_1d(diagonal_value))
    dset = h.create_dataset('n_grayordinates', data=np.atleast_1d(dat.shape[0]))

    h.close()

    print('Saving...')
    print(f'{file_out}')

    end = time.time()
    print(f"HDF5 took {end - start} seconds")







    




