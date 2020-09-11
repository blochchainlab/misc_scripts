#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import TensorModel


from scilpy.utils.bvec_bval_tools import (normalize_bvecs, is_normalized_bvecs)
# from scilpy.utils.filenames import add_filename_suffix, split_name_with_nii

from scipy.stats import gaussian_kde


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input', metavar='input',
                   help='Path of the input diffusion volume.')
    p.add_argument('bvals', metavar='bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs', metavar='bvecs',
                   help='Path of the bvecs file, in FSL format.')
    p.add_argument('bmax', metavar='bmax',
                   help='bmax to use')
    p.add_argument(
        '--mask', dest='mask', metavar='mask',
        help='Path to a binary mask.\nOnly data inside the mask will be used '
             'for computations and reconstruction. (Default: None)')
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    img = nib.load(args.input)
    data = img.get_data()

    bmax = int(args.bmax)


    print('\ndata shape ({}, {}, {}, {})'.format(data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    print('total voxels {}'.format(np.prod(data.shape[:3])))

    # remove negatives
    print('\ncliping negative ({} voxels, {:.2f} % of total)'.format((data<0).sum(),100*(data<0).sum()/float(np.prod(data.shape[:3]))))
    data = np.clip(data, 0, np.inf)


    affine = img.affine
    if args.mask is None:
        mask = None
        masksum = np.prod(data.shape[:3])
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)
        masksum = mask.sum()

    print('\nMask has {} voxels, {:.2f} % of total'.format(masksum,100*masksum/float(np.prod(data.shape[:3]))))

    # Validate bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    if not is_normalized_bvecs(bvecs):
        print('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)



    # Get tensors
    method = 'WLS'
    min_signal = 1e-16
    print('\nUsing fitting method {}'.format(method))
    # print('Using minimum signal = {}'.format(min_signal)

    b0_thr = bvals.min() + 10
    print('\nassuming existence of b0 (thr = {})\n'.format(b0_thr))


    # restricted gtab
    gtab = gradient_table(bvals[bvals<bmax+22], bvecs[bvals<bmax+22], b0_threshold=b0_thr)

    tenmodel = TensorModel(gtab, fit_method=method, min_signal=min_signal)

    tenfit = tenmodel.fit(data[..., bvals<bmax+22], mask)

    MD = tenfit.md
    FA = tenfit.fa

    evalmax = np.max(tenfit.evals, axis=3)
    invevalmax = evalmax**-1
    invevalmax[np.isnan(invevalmax)] = 0
    invevalmax[np.isinf(invevalmax)] = 0

    evalmin = np.min(tenfit.evals, axis=3)

    weird_contrast = np.exp(-bmax*evalmin) - np.exp(-bmax*evalmax)



    invMD = MD**-1
    invMD[np.isnan(invMD)] = 0
    invMD[np.isinf(invMD)] = 0


    nib.nifti1.Nifti1Image(MD, img.affine).to_filename('./MD_bmax_{}'.format(bmax))
    nib.nifti1.Nifti1Image(invMD, img.affine).to_filename('./invMD_bmax_{}'.format(bmax))
    nib.nifti1.Nifti1Image(FA, img.affine).to_filename('./FA_bmax_{}'.format(bmax))
    nib.nifti1.Nifti1Image(invevalmax, img.affine).to_filename('./inv_e1_bmax_{}'.format(bmax))
    nib.nifti1.Nifti1Image(weird_contrast, img.affine).to_filename('./minmax_contrast_bmax_{}'.format(bmax))






if __name__ == "__main__":
    main()



