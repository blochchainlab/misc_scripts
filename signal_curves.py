#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import nibabel as nib

from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from time import time

import argparse



DESCRIPTION = """
Compute voxelwise signal intensity curves along fitted DTI eigenvector.

1. Fit DTI on full dataset
2. Extract eigenvector e1, e2, e3.
3. For each shell, refit DTI.
4. For each shell, predict signal at "bvec" e1, e2, e3
5. Save the parallel curve S_e1(b)
6. Save the mean perpendicular curve (S_e2(b)+S_e3(b))/2
7. Save the abs. perpendicular difference curve  abs(S_e2(b)-S_e3(b))

In the case of multiple data/bval/bvec, they will be appended together.
In the case of multiple masks, they will be AND'd together.
"""

EPILOG = """
Michael Paquette, MPI CBS, 2021.
"""


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                epilog=EPILOG,
                                formatter_class=CustomFormatter)

    p.add_argument('--data', type=str, nargs='+', default=[],
                   help='Path of the input data (one or more).')

    p.add_argument('--bval', type=str, nargs='+', default=[],
                   help='Path of the input bval (one or more).')

    p.add_argument('--bvec', type=str, nargs='+', default=[],
                   help='Path of the input bvec (one or more).')

    p.add_argument('--mask', type=str, nargs='*', default=[],
                   help='Path of the input mask (one or more).')

    p.add_argument('--par', type=str,
                   help='Path of the output parallel curve.')

    p.add_argument('--perp', type=str,
                   help='Path of the output perpendicular curve.')

    p.add_argument('--diff', type=str,
                   help='Path of the output perpendicular difference curve.')

    return p



def main():
    parser = buildArgsParser()
    args = parser.parse_args()


    if args.par is None:
        print('Need output name for parallel curve')
        return None
    if args.perp is None:
        print('Need output name for perpendicular curve')
        return None
    if args.diff is None:
        print('Need output name for perpendicular difference curve')
        return None


    # load and concatenate all the bval
    print('Loading bval')
    bvals = [np.genfromtxt(fname) for fname in args.bval]
    bval = np.concatenate(bvals, axis=0)
    print('{:} b-values'.format(bval.shape[0]))
    del bvals


    # load and concatenate all the bvec
    print('Loading bvec')
    bvecs = []
    for fname in args.bvec:
        tmp = np.genfromtxt(fname)
        if tmp.shape[1] != 3:
            tmp = tmp.T
        bvecs.append(tmp)
    bvec = np.concatenate(bvecs, axis=0)
    print('{:} b-vectors'.format(bvec.shape[0]))
    del bvecs


    if bval.shape[0] != bvec.shape[0]:
        print('Mismatch of bval and bvec')
        return None


    # load and concatenate all the data
    print('Loading data')
    data_img = [nib.load(fname) for fname in args.data]
    affine = data_img[0].affine
    data_data = []
    for img in data_img:
        tmp = img.get_fdata()
        print('data shape = {:}'.format(tmp.shape))
        # need 4D data for the concatenate
        if tmp.ndim == 3:
            tmp = tmp[..., None]
        data_data.append(tmp)
    data = np.concatenate(data_data, axis=3)
    print('Full data shape = {:}'.format(data.shape))
    del data_data


    if bval.shape[0] != data.shape[3]:
        print('Mismatch of bval/bvec and data')
        return None


    # load and multiply all the mask
    print('Loading Mask')
    mask = np.ones(data.shape[:3], dtype=np.bool)
    mask_data = [nib.load(fname).get_fdata().astype(np.bool) for fname in args.mask]
    for tmp in mask_data:
        mask = np.logical_and(mask, tmp)
    print('Final mask has {:} voxels ({:.1f} % of total)'.format(mask.sum(), 100*mask.sum()/np.prod(data.shape[:3])))
    del mask_data


    b0_th = 50 # threshold below which a bvalues is considered a b0
    round_th = 250 # threshold used to round the bvals into shells
    print('bval below {:} are round to 0'.format(b0_th))
    print('rounding bval to nearest {:}'.format(round_th))

    bval[bval<b0_th] = 0
    bval = round_th*np.round(bval/round_th, decimals=0)
    bval_shell = sorted(list(set(bval)))
    print('#Vol | bval Shell')
    for b in bval_shell:
        print('{:}  at  b = {:.0f}'.format((bval==b).sum(), b))


    # clean data
    data = np.clip(data, 0, np.inf)
    data[np.isinf(data)] = 0
    data[np.isnan(data)] = 0


    # Fit DTI with all data
    # keep eigenvectors
    print('Fit DTI on full data')
    gtab = gradient_table(bval, bvec)
    tenmodel = TensorModel(gtab, fit_method='WLS', min_signal=1e-16)
    start_time = time()
    tenfit = tenmodel.fit(data, mask)
    end_time = time()
    print('elapsed time = {:.0f} sec'.format(end_time-start_time))
    eigenvectors1 = tenfit.evecs[..., :, 0]
    eigenvectors2 = tenfit.evecs[..., :, 1]
    eigenvectors3 = tenfit.evecs[..., :, 2]


    S_par_list = []
    S_perp1_list = []
    S_perp2_list = []
    # Fit DTI on each shell
    for bshell in bval_shell:
        if bshell > 0:
            print('Fitting b = {:}'.format(bshell))
            shell_mask = np.logical_or(bval==0, bval==bshell)
            shell_gtab = gradient_table(bval[shell_mask], bvec[shell_mask])
            shell_tenmodel = TensorModel(shell_gtab, fit_method='WLS', min_signal=1e-16)
            start_time = time()
            shell_tenfit = shell_tenmodel.fit(data[..., shell_mask], mask)
            end_time = time()

            adc_par = np.einsum('...i,...i->...', np.einsum('...i,...ij->...j', eigenvectors1, tenfit.quadratic_form), eigenvectors1)
            adc_perp1 = np.einsum('...i,...i->...', np.einsum('...i,...ij->...j', eigenvectors2, tenfit.quadratic_form), eigenvectors2)
            adc_perp2 = np.einsum('...i,...i->...', np.einsum('...i,...ij->...j', eigenvectors3, tenfit.quadratic_form), eigenvectors3)

            S_par = np.exp(-bshell*adc_par)
            S_perp1 = np.exp(-bshell*adc_perp1)
            S_perp2 = np.exp(-bshell*adc_perp2)

            S_par_list.append(S_par[...,None])
            S_perp1_list.append(S_perp1[...,None])
            S_perp2_list.append(S_perp2[...,None])


    data_par = np.concatenate(S_par_list, axis=3)
    data_perp1 = np.concatenate(S_perp1_list, axis=3)
    data_perp2 = np.concatenate(S_perp2_list, axis=3)

    data_perp = (data_perp1 + data_perp2) / 2.0
    data_diff = np.abs(data_perp1 - data_perp2)


    nib.Nifti1Image(data_par*mask[...,None], affine).to_filename(args.par)
    nib.Nifti1Image(data_perp*mask[...,None], affine).to_filename(args.perp)
    nib.Nifti1Image(data_diff*mask[...,None], affine).to_filename(args.diff)



if __name__ == "__main__":
    main()



