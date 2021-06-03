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

def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input', metavar='input',
                   help='Path of the input diffusion volume.')
    p.add_argument('bvals', metavar='bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('bvecs', metavar='bvecs',
                   help='Path of the bvecs file, in FSL format.')
    p.add_argument(
        '--mask', dest='mask', metavar='mask',
        help='Path to a binary mask.\nOnly data inside the mask will be used '
             'for computations and reconstruction. (Default: None)')
    return p



def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    img = nib.load(args.input)
    data = img.get_fdata()

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


    # detect unique b-shell and assign shell id to each volume
    # sort bvals to get monotone increasing bvalue
    bvals_argsort = np.argsort(bvals)
    bvals_sorted = bvals[bvals_argsort]

    b_shell_threshold = 25.
    unique_bvalues = []
    shell_idx = []

    unique_bvalues.append(bvals_sorted[0])
    shell_idx.append(0)
    for newb in bvals_sorted[1:]:
        # check if volume is in existing shell
        done = False
        for i,b in enumerate(unique_bvalues):
            if (newb - b_shell_threshold < b) and (newb + b_shell_threshold > b):
                shell_idx.append(i)
                done = True
        if not done:
            unique_bvalues.append(newb)
            shell_idx.append(i+1)

    unique_bvalues = np.array(unique_bvalues)
    # un-sort shells
    shells = np.zeros_like(bvals)
    shells[bvals_argsort] = shell_idx



    print('\nWe have {} shells'.format(len(unique_bvalues)))
    print('with b-values {}\n'.format(unique_bvalues))

    for i in range(len(unique_bvalues)):
        shell_b = bvals[shells==i]
        print('shell {}: n = {}, min/max {} {}'.format(i, len(shell_b), shell_b.min(), shell_b.max()))




    # Get tensors
    method = 'WLS'
    min_signal = 1e-16
    print('\nUsing fitting method {}'.format(method))
    # print('Using minimum signal = {}'.format(min_signal)

    b0_thr = bvals.min() + 10
    print('\nassuming existence of b0 (thr = {})\n'.format(b0_thr))



    fas = []
    mds = []
    lams_max = []
    lams_min = []
    delta_S = []
    raw_signal = []
    for i in range(len(unique_bvalues)-1):
        # max_shell = i+1
        print('fitting using {} th shells (bmax = {})'.format(i+2, bvals[shells==i+1].max()))

        # restricted gtab
        # gtab = gradient_table(bvals[shells <= i+1], bvecs[shells <= i+1], b0_threshold=b0_thr)
        gtab = gradient_table(bvals[np.logical_or(shells == i+1, shells == 0)], bvecs[np.logical_or(shells == i+1, shells == 0)], b0_threshold=b0_thr)

        tenmodel = TensorModel(gtab, fit_method=method, min_signal=min_signal)

        tenfit = tenmodel.fit(data[..., np.logical_or(shells == i+1, shells == 0)], mask)
        raw_signal.append(data[..., np.logical_or(shells == i+1, shells == 0)][mask].mean(axis=1))

        evalmax = np.max(tenfit.evals, axis=3)
        evalmin = np.min(tenfit.evals, axis=3)

        evalmax[np.isnan(evalmax)] = 0
        evalmin[np.isnan(evalmin)] = 0
        evalmax[np.isinf(evalmax)] = 0
        evalmin[np.isinf(evalmin)] = 0

        weird_contrast = np.exp(-unique_bvalues[i+1]*evalmin) - np.exp(-unique_bvalues[i+1]*evalmax)


        mds.append(tenfit.md[mask])
        fas.append(tenfit.fa[mask])
        lams_max.append(evalmax[mask])
        lams_min.append(evalmin[mask])
        delta_S.append(weird_contrast[mask])


    bmaxs = np.array([bvals[shells==i+1].max() for i in range(len(unique_bvalues)-1)])


    names = ['FA',
             'MD',
             'eval_max',
             'eval_min',
             'delta_S',
             'eval_max_minus_eval_min',
             'raw_signal']

    units = ['a.u.',
             'mm^2/s',
             'mm^2/s',
             'mm^2/s',
             'contrast (a.u.)',
             'mm^2/s',
             'raw signal (a.u.)']


    datas = [np.array(fas).mean(axis=1),
             np.array(mds).mean(axis=1),
             np.array(lams_max).mean(axis=1),
             np.array(lams_min).mean(axis=1),
             np.array(delta_S).mean(axis=1),
             (np.array(lams_max)-np.array(lams_min)).mean(axis=1),
             np.array(raw_signal).mean(axis=1)]


    for i in range(len(names)):
        plt.figure()
        plt.plot(bmaxs, datas[i])
        plt.title(names[i])
        plt.xlabel('bval (s/mm^2)')
        plt.ylabel(units[i])

        plt.savefig('./roi_plot_'+names[i]+'.png', dpi=150)


    plt.show()





if __name__ == "__main__":
    main()






