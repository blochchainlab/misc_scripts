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


    mds = []
    for i in range(len(unique_bvalues)-1):
        # max_shell = i+1
        print('fitting using {} th shells (bmax = {})'.format(i+2, bvals[shells==i+1].max()))

        # restricted gtab
        # gtab = gradient_table(bvals[shells <= i+1], bvecs[shells <= i+1], b0_threshold=b0_thr)
        gtab = gradient_table(bvals[np.logical_or(shells == i+1, shells == 0)], bvecs[np.logical_or(shells == i+1, shells == 0)], b0_threshold=b0_thr)

        tenmodel = TensorModel(gtab, fit_method=method, min_signal=min_signal)

        tenfit = tenmodel.fit(data[..., np.logical_or(shells == i+1, shells == 0)], mask)

        evalmax = np.max(tenfit.evals, axis=3)
        evalmin = np.min(tenfit.evals, axis=3)

        evalmax[np.isnan(evalmax)] = 0
        evalmin[np.isnan(evalmin)] = 0
        evalmax[np.isinf(evalmax)] = 0
        evalmin[np.isinf(evalmin)] = 0

        weird_contrast = np.exp(-unique_bvalues[i+1]*evalmin) - np.exp(-unique_bvalues[i+1]*evalmax)

        mds.append(weird_contrast[mask])






    # peaks = []
    oneq = []
    twoq = []
    threeq = []
    th = 0.01
    print('\nonly using values inside quantile [{}, {}] for plotting'.format(th, 1-th))
    for i in range(len(unique_bvalues)-1):
        plt.figure()
        tit = 'exp(-b diff_MIN) - exp(-b diff_MAX), {} th shells (bmax = {})'.format(i+2, bvals[shells==i+1].max())
        print('\nbmax = {}'.format(bvals[shells==i+1].max()))
        # truncate lower and upper MD to remove crazy outliers
        minval = 0
        # maxval = np.quantile(mds[i], 1-th)
        tmp = mds[i]
        # vv1 = tmp.shape[0]
        tmp = tmp[tmp > minval]
        # vv2 = tmp.shape[0]
        print('removed {} zeros'.format(mds[i].shape[0] - tmp.shape[0]))

        # # remove high diffusivity non physical outlier
        # idx1 = (tmp <= 1/3.0e-3) # free water diffusivity at in-vivo brain temperature
        # print('{} voxels above free water diffusivity ({:.2f} % of mask)'.format(idx1.sum(), 100*idx1.sum()/float(masksum)))
        # # remove low diffusivity probable outlier
        # th_diff = 0.05
        # idx2 = (tmp >= 1/(th_diff*1.0e-3)) # 1% of mean diffusivity of in-vivo WM at in-vivo brain temperature
        # print('{} voxels below {} of in-vivo WM diffusivity ({:.2f} % of mask)'.format(idx2.sum(),th_diff, 100*idx2.sum()/float(masksum)))
        # tmp = tmp[np.logical_not(np.logical_or(idx1, idx2))]
        # fit smoothed curve for peak extraction
        # gkde = gaussian_kde(tmp)

        # plt.hist(tmp, bins=100, density=True, color='grey')
        logbins = np.logspace(-2,np.log10(0.7),100)
        plt.hist(tmp, bins=logbins, density=True, color='grey')
        # bs = np.linspace(tmp.min(), tmp.max(), 1000)
        # bs = np.logspace(np.log10(tmp.min()), np.log10(tmp.max()), 1000)
        # smoothed = gkde.pdf(bs)
        # plt.plot(bs, smoothed, color='blue', linewidth=2)
        plt.semilogx([],[])
        # plt.semilogx(bs, smoothed, color='blue', linewidth=2)
        # peak extraction
        # smoothed_peak = bs[smoothed.argmax()]
        # plt.axvline(smoothed_peak, color='red', label='peak ({:.0f})'.format(smoothed_peak))
        # peaks.append(smoothed_peak)
        # useless extra lines
        onequart = np.quantile(tmp, 0.25)
        twoquart = np.quantile(tmp, 0.5)
        threequart = np.quantile(tmp, 0.75)
        oneq.append(onequart)
        twoq.append(twoquart)
        threeq.append(threequart)
        plt.axvline(onequart, color='pink', label='25% ({:.2f})'.format(onequart))
        plt.axvline(twoquart, color='yellow', label='50% ({:.2f})'.format(twoquart))
        plt.axvline(threequart, color='green', label='75% ({:.2f})'.format(threequart))
        plt.title(tit)
        plt.legend(loc=1)

        plt.xlim([0.01,0.7])

        plt.savefig('./dSEst_SS_bmax_{:.0f}.png'.format(unique_bvalues[i]))

    # plt.show()





    # print('\nHigher-than-required bmax will artifactually decrease MD, increasing 1/MD')
    # print('The error on the estimation of 1/MD should be small when the peak is close to bmax')
    # print('This is under the assumption that we have a valid WM mask so that the tissues are somewhat uniform')
    


    bmaxs = np.array([bvals[shells==i+1].max() for i in range(len(unique_bvalues)-1)])


    # plt.figure()
    # plt.plot(bmaxs, peaks, '-x', label = 'fit')
    # plt.plot(bmaxs, bmaxs, label = 'identity')
    # plt.xlabel('bmax')
    # plt.ylabel('MD^-1')
    # plt.legend()
    # plt.title('PEAK')

    # plt.savefig('./bvalEst_peak.png')

    plt.figure()
    plt.grid()
    plt.plot(bmaxs, oneq, '-x', label = 'fit')
    # plt.plot(bmaxs, bmaxs, label = 'identity')
    plt.xlabel('bmax')
    # plt.ylabel('MD^-1')
    # plt.legend()
    # plt.title('25% quartile')

    # plt.savefig('./bvalEst_50Q.png')

    # plt.figure()
    plt.plot(bmaxs, twoq, '-x', label = 'fit')
    # plt.plot(bmaxs, bmaxs, label = 'identity')
    plt.xlabel('bmax')
    # plt.ylabel('MD^-1')
    # plt.legend()
    # plt.title('50% quartile')

    # plt.savefig('./bvalEst_50Q.png')

    # plt.figure()
    plt.plot(bmaxs, threeq, '-x', label = 'fit')
    # plt.plot(bmaxs, bmaxs, label = 'identity')
    plt.xlabel('bmax')
    plt.ylabel('delta S')
    # plt.legend()
    # plt.title('75% quartile')
    plt.title('Quartile')

    plt.savefig('./bvalEst_SS_Qs.png')

    # plt.show()



if __name__ == "__main__":
    main()



