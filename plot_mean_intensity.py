#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse

import matplotlib.pyplot as pl
import nibabel as nib
import numpy as np


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input', metavar='input',
                   help='Path of the input diffusion volume.')
    p.add_argument('bvals', metavar='bvals',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument(
        '--mask', dest='mask', metavar='mask',
        help='Path to a binary mask.\nOnly data inside the mask will be used '
             'for computations. (Default: None)')
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    img = nib.load(args.input)
    data = img.get_data()

    if args.mask is None:
        mask = None
        masksum = np.prod(data.shape[:3])
    else:
        mask = nib.load(args.mask).get_data().astype(np.bool)
        masksum = mask.sum()

    bvals = np.genfromtxt(args.bvals)

    b0_th = 75.
    b0_index = np.where(bvals < b0_th)[0]



    volume_mean_intensity = data[mask].mean(axis=0)

    viz_hack = volume_mean_intensity.copy()
    viz_hack[b0_index] = np.nan

    pl.figure()
    pl.plot(viz_hack, color='black', label='mean intensity in mask')
    pl.title('Mean intensity in mask (WITHOUT b0)')
    # pl.legend()

    pl.show()

if __name__ == "__main__":
    main()
