#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse

import nibabel as nib
import numpy as np

from dipy.reconst.shm import calculate_max_order



def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input', metavar='input',
                   help='Path of the input sh volume.')
    p.add_argument('output', metavar='output',
                   help='Path of the output sh volume.')
    p.add_argument('flip', metavar='flip',
                   help='Orientation to flip the sh (x, y or z)')
    return p



def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    img = nib.load(args.input)
    data = img.get_fdata()
    affine = img.affine

    print('Assuming sh are in Mrtrix non-legacy basis.')
    print('Assuming sh are not full basis')

    # build l and m vectors
    lmax = calculate_max_order(data.shape[-1], full_basis=False)
    l = []
    m = []
    for ll in range(0, lmax+1, 2):
        for mm in range(-ll, ll+1):
            l.append(ll)
            m.append(mm)
    l = np.array(l)
    m = np.array(m)

    # build relevant order mask
    mask_pos_odd = np.logical_and((m%2).astype(bool), m>0)
    mask_neg_odd = np.logical_and((m%2).astype(bool), m<0)
    mask_pos_even = np.logical_and(~(m%2).astype(bool), m>=0) # include m==0
    mask_neg_even = np.logical_and(~(m%2).astype(bool), m<0)


    flip_signs = np.ones(data.shape[-1])
    if   (args.flip == 'x' or args.flip == 'X'):
        print('Flipping in the X axis')
        # build the corresponding sign flips
        flip_signs[mask_pos_odd]  = -1.
        flip_signs[mask_neg_odd]  = +1.
        flip_signs[mask_pos_even] = +1.
        flip_signs[mask_neg_even] = -1.
    elif (args.flip == 'y' or args.flip == 'Y'):
        print('Flipping in the Y axis')
        # build the corresponding sign flips
        flip_signs[mask_pos_odd]  = +1.
        flip_signs[mask_neg_odd]  = -1.
        flip_signs[mask_pos_even] = +1.
        flip_signs[mask_neg_even] = -1.
    elif (args.flip == 'z' or args.flip == 'Z'):
        print('Flipping in the Z axis')
        # build the corresponding sign flips
        flip_signs[mask_pos_odd]  = -1.
        flip_signs[mask_neg_odd]  = -1.
        flip_signs[mask_pos_even] = +1.
        flip_signs[mask_neg_even] = +1.
    else:
        print('flip can only be x, y, or z')
        return


    # flip signs and save
    nib.Nifti1Image(data*flip_signs[None, None, None], img.affine).to_filename(args.output)




if __name__ == "__main__":
    main()


