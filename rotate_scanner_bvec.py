#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse

import nibabel as nib
import numpy as np

from scipy.spatial.transform import Rotation as R


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('bvecs', metavar='bvecs',
                   help='Path of the scanner bvecs.')
    p.add_argument('data', metavar='data',
                   help='Path of the data with relevant affine')
    p.add_argument('output', metavar='output',
                   help='Path of the output bvecs.')
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    img_data = nib.load(args.data)
    affine = img_data.affine
    voxsize = np.array(img_data.header.get_zooms()[:3])
    print('Voxel size seem to be ({} {} {})'.format(*voxsize))
    voxsize = np.repeat(voxsize[:, None], 3, axis=1)

    print('Affine matrix')
    print(affine)


    bvec_scanner = np.genfromtxt(args.bvecs)
    if bvec_scanner.shape[1] != 3:
        print('We want bvecs in 3 columns, TRANSPOSING')
        bvec_scanner = bvec_scanner.T   
    print('bvecs shape is ({}, {})'.format(*bvec_scanner.shape))

    print('bvecs norms:')
    print(np.linalg.norm(bvec_scanner, axis=1))

    bvec_scanner /= np.linalg.norm(bvec_scanner, axis=1)[:,None]
    bvec_scanner[np.isnan(bvec_scanner)] = 0
    print('bvecs norms after normalization:')
    print(np.linalg.norm(bvec_scanner, axis=1))


    affine_rot = (1/voxsize)*affine[:3,:3]
    print('Affine after voxelsize normalization, rotation only')
    print(affine_rot)

    print('SANITY CHECK: affine minus inv(affine).T   (should be near 0)')
    print(affine_rot - np.linalg.inv(affine_rot).T)

    # # experimental
    # trans = R.from_matrix(affine_rot)
    # rv = trans.as_rotvec()
    # rotangle = np.linalg.norm(rv)
    # rv_normed = rv/rotangle
    # print('original affine is equivalent to {:.1f} degree rotation around [{:.2f} {:.2f} {:.2f}]'.format((180/np.pi)*rotangle, rv_normed[0], rv_normed[1], rv_normed[2]))



    print('The bvectors transformation assumes stride +1,+2,+3, be sure to check for flips')
    fixed_affine = affine_rot.copy()
    fixed_affine[:,1] *= -1
    fixed_affine[1,:] *= -1
    fixed_affine *= -1


    # # experimental
    # trans = R.from_matrix(fixed_affine)
    # rv = trans.as_rotvec()
    # rotangle = np.linalg.norm(rv)
    # rv_normed = rv/rotangle
    # print('Fixed affine is equivalent to {:.1f} degree rotation around [{:.2f} {:.2f} {:.2f}]'.format((180/np.pi)*rotangle, rv_normed[0], rv_normed[1], rv_normed[2]))



    est_bvec = fixed_affine.dot(bvec_scanner.T).T
    print('Saving rotated bveccs as {}'.format(args.output))
    np.savetxt(args.output, est_bvec)


if __name__ == "__main__":
    main()





