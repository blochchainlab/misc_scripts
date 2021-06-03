#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse

import nibabel as nib
import numpy as np


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('data', metavar='data',
                   help='Path of the data with relevant affine')
    p.add_argument('mask', metavar='mask',
                   help='Path of the mask')
    p.add_argument('output_root', metavar='output_root',
                   help='Path of the output without extension')
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    img_data = nib.load(args.data)
    data = img_data.get_fdata()

    img_mask = nib.load(args.mask)
    mask = img_mask.get_fdata().astype(np.bool)

    # root path and name
    # we will save neighboor_mean, neighboor_std, difference_with_neighboor
    output_root = args.output_root



    # count neighboor
    neighboor_count = np.zeros(data.shape[:3])
    # sum neighboor radian diff
    neighboor_sum = np.zeros(data.shape[:3])
    print('{} voxels in mask'.format(mask.sum()))
    for ix in range(1, data.shape[0]-1):
        print('slice x = {}/{}'.format(ix, data.shape[0]-2))
        for iy in range(1, data.shape[1]-1):
            for iz in range(1, data.shape[2]-1):
                if mask[ix,iy,iz]:
                    if mask[ix+1, iy, iz]:
                        neighboor_sum[ix, iy, iz] += angle_between(data[ix+1, iy, iz], data[ix, iy, iz])
                        neighboor_count[ix, iy, iz] += 1
                    if mask[ix-1, iy, iz]:
                        neighboor_sum[ix, iy, iz] += angle_between(data[ix-1, iy, iz], data[ix, iy, iz])
                        neighboor_count[ix, iy, iz] += 1
                    if mask[ix, iy+1, iz]:
                        neighboor_sum[ix, iy, iz] += angle_between(data[ix, iy+1, iz], data[ix, iy, iz])
                        neighboor_count[ix, iy, iz] += 1
                    if mask[ix, iy-1, iz]:
                        neighboor_sum[ix, iy, iz] += angle_between(data[ix, iy-1, iz], data[ix, iy, iz])
                        neighboor_count[ix, iy, iz] += 1
                    if mask[ix, iy, iz+1]:
                        neighboor_sum[ix, iy, iz] += angle_between(data[ix, iy, iz+1], data[ix, iy, iz])
                        neighboor_count[ix, iy, iz] += 1
                    if mask[ix, iy, iz-1]:
                        neighboor_sum[ix, iy, iz] += angle_between(data[ix, iy, iz-1], data[ix, iy, iz])
                        neighboor_count[ix, iy, iz] += 1


    # # should make sure no voxel IN mask has 0 neighboor
    # neighboor_count[mask].min()

    # mean neighboor radian diff
    neighboor_mean = np.zeros(data.shape[:3])      
    neighboor_mean[mask] = neighboor_sum[mask] / neighboor_count[mask]



    nib.nifti1.Nifti1Image(neighboor_mean*(180/np.pi), img_data.affine).to_filename(output_root + '_mean_degree_neigh_diff.nii.gz')



def angle_between(v1, v2):
    # assumes norm=1
    # in radians
    tmp1 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    tmp2 = np.arccos(np.clip(np.dot(v1, -v2), -1.0, 1.0))
    # antipodally sym
    return min(tmp1, tmp2)


if __name__ == "__main__":
    main()





