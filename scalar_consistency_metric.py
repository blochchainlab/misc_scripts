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
    # sum neighboor
    neighboor_sum = np.zeros(data.shape[:3])
    print('{} voxels in mask'.format(mask.sum()))
    for ix in range(1, data.shape[0]-1):
        print('slice x = {}/{}'.format(ix, data.shape[0]-2))
        for iy in range(1, data.shape[1]-1):
            for iz in range(1, data.shape[2]-1):
                if mask[ix,iy,iz]:
                    neighboor_count[ix, iy, iz] += mask[ix+1, iy, iz]
                    neighboor_count[ix, iy, iz] += mask[ix-1, iy, iz]
                    neighboor_count[ix, iy, iz] += mask[ix, iy+1, iz]
                    neighboor_count[ix, iy, iz] += mask[ix, iy-1, iz]
                    neighboor_count[ix, iy, iz] += mask[ix, iy, iz+1]
                    neighboor_count[ix, iy, iz] += mask[ix, iy, iz-1]

                    neighboor_sum[ix, iy, iz] += data[ix+1, iy, iz]
                    neighboor_sum[ix, iy, iz] += data[ix-1, iy, iz]
                    neighboor_sum[ix, iy, iz] += data[ix, iy+1, iz]
                    neighboor_sum[ix, iy, iz] += data[ix, iy-1, iz]
                    neighboor_sum[ix, iy, iz] += data[ix, iy, iz+1]
                    neighboor_sum[ix, iy, iz] += data[ix, iy, iz-1]


    # # should make sure no voxel IN mask has 0 neighboor
    # neighboor_count[mask].min()

    # mean neighboor
    neighboor_mean = np.zeros(data.shape[:3])      
    neighboor_mean[mask] = neighboor_sum[mask] / neighboor_count[mask]




    # std neighboor
    neighboor_meandiff = np.zeros(data.shape[:3])
    for ix in range(1, data.shape[0]-1):
        print('slice x = {}/{}'.format(ix, data.shape[0]-2))
        for iy in range(1, data.shape[1]-1):
            for iz in range(1, data.shape[2]-1):
                if mask[ix,iy,iz]:
                    neighboor_meandiff[ix, iy, iz] += np.abs(data[ix+1, iy, iz] - neighboor_mean[ix, iy, iz])
                    neighboor_meandiff[ix, iy, iz] += np.abs(data[ix-1, iy, iz] - neighboor_mean[ix, iy, iz])
                    neighboor_meandiff[ix, iy, iz] += np.abs(data[ix, iy+1, iz] - neighboor_mean[ix, iy, iz])
                    neighboor_meandiff[ix, iy, iz] += np.abs(data[ix, iy-1, iz] - neighboor_mean[ix, iy, iz])
                    neighboor_meandiff[ix, iy, iz] += np.abs(data[ix, iy, iz+1] - neighboor_mean[ix, iy, iz])
                    neighboor_meandiff[ix, iy, iz] += np.abs(data[ix, iy, iz-1] - neighboor_mean[ix, iy, iz])    


    # std neighboor
    neighboor_std = np.zeros(data.shape[:3])      
    neighboor_std[mask] = neighboor_meandiff[mask] / neighboor_count[mask]



    # error with neigboor mean
    data_consistency = data - neighboor_mean 


    nib.nifti1.Nifti1Image(data_consistency, img_data.affine).to_filename(output_root + '_difference_with_neigh_mean.nii.gz')
    nib.nifti1.Nifti1Image(np.abs(data_consistency), img_data.affine).to_filename(output_root + '_difference_abs_with_neigh_mean.nii.gz')
    nib.nifti1.Nifti1Image(neighboor_mean, img_data.affine).to_filename(output_root + '_neigh_mean.nii.gz')
    nib.nifti1.Nifti1Image(neighboor_std, img_data.affine).to_filename(output_root + '_neigh_std.nii.gz')





if __name__ == "__main__":
    main()





