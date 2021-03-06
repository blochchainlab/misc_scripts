#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse

import nibabel as nib
import numpy as np

from scipy.ndimage import convolve
from skimage.morphology import selem

from itertools import combinations, permutations

from time import time


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('data',
                   help='Path of the data with relevant affine')
    p.add_argument('mask',
                   help='Path of the mask')
    p.add_argument('output_fname',
                   help='Path of the output neighboor mean')
    return p

def eucl2deg(eucl):
    raw = (180/np.pi)*np.arccos(np.clip(1 - 0.5*eucl**2, -1, 1))
    return np.minimum(raw, 180-raw)

def main():
    print('Load data')
    parser = _build_args_parser()
    args = parser.parse_args()

    img_data = nib.load(args.data)
    data = img_data.get_fdata() # shape = (X,Y,Z,NMAX,3)

    img_mask = nib.load(args.mask)
    mask = img_mask.get_fdata().astype(np.bool)


    # 6-neighboors counting kernel
    W = selem.ball(1)
    W[1,1,1] = 0

    # # 26-neighboor counting kernel
    # W = selem.cube(3)
    # W[1,1,1] = 0

    # count neighboor
    print('Count neighboor')
    data_neigh_count = convolve(mask.astype(np.int), W, mode='constant', cval=0)

    # correct mask by removing voxel with no neighboor
    mask[data_neigh_count<0.5] = False

    # set to 0 stuff outside the mask
    data[np.logical_not(mask)] = 0

    # compute nufo
    print('Compute Nufo')
    nufo = np.round(np.linalg.norm(data, axis=4).sum(axis=3)).astype(np.int)

    # print "histogram" to detect weird stuff (for ex. non zeros vectors everywhere)
    totalvox = mask.sum()
    tmp = nufo[mask]
    val = [np.logical_and(tmp>i-0.5, tmp<i+0.5).sum() for i in range(11)]
    for i in range(len(val)):
        print('nufo={:} --> {:.3f} %'.format(i, 100*val[i]/totalvox))



    # build basic neighboor list from kernel
    tmp = np.where(W)
    neigh_indexing = [a for a in zip(tmp[0]-1, tmp[1]-1, tmp[2]-1)]

    # pad the 3 spacial dim
    datapad = np.pad(data, ((1,),(1,),(1,),(0,),(0,)))
    maskpad = np.pad(mask, 1)

    tt1 = time()
    metric = np.zeros(mask.shape)
    totali = 0
    for xyz in np.ndindex(data.shape[:3]):
        # translate coords into padcoords
        x = xyz[0]+1
        y = xyz[1]+1
        z = xyz[2]+1
        if maskpad[x,y,z]:

            if not totali % 10000:
                print('{:} / {:}  ({:.2f} %) ({:.0f} s)'.format(totali, totalvox, 100*totali/totalvox, time()-tt1))
            totali+=1

            nufo_center = nufo[xyz]
            tmp = []
            for indexing in neigh_indexing:
                xn = x+indexing[0]
                yn = y+indexing[1]
                zn = z+indexing[2]
                if maskpad[xn,yn,zn]:
                    nufo_neigh = nufo[xn-1, yn-1, zn-1] # because nufo isnt padded
                    Mmin = min(nufo_center, nufo_neigh)
                    Mmax = max(nufo_center, nufo_neigh)
                    if nufo_center <= nufo_neigh:
                        A = datapad[x, y, z]
                        B = datapad[xn, yn, zn]
                    else:
                        B = datapad[x, y, z]
                        A = datapad[xn, yn, zn]

                    # precompute all comparaisons for each Mmin
                    precomp = []
                    for i1 in range(Mmin):
                        precomp_tmp = []
                        for i2 in range(Mmax):
                            precomp_tmp.append(eucl2deg(np.linalg.norm(A[i1]-B[i2])))
                        precomp.append(precomp_tmp)

                    orderings = [el for subg in combinations(range(Mmax), Mmin) for el in permutations(subg)] 
                    candidate_val = [] 
                    for test_i in orderings:
                        tmp_sum = 0
                        for el_i in range(Mmin):
                            tmp_sum += precomp[el_i][test_i[el_i]]
                        candidate_val.append(tmp_sum/Mmin)

                    tmp.append(np.min(candidate_val))
            metric[xyz] = np.mean(tmp)

    print('{:.0f} s'.format(time()-tt1))



    print('Save map')
    nib.nifti1.Nifti1Image(metric, img_data.affine).to_filename(args.output_fname)


if __name__ == "__main__":
    main()





