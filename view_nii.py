#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl


def disp3D(data):

	X = data.shape[0] // 2
	Y = data.shape[1] // 2
	Z = data.shape[2] // 2

	slice_YZ = data[X, :, :]
	slice_XZ = data[:, Y, :]
	slice_XY = data[:, :, Z]

	mean_X = data.mean(axis=0)
	mean_Y = data.mean(axis=1)
	mean_Z = data.mean(axis=2)

	mean_YZ = data.mean(axis=(1,2))
	mean_XZ = data.mean(axis=(0,2))
	mean_XY = data.mean(axis=(0,1))


	pl.figure()

	pl.subplot(3,3,1)
	pl.imshow(slice_YZ)
	pl.title('mid slice YZ (min/max = {:.2e} / {:.2e})'.format(slice_YZ.min(), slice_YZ.max()))
	pl.colorbar()
	pl.axis('off')
	pl.subplot(3,3,2)
	pl.imshow(slice_XZ)
	pl.title('mid slice XZ (min/max = {:.2e} / {:.2e})'.format(slice_XZ.min(), slice_XZ.max()))
	pl.colorbar()
	pl.axis('off')
	pl.subplot(3,3,3)
	pl.imshow(slice_XY)
	pl.title('mid slice XY (min/max = {:.2e} / {:.2e})'.format(slice_XY.min(), slice_XY.max()))
	pl.colorbar()
	pl.axis('off')

	pl.subplot(3,3,4)
	pl.imshow(mean_X)
	pl.title('mean over X (min/max = {:.2e} / {:.2e})'.format(mean_X.min(), mean_X.max()))
	pl.colorbar()
	pl.axis('off')
	pl.subplot(3,3,5)
	pl.imshow(mean_Y)
	pl.title('mean over Y (min/max = {:.2e} / {:.2e})'.format(mean_Y.min(), mean_Y.max()))
	pl.colorbar()
	pl.axis('off')
	pl.subplot(3,3,6)
	pl.imshow(mean_Z)
	pl.title('mean over Z (min/max = {:.2e} / {:.2e})'.format(mean_Z.min(), mean_Z.max()))
	pl.colorbar()
	pl.axis('off')

	pl.subplot(3,3,7)
	pl.plot(mean_YZ)
	pl.title('mean over YZ (min/max = {:.2e} / {:.2e})'.format(mean_YZ.min(), mean_YZ.max()))
	pl.subplot(3,3,8)
	pl.plot(mean_XZ)
	pl.title('mean over XZ (min/max = {:.2e} / {:.2e})'.format(mean_XZ.min(), mean_XZ.max()))
	pl.subplot(3,3,9)
	pl.plot(mean_XY)
	pl.title('mean over XY (min/max = {:.2e} / {:.2e})'.format(mean_XY.min(), mean_XY.max()))


	pl.show()




# data = np.random.rand(100,100,60)
# disp3D(data)




def main(fname):
	data = nib.load(fname).get_fdata()

	print('data shape ', data.shape)
	print('NaN count = {}'.format(np.isnan(data).sum()))
	data[np.isnan(data)] = 0

	if data.ndim == 4:
		data = data[:,:,:,0]



	disp3D(data)



if __name__ == "__main__":
	import sys
	import nibabel as nib
	main(sys.argv[1])










