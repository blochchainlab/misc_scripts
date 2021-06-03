#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import pylab as pl
import argparse
# from dipy.io.gradients import read_bvals_bvecs
# from scilpy.utils.bvec_bval_tools import (normalize_bvecs, is_normalized_bvecs)
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti




# desciption = """
# Correct the mean instensity of the input volume to be equal inside a brain mask.
# Uses the minimum as the normalization target

# """




# def _build_args_parser():
#     p = argparse.ArgumentParser(description=desciption, formatter_class=argparse.RawTextHelpFormatter)
#     p.add_argument('data', metavar='data', help='Path of the nifti file.')
#     p.add_argument('output', metavar='output', help='Path of the output nifti.')
#     p.add_argument('scale', metavar='scale', help='Path of the output scaling.')
#     p.add_argument('--mask', metavar='mask', help='Path of the brain mask for normalization.')
#     return p


# def main():
#     parser = _build_args_parser()
#     args = parser.parse_args()



# # data
# img_data = nib.load(args.data)
# data = img_data.get_fdata()
# affine = img_data.affine


basename = '/data/pt_02101_dMRI/009_C_W_HOIMA2/preprocessed/201008_009_C_W_HOIMA2_Bremerhaven/CE_Pre_Assessment/nii'

b0_img = nib.load(basename + '/Dti_Epi_center_b0_RG20_rescale.nii.gz')
b0_data = b0_img.get_fdata()

data_imgs = [nib.load(basename+'/Dti_Epi_{}_of_6_rescale.nii.gz'.format(num)) for num in range(1,7)]
datas = [img.get_fdata() for img in data_imgs]

data_cat = np.concatenate((b0_data[...,None],) + tuple(datas), axis=3)
del datas

# # Validate bvals and bvecs
# bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

# if not is_normalized_bvecs(bvecs):
#     print('Your b-vectors do not seem normalized...')
#     bvecs = normalize_bvecs(bvecs)

all_bvecs = [np.genfromtxt(basename+'/Dti_Epi_{}_of_6.bvec'.format(num)) for num in range(1,7)]
all_bvals = [np.genfromtxt(basename+'/Dti_Epi_{}_of_6.bval'.format(num)) for num in range(1,7)]

bvec = np.concatenate((np.array([[0,0,0]]),) + tuple(all_bvecs), axis=0)
bval = np.concatenate(([0],) + tuple(all_bvals), axis=0)

gtab = gradient_table(bval, bvec)

# mask
# mask_path = args.mask
mask_path = '/data/pt_02101_dMRI/009_C_W_HOIMA2/preprocessed/201008_009_C_W_HOIMA2_Bremerhaven/MP_noise_test/fslfast_afterN4_mixeltype.nii.gz'
img_mask = nib.load(mask_path)
mask_raw = img_mask.get_fdata()
mask = np.zeros(mask_raw.shape, dtype=np.bool)
mask[np.logical_and(mask_raw<4.5, mask_raw>3.5)] = True

volume_mean_intensity = data_cat[mask].mean(axis=0)
# if mask_path is None:
#     print('No mask given, using the full volumes')

#     # mean intensity inside each volume
#     volume_mean_intensity = data_cat.mean(axis=(0,1,2))
# else:
#     img_mask = nib.load(mask_path)
#     mask = img_mask.get_fdata().astype(np.bool)

#     # mean intensity inside mask for each volume
#     volume_mean_intensity = data_cat[mask].mean(axis=0)



selection_method = 1

# fake file
low_idx = np.zeros(bval.shape[0], dtype=np.bool)
low_idx[11:30] = True
low_idx[bval < 50] = True

if selection_method == 0:
	# select last 12 + b0
	print('Selecting last 12 image and all b0 for low-dir fit')
	low_idx = np.zeros(bval.shape[0], dtype=np.bool)
	low_idx[-12:] = True
	low_idx[bval < 50] = True
elif selection_method == 1:
	# read a file
	low_idx = low_idx
else:
	print('Not implemented')



pl.figure()
pl.plot(volume_mean_intensity[bval > 50], color='black', label='mean intensity in mask')
pl.scatter(np.where(low_idx), np.ones(low_idx.sum())*(0.95*volume_mean_intensity.min()), label='Selection', color='red')
pl.legend()
pl.show()




# todo, make everything masked

low_bvec = bvec[low_idx]
low_bval = bval[low_idx]

data_low = data_cat[..., low_idx]


gtab_low = gradient_table(low_bval, low_bvec)
tenmodel_low = dti.TensorModel(gtab_low)


# tenmodel_low = dti.TensorModel(gtab_low, fit_method='WLS', return_S0_hat=True)
tenmodel_low = dti.TensorModel(gtab_low, fit_method='WLS')
tenfit_low = tenmodel_low.fit(data_low, mask=mask)


# predicted_S0 = tenfit_low.S0_hat
predicted_S0 = np.ones(data_low.shape[:3])

predicted_signal = tenfit_low.predict(gtab)

predicted_normalized_signal = predicted_signal / predicted_S0[..., None]
predicted_normalized_signal[np.isnan(predicted_normalized_signal)] = 0
predicted_normalized_signal[np.isinf(predicted_normalized_signal)] = 0

predicted_adc = -np.log(predicted_normalized_signal)/bval[None, None, None, :]
predicted_adc[np.isinf(predicted_adc)] = 0
predicted_adc[np.isnan(predicted_adc)] = 0


# ks = 1 - (np.log(data_cat/predicted_signal) * (bval[None, None, None, :]*predicted_adc)**-1)
ks = 1 - (np.log(data_cat/(predicted_signal*data_cat[..., 0, None])) * (bval[None, None, None, :]*predicted_adc)**-1)

ks[np.isnan(ks)] = 1
ks[np.isinf(ks)] = 1




pl.figure()
pl.plot(np.mean(ks[mask], axis=(0,)))

pl.figure()
pl.plot(np.median(ks[mask], axis=(0,)))

pl.show()



tmp = ks[mask][:,1].ravel()
_ = pl.hist(tmp, bins=100, range=[0, 2], color='blue')
# pl.axvline(tmp.mean(), label='mean = {:.3f}'.format(tmp.mean()), color='red')
pl.axvline(np.median(tmp), label='median = {:.3f}'.format(np.median(tmp)), color='black')
pl.legend()
pl.show()




mean_directional_k = np.median(ks[mask], axis=(0,))

# expected signal from mean k
tmp = np.zeros(data_cat.shape)
for i in range(len(bval)):
    tmp[...,i] = np.exp(-bval[i]*mean_directional_k[i]*predicted_adc[...,i])
    
# S_pred / S_obs     a.k.a  S_low / S_heat    
Cs = predicted_normalized_signal / tmp




pl.figure()
pl.plot(np.mean(Cs[mask], axis=(0,)))

pl.figure()
pl.plot(np.median(Cs[mask], axis=(0,)))

pl.show()



tmp = Cs[mask][:,1].ravel()
_ = pl.hist(tmp, bins=100, range=[0, 2], color='blue')
pl.axvline(tmp.mean(), label='mean = {:.3f}'.format(tmp.mean()), color='red')
pl.axvline(np.median(tmp), label='median = {:.3f}'.format(np.median(tmp)), color='black')
pl.legend()
pl.show()


meank_correction = Cs[mask].mean(axis=(0,))

data_meank = data_cat * meank_correction[None,None,None,:]
meandata_meank = data_meank[mask].mean(axis=(0,))

pl.figure()
pl.plot(meandata_meank[bval>50], color='red', label='meank')
pl.plot(volume_mean_intensity[bval > 50], color='black', label='raw')
pl.legend()
pl.title('Correction using mean correction factor from mean-k adjustement')
pl.show()






# since we use b=5, we'll calibrate on ADC = 1/b = 0.2
D_calibrate = 1/bval[1]
heat_calibrate = np.exp(-bval*mean_directional_k*D_calibrate)

# pl.plot(np.exp(-bval*D_calibrate)/heat_calibrate)
meank_fix_correction = np.exp(-bval*D_calibrate)/heat_calibrate

data_meank_fix = data_cat * meank_fix_correction[None,None,None,:]
meandata_meank_fix = data_meank_fix[mask].mean(axis=(0,))

pl.figure()
pl.plot(meandata_meank_fix[bval>50], color='red', label='meank_fix')
pl.plot(volume_mean_intensity[bval > 50], color='black', label='raw')
pl.legend()
pl.title('Correction using mean correction factor from mean-k_fix adjustement')
pl.show()



pl.figure()
pl.plot(meandata_meank_fix[bval>50], color='red', label='meank_fix')
pl.plot(volume_mean_intensity[bval > 50], color='black', label='raw')
pl.plot(meandata_meank[bval>50], color='blue', label='meank')
pl.legend()
# pl.title('Correction using mean correction factor from mean-k_fix adjustement')
pl.show()



