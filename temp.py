#Make sure code runs with Python 3.10.9 (base:conda), or libraries will not work
#from NonnegMFPy import nmf
import nmf_imaging
from astropy.io import fits
import numpy as np
import os
import pyklip.klip as pyklip
from astropy.wcs import WCS
import copy
import time
#import jax.numpy as jpn
#from jax import grad
#t0 = time.time()
a, ref, angles, coords, = [], [], [], []
cond, counter2 = True, 0

trg_bool = True
directory = os.fsencode("/home/ivanabreu/test_work/individual_exposures_calib-2")
  #Put in path to references

ra1, dec1 = (344.4110073, -29.6186750)
ra2, dec2 = (344.4118405, -29.6206386)
masks_fom = np.zeros((12,1100,1100))
for x in sorted(os.listdir(directory)):
    temp_file = fits.open(os.path.join(directory, x))
    if cond:
        header = WCS(temp_file[0].header)
        cond = False
    if counter2 == 0:
        a.append(temp_file[0].data)
        w = WCS(temp_file[0].header)
        x1, y1 = w.all_world2pix(ra1, dec1, 1)
        x2, y2 = w.all_world2pix(ra2, dec2, 1)
        #print(int(x1),int(x2), int(y1),int(y2))
        masks_fom[counter][int(y1):int(y2),int(x1):int(x2)] = 1.
        angles.append(temp_file[0].header['ORIENTAT'])
        coords.append([temp_file[0].header['CRPIX1'], temp_file[0].header['CRPIX2']])
    else:
        a.append(temp_file[0].data)
    counter2 += 1
    if counter2 == 28:
        counter2 = 0
        ref_min = np.median(np.nanmedian(a, 0))
        ref.append(np.nanmedian(a, 0) - ref_min)
        a.clear()

    temp_file.close()

ref = np.array(ref)
masks_fom[masks_fom < 0.9] = 0.1


ref[ref<=0] = 1e-6


file_mask = fits.open('mask_loci2spikes.fits')
  #Mask must be in directory 
mask = file_mask[0].data
#y, x = np.indices(mask.shape)
#mask[x<100]=0
#mask[y>900]=0


mask[mask >= -1] = 1.
mask[mask < -1] = 0.1 #mask cleanup


refshape = (ref.shape[0]-1, ref.shape[1], ref.shape[2])
ref_err = np.ones(refshape)
trg_errs = np.ones(ref.shape)
componentNum = 11
#components = nmf_imaging.NMFcomponents(ref, ref_err=b, mask = mask, n_components = componentNum, oneByOne=True)
results = np.zeros(ref.shape) # Say trgs is a 3D array containing the targets that need NMF modeling, then results store the NMF subtraction results.
maxiters = 1e5


t0 = time.time()
for i in range(ref.shape[0]):
      #can replace component, model, and result lines with commented lines below it to process whole image instead of doing data imputation
    ref_new = copy.deepcopy(ref)
    ref_new = np.delete(ref_new, i, 0)
    components = nmf_imaging.NMFcomponents(ref_new, ref_err=ref_err, mask = masks_fom[i], n_components = componentNum, oneByOne=True, maxiters=maxiters)
    #components = nmf_imaging.NMFcomponents(ref_new, ref_err=ref_err, mask = mask, n_components = componentNum, oneByOne=True, maxiters=maxiters)
    trgs = ref[i]
    trg_err = trg_errs[i]
    model = nmf_imaging.NMFmodelling(trg = trgs, trg_err=trg_err, components = components, n_components = componentNum, mask_components=mask, trgThresh=0.0, maxiters=maxiters, mask_data_imputation=masks_fom[i]) # Model the target with the constructed components.
    #model = nmf_imaging.NMFmodelling(trg = trgs, trg_err=trg_err, components = components, n_components = componentNum, mask_components=mask, trgThresh=0.0, maxiters=maxiters)
    best_frac = 1.
    result = nmf_imaging.NMFsubtraction(trgs, model, masks_fom[i], frac = best_frac) # Subtract the best model from the target
    #result = nmf_imaging.NMFsubtraction(trgs, model, mask, frac = best_frac)
    print(str(i)+" target calculated")
    results[i] = result
    # Now `results' stores the NMF subtraction results of the targets.


results_new = []
for i in range(ref.shape[0]):
    if i == 0:
        results_new.append(pyklip.rotate(results[i], angles[i], coords[i], astr_hdr=header))
        #results_new.append(pyklip.rotate(results[i], angles[i], coords[i], new_center=(550,550), astr_hdr=header))
    else:
        results_new.append(pyklip.rotate(results[i], angles[i], coords[i]))
        #results_new.append(pyklip.rotate(results[i], angles[i], coords[i], new_center=(550,550)))


hdr = header.to_header()
results_median = np.nanmedian(results_new, 0)
results_mean = np.nanmean(results_new, 0)


#result = nmf_imaging.nmf_func(trg = trg_fits[0].data, refs = a, mask = mask, componentNum=5)
#result_img = result.reshape(-1, result.shape[2])
#hdu = fits.PrimaryHDU(result_img, header2)
#hdu.header.comments['ORIGIN'] += 'NMF_imaging 24/7/2023'

hdu = fits.PrimaryHDU(results_mean, hdr)
hdu.writeto('output_mean.fits', overwrite=True)


hdu = fits.PrimaryHDU(results_median, hdr)
hdu.writeto('output_median.fits', overwrite=True)
t1 = time.time()
print(t1-t0, "seconds")

file_mask.close()
