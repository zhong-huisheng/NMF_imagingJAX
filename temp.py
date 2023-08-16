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
import radonCenter

a, trg, angles, coords, head = [], [], [], [], []
ref = np.zeros((12, 1100, 1100))
cond, counter2, counter = True, 0, 0

trg_bool = True
directory = os.fsencode("/home/ivanabreu/test_work/individual_exposures_calib-2")

ra1, dec1 = (344.4110073, -29.6186750)
ra2, dec2 = (344.4118405, -29.6206386)
masks_fom = np.zeros((12,1100,1100))
for x in sorted(os.listdir(directory)):
    temp_file = fits.open(os.path.join(directory, x))
    if cond:
        header = WCS(temp_file[0].header)
        cond = False
    if counter2 == 0:
        head.append(temp_file[0].header)
        w = WCS(temp_file[0].header)
        x1, y1 = w.all_world2pix(ra1, dec1, 1)
        x2, y2 = w.all_world2pix(ra2, dec2, 1)
        if x1 < x2:
            topx = int(x2)
            botx = int(x1)
        else:
            topx = int(x1)
            botx = int(x2)
        if y1 < y2:
            topy = int(y2)
            boty = int(y1)
        else:
            topy = int(y1)
            boty = int(y2)
        masks_fom[counter][boty:topy,botx:topx] = 1.
        angles.append(temp_file[0].header['ORIENTAT'])
    a.append(temp_file[0].data)
    counter2 += 1
    if counter2 == 28:
        counter2 = 0
        ref_min = np.median(np.nanmedian(a, 0))
        ref[counter] = np.nanmedian(a, 0) - ref_min
        coords.append(radonCenter.searchCenter(ref[-1], temp_file[0].header['CRPIX1'], temp_file[0].header['CRPIX2'], size_window = ref.shape[1]/2))
        a.clear()
        counter += 1

    temp_file.close()
masks_fom[masks_fom < 0.9] = 0.1

#trg[trg<=0] = 1e-6
#ref[ref<=0] = 1e-6

file_mask = fits.open('mask_loci2spikes.fits')
mask = file_mask[0].data

mask[mask >= -1] = 1.
mask[mask < -1] = 0.1 #mask cleanup


refshape = (ref.shape[0]-1, ref.shape[1], ref.shape[2])
ref_err = np.ones(refshape)

trg_errs = np.ones(ref.shape)
componentNum = 5
#components = nmf_imaging.NMFcomponents(ref, ref_err=b, mask = mask, n_components = componentNum, oneByOne=True)
results = np.zeros((2, ref.shape[0], ref.shape[1], ref.shape[2])) # Say trgs is a 3D array containing the targets that need NMF modeling, then results store the NMF subtraction results.
maxiters = 1e5

for i in range(ref.shape[0]):
    ref[i] = pyklip.rotate(ref[i], 0, coords[i], new_center=(550,550), astr_hdr=header)
    masks_fom[i] = pyklip.rotate(masks_fom[i], 0, coords[i], new_center=(550,550), astr_hdr=header)
mask = pyklip.rotate(mask, 0, coords[0], new_center=(550,550), astr_hdr=header)

t0 = time.time()
ref[ref<=0] = 1e-6
ref[np.isnan(ref)] = 1e-6
mask[np.isnan(mask)] = 1e-6
masks_fom[np.isnan(masks_fom)] = 1e-6

for i in range(ref.shape[0]):
    ref_new = copy.deepcopy(ref)
    ref_new = np.delete(ref_new, i, 0)
    components = nmf_imaging.NMFcomponents(ref_new, ref_err=ref_err, mask = masks_fom[i], n_components = componentNum, oneByOne=True, maxiters=maxiters)
    for x in range(components.shape[0]):
        results[1][x] = components[x]
    trgs = ref[i]
    trg_err = trg_errs[i]
    model = nmf_imaging.NMFmodelling(trg = trgs, trg_err=trg_err, components = components, n_components = componentNum, mask_components=mask, trgThresh=0.0, maxiters=maxiters, mask_data_imputation=masks_fom[i]) # Model the target with the constructed components.
      #best_frac =  nmf_imaging.NMFbff(trgs, model, mask) # Perform BFF procedure to find out the best fraction to model the target.
    best_frac = 1.0
      #result = trgs - model
    result = nmf_imaging.NMFsubtraction(trgs, model, masks_fom[i], frac = best_frac) # Subtract the best model from the target
    print(i,"targets calculated")
    results[0][i] = result

t1 = time.time()
print(t1-t0, "seconds")

results_new = []
for i in range(ref.shape[0]):
    if i == 0:
        results_new.append(pyklip.rotate(results[0][i], angles[i], (550,550), astr_hdr=header))
    else:
        results_new.append(pyklip.rotate(results[0][i], angles[i], (550,550)))

comp_save = copy.deepcopy(results[1])
for i in range(comp_save.shape[0] - components.shape[0]):
    comp_save = np.delete(comp_save, comp_save.shape[0]-1, 0)

hdu = fits.PrimaryHDU(comp_save)
hdu.writeto('components.fits', overwrite=True)

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

file_mask.close()
