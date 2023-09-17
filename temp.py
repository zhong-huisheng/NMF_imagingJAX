#Make sure code runs with Python 3.10.9 (base:conda), or libraries will not work
import nmf_imaging
from astropy.io import fits
import numpy as np
import os
import pyklip.klip as pyklip
from astropy.wcs import WCS
import copy
import time
import radonCenter
import matplotlib.pyplot as plt


#Functions to check if the indeces of a mask are within given ellipses
def ellipse1(row, col):
    if (((row - 550)**2)/(116)**2 + ((col - 550)**2)/(292)**2) < 1:
        return False
    return True

def ellipse2(row, col):
    if (((row - 550)**2)/(181)**2 + ((col - 550)**2)/(409)**2) > 1:
        return False
    return True




a, trg, angles, coords, head = [], [], [], [], []
ref = np.zeros((12, 1100,1100)) #initializing references and masks arrays. Change these sizes according to the number of references and size of images
mask_disk = np.ones((12,1100,1100))
masks_fom = np.ones((12,1100,1100))
cond, counter2, counter = True, 0, 0

trg_bool = True
directory = os.fsencode("###") #input path to directory with ref

ra1, dec1 = (344.4110073, -29.6186750)
ra2, dec2 = (344.4118405, -29.6206386)

#ra1, dec1 = (344.4090073, -29.6186750) #2010
#ra2, dec2 = (344.4118405, -29.6206386)


for x in sorted(os.listdir(directory)):
    temp_file = fits.open(os.path.join(directory, x))
    if cond:
        header = WCS(temp_file[0].header)
        cond = False
    if counter2 == 0:
        #Block of code below modifies mask_fom to make a data imputation mask for exoplanet
        #the RAs and DECs of where the planet is needs to be defined
        """
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
        masks_fom[counter][boty:topy,botx:topx] = 0.
        """
        angles.append(temp_file[0].header['ORIENTAT'])
    
    a.append(temp_file[0].data)
    counter2 += 1
    if counter2 == 28: #this block medians/stacks all of the references based on the visit. Change the counter value based on the number of images per visit (28 img are used for the 2012 and 2013 data)
        counter2 = 0
        ref_min = np.median(np.nanmedian(a, 0))
        ref[counter] = np.nanmedian(a, 0) - ref_min
        coords.append(radonCenter.searchCenter(ref[-1], temp_file[0].header['CRPIX1'], temp_file[0].header['CRPIX2'], size_window = ref.shape[1]/2))
        a.clear()
        counter += 1

    temp_file.close()


#Loop below makes a mask for the disks. Then rotates disk mask to the location of where the disk is in each reference according to the angles given in the .fits file
#This requires the angles for each reference to be known
"""
for z in range(mask_disk.shape[0]):
    for i in range(mask_disk.shape[1]):
        for j in range(mask_disk.shape[2]):
            if ellipse1(i, j) and ellipse2(i, j):
                mask_disk[z][i][j] = 0.
    mask_disk[z] = pyklip.rotate(mask_disk[z], -67, (550,550))
    mask_disk[z] = pyklip.rotate(mask_disk[z], angles[z], (550,550))
mask_disk[np.isnan(mask_disk)] = 1
masks_fom[masks_fom < 0.9] = 0.1
"""



file_mask = fits.open('###') #opens .fits file for mask. Mask needs to be in directory. Can add a .os read to find file if not in same directory
mask = file_mask[0].data


mask[mask >= -1] = 1.
mask[mask < -1] = 0 #mask cleanup


refshape = (ref.shape[0]-1, ref.shape[1], ref.shape[2]) #defining errors and shape of references. Can set these errors to any errors provided
ref_err = np.ones(refshape)
trg_errs = np.ones(ref.shape)


results = np.zeros((ref.shape[0], ref.shape[1], ref.shape[2])) # Say trgs is a 3D array containing the targets that need NMF modeling, then results store the NMF subtraction results.
result2 = np.zeros((ref.shape[0], ref.shape[1], ref.shape[2]))
maxiters = 1e5


#block of code re-centers masks. Loop re-centers masks of the planet used for data imputation
"""
for i in range(ref.shape[0]):
    ref[i] = pyklip.rotate(ref[i], 0, coords[i], new_center=(550,550), astr_hdr=header)
    masks_fom[i] = pyklip.rotate(masks_fom[i], 0, coords[i], new_center=(550,550), astr_hdr=header)
"""
mask = pyklip.rotate(mask, 0, coords[0], new_center=(550,550), astr_hdr=header)


#unifies all masks if data imputation is used
#mask_new = mask_fom * mask_disk * mask


#cleaning up nans and 0s to make it compatible due to NMF having trouble with nans and 0s. 
t0 = time.time()
ref[ref<=0] = 1e-6
ref[np.isnan(ref)] = 1e-6
mask[mask == 0] = 1e-6
mask_disk[mask_disk == 0] = 1e-6
mask[np.isnan(mask)] = 1e-6
masks_fom[np.isnan(masks_fom)] = 1e-6
mask_new[np.isnan(mask_new)] = 1e-6
mask_new[mask_new == 0] = 1e-6


componentNum = 11 #desired number of components. Has to be greater than 2 but less than reference number


#line below builds the components if references are not the same as targets, or if only one target img is desired
#components = nmf_imaging.NMFcomponents(ref, ref_err=b, mask = mask, n_components = componentNum, oneByOne=True, maxiters=maxiters) 


#loop used for NMF_imaging
for i in range(ref.shape[0]-1):
    ref_new = copy.deepcopy(ref)
    ref_new = np.delete(ref_new, i, 0)
    components = nmf_imaging.NMFcomponents(ref_new, ref_err=ref_err, mask = mask, n_components = componentNum, oneByOne=True, maxiters=maxiters)
    trgs = ref[i]
    trg_err = trg_errs[i]
    model = nmf_imaging.NMFmodelling(trg = trgs, trg_err=trg_err, components = components, n_components = componentNum, mask_components=mask, trgThresh=0.0, maxiters=maxiters) # Model the target with the constructed components.
    #best_frac =  nmf_imaging.NMFbff(trgs, model, mask) # Perform BFF procedure to find out the best fraction to model the target.
    best_frac = 1.0 #best_frac set to 1.0 for a planet
    result = nmf_imaging.NMFsubtraction(trgs, model, mask, frac = best_frac) # Subtract the best model from the target
    print(i,"targets calculated")
    results[i] = result
    # Now `results' stores the NMF subtraction results of the targets.


results_new = []

#rotates results of each trg based on angles found in .fits files
for z in range(ref.shape[0]-1):
    if z == 0:
        results_new.append(pyklip.rotate(results[z], angles[z], (550,550), astr_hdr=header))
    else:
        results_new.append(pyklip.rotate(results[z], angles[z], (550,550)))

#getting median and mean of each of the results 
results_median =(np.nanmedian(results_new, 0))
results_mean =(np.nanmedian(results_new, 0))


#timing code
t1 = time.time()
print(t1-t0, "seconds")


#outputing results into .fits files. Header is not translated properly, so need to look into it. 
hdr = header.to_header()

hdu = fits.PrimaryHDU(results_mean, hdr)
hdu.writeto('output_mean.fits', overwrite=True)

hdu = fits.PrimaryHDU(results_median, hdr)
hdu.writeto('output_median.fits', overwrite=True)


file_mask.close()
