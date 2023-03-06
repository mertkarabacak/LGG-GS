import os
import re
import numpy as np
import nibabel as nib
from scipy import ndimage
from configs import *


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    absmin = -1000
    absmax = 1000
    volume[volume < absmin] = absmin
    volume[volume > absmax] = absmax
    volume = (volume - np.nanmin(volume)) / (np.nanmax(volume) - np.nanmin(volume))
    volume = volume.astype("float32")
    return volume


def resize_volume(img, depth, width, height):
    """Resize across z-axis"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth_f = current_depth / depth
    width_f = current_width / width
    height_f = current_height / height
    depth_factor = 1 / depth_f
    width_factor = 1 / width_f
    height_factor = 1 / height_f
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


### Preprocess Niftis if necessary

def process_scan(path):
    volume = read_nifti_file(path)
    #volume = normalize(volume)
    #volume = resize_volume(volume, depth, width, height)
    return volume


###  Get Nifti file paths

rx = ['.*t1_dicom_segmented\.nii\.gz$',
      '.*t2_dicom_segmented\.nii\.gz$',
      '.*t1c_dicom_segmented\.nii\.gz$',
      '.*fl_dicom_segmented\.nii\.gz$']
paths = [[], [], [], [], []]
patient_ids = []

rootpath = "./data/segmented-normalized/"
for root, dirs, files in os.walk(""): #
    for dir in dirs:
        patient_ids.append(dir)
        for root2, dirs2, files2 in os.walk(rootpath + dir):
            for file in files2:
                for i, str in enumerate(rx):
                    res = re.match(str, file)
                    if res:
                        paths[i].append(rootpath + dir + "/" + file)

print(patient_ids)

### Write patient names to txt so that clinic data can be retrieved in the same order.

with open('patients.txt', 'w') as filehandle:
    for id in patient_ids:
        filehandle.write('%s\n' % id)

### Handle multiple channel data (different mri types)

for i, mri_type in enumerate(mri_types):
    if i == 0:
        scans = np.array([process_scan(path) for path in paths[mri_type.value]])
        scans = np.expand_dims(scans, axis=4)
    else:
        new_scan = np.array([process_scan(path) for path in paths[mri_type.value]])
        new_scan = np.expand_dims(new_scan, axis=4)
        scans = np.concatenate([scans, new_scan], axis=4)

### Separate data intp train and test sets
x_train = scans[val_set_size:]
x_val = scans[:val_set_size]

### Save data
np.save("np/x_train", x_train)
np.save("np/x_val", x_val)


