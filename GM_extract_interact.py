# This code is for 90 aal segmentation in fMRI

## Load libraries
import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img

template = load_mni152_template()
saving_path = input("\n Giving the output directory: ")
roi_path = input("\n Where are the rois? ")
roi_extension = input("\n And what's the roi extension? ")
data_path = input("\n Where's the data? ")
file_extension = input("\n And what's the data extension? ")

#Define a function for data resampling
def resample(dirr, file_extension):
    array_list = []
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort()
    for file in path_list:
        if file.endswith(file_extension):
            img = nib.load(os.path.join(dirr, file))
            resampled_img = resample_to_img(img, template)
            img_array = resampled_img.get_fdata()
            img_array = img_array[None,...]
            array_list.append(img_array)
    imgs_array = np.concatenate(array_list, axis=0)
    return path_list, imgs_array

# load 90 ROI templates
def fileread(dirr, file_extension):
    array_list = []
    path_list=[f for f in os.listdir(dirr) if not f.startswith('.')]
    path_list.sort() #對讀取的路徑進行排序
    for file in path_list:
        if file.endswith(file_extension):
            img = nib.load(os.path.join(dirr, file))
            img_array = img.get_fdata()
            img_array = img_array[None,...]
            array_list.append(img_array)
    imgs_array = np.concatenate(array_list, axis=0)
    return path_list, imgs_array


# segment 90 cerebral subregions based on AAL ROI
def segment90(samples, rois):
    numbers = 0
    whole_brain = []
    subjects = []
    for sample in samples:
        for roi in rois:
            Area = np.multiply(sample,roi) #entrywise product
            Area = Area[Area>0]
            mean = Area.mean()
            #mean = mean.reshape(1)
            whole_brain.append(mean)
            numbers += 1
        subjects.append(whole_brain)
        whole_brain = []
    #print("input subject shape: ", subjects.shape)
    return subjects



fileName, first = fileread(data_path, file_extension)
roiName, rois = fileread(roi_path, roi_extension)


whole_data = segment90(first,rois)

df = pd.DataFrame(whole_data, columns = roiName, index = fileName)

outputName = input("\n \033[93mName of the output? \033[00m")

os.chdir(saving_path)
df.to_csv(f"{outputName}.csv")


print(f'saving completed, check the output file at {saving_path} for the result')