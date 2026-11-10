import numpy as np
import pandas as pd
from laion_fmri.config import dataset_initialize
from laion_fmri.subject import load_subject
from pathlib import Path
from skimage.transform import resize
from datetime import datetime
import nibabel as nib
import nilearn
import nilearn.image
import nilearn.plotting
import numpy as np
import nibabel as nib
import torch
import tqdm
from pathlib import Path
from os.path import join as opj
import random
from config import *


subj = 1
random.seed(42)

test_fmri = np.load(processed_data_path(subj) / f"laion_test_fmriavg_laiongeneral_sub{subj:02d}.npy")
test_imgs = np.load(processed_data_path(subj) / f"laion_test_stim_sub{subj:02d}.npy")

indices = sorted(random.sample(range(len(test_fmri)), 100))
stims = test_imgs[indices]

fmri_test_data = test_fmri[indices]

print(test_fmri.mean(0)[:5], test_fmri.std(0)[:5])

DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
dataset_initialize(DATA_DIR)

tgt_dir = subject_path(ROI_PATH, subj)
plot_dir = subject_path(ROI_PLOTS, subj)

sub = load_subject(f"sub-{subj:02d}")



roi_mask = sub.get_roi_mask("laiongeneral")
roi_file = tgt_dir / "laiongeneral_roi.nii.gz"
sub.to_nifti(roi_mask.astype(np.float32), roi_file)

roi_img = nib.load(roi_file)
roi = roi_img.get_fdata()
print(roi.shape, np.unique(roi), int(roi.sum()), int(roi_mask.sum()))
coordinates=np.array(np.where(roi==1)).T

nsdroi = nilearn.image.math_img("img > 0", img=roi_img)
img_mean = roi_img
data = roi
vox_mm = sub.get_voxel_coordinates()
vox = np.round(nib.affines.apply_affine(np.linalg.inv(roi_img.affine), vox_mm)).astype(int)

train_fmri = np.load(processed_data_path(subj) / f"laion_train_fmriavg_laiongeneral_sub{subj:02d}.npy")
mu = train_fmri.mean(0)
sd = train_fmri.std(0)  + 1e-8 
del train_fmri

base_pattern = (fmri_test_data[0] - mu) / sd

base_map=np.zeros(data.shape)

for v,(x,y,z) in enumerate(coordinates):
    base_map[x,y,z]=base_pattern[v]

base_map=nib.Nifti1Image(base_map,affine=img_mean.affine)
base_map=nilearn.image.threshold_img(base_map,threshold=0,cluster_threshold=0,mask_img=nsdroi)

vals = base_map.get_fdata()[vox[:,0], vox[:,1], vox[:,2]].astype(np.float32)

print("vox range:", vox.min(0), vox.max(0))
print("coords match:", np.array_equal(vox[roi_mask], coordinates))
print("base_pattern nonzero:", np.count_nonzero(base_pattern), base_pattern.min(), base_pattern.max())
print("base_map nonzero:", np.count_nonzero(base_map.get_fdata()))
print("vals nonzero:", np.count_nonzero(vals))


nib.save(sub.to_template(vals, "MNI305"), opj(tgt_dir, "base_concept.nii.gz"))


display = nilearn.plotting.plot_glass_brain(opj(tgt_dir, "base_concept.nii.gz"))
display.savefig(opj(plot_dir, f"base_pattern.png"))


base_MNI = nib.load(opj(tgt_dir, "base_concept.nii.gz"))



target_thr = 90
concepts = ["man","woman","indoor scene","outdoor scene","day","night","summer","winter","sad","happy","crowded","empty"]
pert_dir = subject_path(FMRI_ALGEBRA_PATH, subj)

for concept in tqdm.tqdm(concepts):
    pert = torch.load(opj(pert_dir,f"{concept}_thr_{target_thr}",f"diff_99.pt")) ## pert are same for all base patterns
    pertn = np.asarray(pert).ravel()
    print(concept, pertn.shape, pertn.min(), pertn.max(), np.count_nonzero(pertn), pertn[:3])
    thresholded_map=np.zeros(data.shape)

    for v,(x,y,z) in enumerate(coordinates):
        thresholded_map[x,y,z]=pert[v]

    thresholded_map=nib.Nifti1Image(thresholded_map,affine=img_mean.affine)
    thresholded_map=nilearn.image.threshold_img(thresholded_map,threshold=0,cluster_threshold=20,mask_img=nsdroi)

    vals = thresholded_map.get_fdata()[vox[:,0], vox[:,1], vox[:,2]].astype(np.float32)
    nib.save(sub.to_template(vals, "MNI305"), opj(tgt_dir, f"{concept}_thr_{90}.nii.gz"))

    added = nib.load(opj(tgt_dir,f"{concept}_thr_{90}.nii.gz")) 

    final_sum = nilearn.image.math_img("img1+4*img2",img1=base_MNI,img2=added)

    nib.save(final_sum,opj(tgt_dir,f"FINAL_{concept}_thr_{90}.nii.gz"))


    display = nilearn.plotting.plot_glass_brain(opj(tgt_dir,f"{concept}_thr_{90}.nii.gz"),title=concept, plot_abs=False, cmap = "cold_white_hot")
    display.savefig(opj(plot_dir, f"pertubation_img_{concept}.png"))

    display.close()

    display = nilearn.plotting.plot_glass_brain(opj(tgt_dir,f"FINAL_{concept}_thr_{90}.nii.gz"),title=concept, plot_abs=False, cmap = "cold_white_hot")
    display.savefig(opj(plot_dir, f"FINAL_img_{concept}.png"))

    display.close()

print("done")

