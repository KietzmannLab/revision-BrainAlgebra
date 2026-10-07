import numpy as np
import pandas as pd
from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_train_test_ids

DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
dataset_initialize(DATA_DIR)

sub = load_subject("sub-01")

# Load betas and trial info as usual (see laion_fmri_package/load).
sessions = sub.get_sessions()
betas_per_ses  = sub.get_betas(session=sessions, roi="laiongeneral")
trials_per_ses = sub.get_trial_info(session=sessions)

# Concatenate across sessions (standard idiom).
betas  = np.concatenate(list(betas_per_ses.values()), axis=0)
trials = pd.concat(list(trials_per_ses.values()), ignore_index=True)
label = trials['label'] 

# print(betas.shape)
# print(len(trials))
# print(trials.shape)

# Method 1: within-distribution generalization on the shared pool.
train_ids, test_ids = get_train_test_ids("tau", pool="shared")
train_mask = trials["label"].isin(train_ids).to_numpy()
test_mask  = trials["label"].isin(test_ids).to_numpy()

# X_train, X_test = betas[train_mask], betas[test_mask]



'''roi_dir = 'nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub)
betas_dir = 'nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub)

mask_filename = 'nsdgeneral.nii.gz'
mask = nib.load(roi_dir+mask_filename).get_fdata()
num_voxel = mask[mask>0].shape[0]

fmri = np.zeros((num_trials, num_voxel)).astype(np.float32)
for i in range(37):
    beta_filename = "betas_session{0:02d}.nii.gz".format(i+1)
    beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32)
    fmri[i*750:(i+1)*750] = beta_f[mask>0].transpose()
    del beta_f
    print(i)
    
print("fMRI Data are loaded.")

f_stim = h5py.File('nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
stim = f_stim['imgBrick'][:]

print("Stimuli are loaded.")

num_train, num_test = len(train_im_idx), len(test_im_idx)
vox_dim, im_dim, im_c = num_voxel, 425, 3
fmri_array = np.zeros((num_train,vox_dim))
stim_array = np.zeros((num_train,im_dim,im_dim,im_c))
for i,idx in enumerate(train_im_idx):
    stim_array[i] = stim[idx]
    fmri_array[i] = fmri[sorted(sig_train[idx])].mean(0)
    print(i)

np.save('processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub),fmri_array )
np.save('processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub),stim_array )


fmri_array = np.zeros((num_test,vox_dim))
stim_array = np.zeros((num_test,im_dim,im_dim,im_c))
for i,idx in enumerate(test_im_idx):
    stim_array[i] = stim[idx]
    fmri_array[i] = fmri[sorted(sig_test[idx])].mean(0)
    print(i)

np.save('processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub),fmri_array )
np.save('processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub),stim_array )

print("Test data is saved.")
'''