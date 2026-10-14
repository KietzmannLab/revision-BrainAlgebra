
import numpy as np
import pandas as pd
from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_train_test_ids
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from datetime import datetime


'''
argument: subject number
betas are averaged for each trial in laiongeneral ROI and saved as train and test numpy files
train set -> average of betas for unique trials (pictures only shown to that participant)
test set -> average of betas for shared trials (pictures shown to all participants)
image array are extracted per trial, resized to fit BrainAlgebra data, and saved as numpy file with order == fmri
'''

#set up directories
DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
dataset_initialize(DATA_DIR)

project = PROJ_ROOT = Path(__file__).resolve().parents[1]
username = project.name
data_path = Path(f"/share/klab/labstudents/{username}")
OUT_DIR = data_path / "processed_data"

def resize_images(img_all):   #img_all = (trial, dim, dim, rgb)
    new_height = 425
    new_width = 425
    new_img=[]

    for i in range(len(img_all)):
        image = img_all[i]
        target_shape = (new_height, new_width, image.shape[2])
        resized_np_array_float = resize(image, target_shape, anti_aliasing=True)
        resized_np_array_uint8 = (resized_np_array_float * 255).astype(np.uint8)
        new_img.append(resized_np_array_uint8)
    img_all_new = np.stack(new_img, axis=0)
    return img_all_new
    

def make_img_dictionaries(sub, sessions, trials_info):
    img_all = []
    for ses in sessions:
        img_ses = sub.images.array(session=ses) #(trials, dim, dim, rgb)
        img_all.append(img_ses)
    img_all = np.concatenate(img_all, axis = 0)
    print(f"[INFO] Collected all Image data, shape: {img_all.shape}")
    img_all = resize_images(img_all)
    print(f"[INFO] Resized Image data, new shape: {img_all.shape}")

    labels = trials_info['label']
    unique_dictionary = {} 
    shared_dictionary = {}
    for n in range(len(labels)):
        if labels[n].startswith('unique'):
            unique_dictionary[labels[n]] = img_all[n,:,:,:]
        elif labels[n].startswith('shared'):
            shared_dictionary[labels[n]] = img_all[n,:,:,:]
        else:
            print("labels different starting characters")
    return unique_dictionary, shared_dictionary


def save_avg_fmri_data(subj=1):
    #set up subdirectory inside loop 
    starttime = datetime.now()
    print("[INFO] Start Time:", starttime)
    SUB_DIR = OUT_DIR / f"subj{subj:02d}"
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    # Load betas and trial info as usual (see laion_fmri_package/load).
    sub = load_subject(f"sub-{subj:02d}")
    sessions = sub.get_sessions()
    #sessions = "ses-01", "ses-02"
    print(f"[INFO] Loading LAION-fMRI data for Subject {subj}\n\tSessions: {', '.join(sessions)}")

    #train set
    #get betas for all sessions and concatenate
    print(f"[INFO] Getting betas for train stimuli: unique, mask: laiongeneral")
    betas_train  = sub.get_betas(session=sessions, roi="laiongeneral", stimuli="unique")
    betas_train  = np.concatenate(list(betas_train.values()), axis=0)
    print(f"[INFO] Collected all Betas, shape: {betas_train.shape}")
    #trials_train, voxels_train = betas_train.shape
    #print(f"[INFO] Collected all Betas, trials: {trials_train}, voxels: {voxels_train}")

    #get all trial info
    trials_info = sub.get_trial_info(session=sessions)
    trials_info = pd.concat(list(trials_info.values()), ignore_index=True)

    #filter by those starting name with "unique", and concatenate
    trials_train = trials_info[trials_info['label'].str.startswith('unique')]
    print(f"[INFO] Collected unique trial info for all sessions, trials: {len(trials_train)}")

    # make a dataframe out of betas and png names to use group.by to then mean the beta
    betas_train_df = pd.DataFrame(betas_train)
    betas_train_df['label'] = trials_train['label'].values
    avg_betas_train = betas_train_df.groupby('label').mean()
    print(f"[INFO] Averaged beta values for train data, trials: {len(avg_betas_train)}")

    #save a np array for the fmri averaged betas and one for the png filenames (maybe change later to already include the stimulus)
    fmri_avg_train = avg_betas_train.to_numpy()
    np.save(SUB_DIR / f"laion_train_fmriavg_laiongeneral_sub{subj:02d}.npy", fmri_avg_train)
    print("[INFO] Saved averaged fMRI activity for train data")

    unique_dictionary, shared_dictionary = make_img_dictionaries(sub=sub, sessions=sessions, trials_info=trials_info)
    print(f"[INFO] Built image dictionaries \n\tunique: {len(unique_dictionary)} \n\tshared: {len(shared_dictionary)}")

    labels_order = avg_betas_train.index.to_numpy()
    img_data = []
    for name in labels_order:
        img_array = unique_dictionary[name]
        img_data.append(img_array)
    img_data = np.stack(img_data, axis = 0)
    print("[INFO] Aligned image data to fMRI trial order")

    np.save(SUB_DIR / f"laion_train_stim_sub{subj:02d}.npy", img_data)
    print("[INFO] Saved image data for train data")

    #test set
    #same exact steps but with shared stim across subjects
    print(f"[INFO] Getting betas for test stimuli: shared, mask: laiongeneral")
    betas_test =  sub.get_betas(session=sessions, roi="laiongeneral", stimuli="shared")
    betas_test  = np.concatenate(list(betas_test.values()), axis=0)
    print(f"[INFO] Collected all Betas, shape: {betas_test.shape}")

    trials_test = trials_info[trials_info['label'].str.startswith('shared')]
    print(f"[INFO] Collected shared trial info for all sessions, trials: {len(trials_test)}")

    betas_test_df = pd.DataFrame(betas_test)
    betas_test_df['label'] = trials_test['label'].values
    avg_betas_test = betas_test_df.groupby('label').mean()
    print(f"[INFO] Averaged beta values for test data, trials: {len(avg_betas_test)}")

    fmri_avg_test = avg_betas_test.to_numpy()
    np.save(SUB_DIR / f"laion_test_fmriavg_laiongeneral_sub{subj:02d}.npy", fmri_avg_test)
    print("[INFO] Saved averaged fMRI activity for test data")

    labels_order = avg_betas_test.index.to_numpy()
    img_data = []
    for name in labels_order:
        img_array = shared_dictionary[name]
        img_data.append(img_array)
    img_data = np.stack(img_data, axis = 0)
    print("[INFO] Aligned image data to fMRI trial order")

    np.save(SUB_DIR / f"laion_test_stim_sub{subj:02d}.npy", img_data)
    print("[INFO] Saved image data for test data")
    endtime = datetime.now()
    print("End Time:", endtime)
    total_time = endtime - starttime
    print("[INFO] Total Time:", total_time)


#save_avg_fmri_data(subj=1)



#optional cahnges
# - make mask an argument
# - make session an argument