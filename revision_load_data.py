
import numpy as np
import pandas as pd
from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_train_test_ids
from pathlib import Path
from PIL import Image
from skimage.transform import resize

'''
argument: subject number
betas are averaged for each trial in laiongeneral ROI and saved as train and test numpy files
train set -> average of betas for unique trials (pictures only shown to that participant)
test set -> average of betas for shared trials (pictures shown to all participants)
stimulus data is currently a string of the png labels
'''

#set up directories
DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
dataset_initialize(DATA_DIR)

project = PROJ_ROOT = Path(__file__).resolve().parents[1]
username = project.name
data_path = Path(f"/share/klab/labstudents/{username}")
OUT_DIR = data_path / "processed_data"


def resize_images(img_all):   #img_all = (trial, dim, dim, rgb)
    new_height, new_width = 425
    target_shape = (img_all.shape[0], new_height, new_width, img_all.shape[3])
    resized_np_array_float = resize(img_all, target_shape, anti_aliasing=True)
    resized_np_array_uint8 = (resized_np_array_float * 255).astype(np.uint8)

    return resized_np_array_uint8
    


def make_img_dictionaries(sub, sessions, trials_info):
    img_all = []
    for ses in sessions:
        img_ses = sub.images.array(session=ses) #(trials, dim, dim, rgb)
        img_all.append(img_ses)
    img_all = np.concatenate(img_all, axis = 0)

    img_all = resize_images(img_all)

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
    SUB_DIR = OUT_DIR / f"subj{subj:02d}"
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    # Load betas and trial info as usual (see laion_fmri_package/load).
    print(f"[INFO] Loading LAION-fMRI data for Subject {subj}")
    sub = load_subject(f"sub-{subj:02d}")
    #sessions = sub.get_sessions()
    sessions = "ses-01", "ses-02"

    #train set
    #get betas for all sessions and concatenate
    print("[INFO] Getting betas for unique stimuli")
    betas_train  = sub.get_betas(session=sessions, roi="laiongeneral", stimuli="unique")
    print("[INFO] Collected all Betas")
    betas_train  = np.concatenate(list(betas_train.values()), axis=0)
    print("[INFO] Concatenated Betas for all sessions")

    #get all trial info, filter by those starting name with "unique", and concatenate
    trials_info = sub.get_trial_info(session=sessions)
    trials_info = pd.concat(list(trials_info.values()), ignore_index=True)

    trials_train = trials_info[trials_info['label'].str.startswith('unique')]
    print("[INFO] Concatenated training data trial info for all sessions")

    # make a dataframe out of betas and png names to use group.by to then mean the beta
    betas_train_df = pd.DataFrame(betas_train)
    betas_train_df['label'] = trials_train['label'].values
    avg_betas_train = betas_train_df.groupby('label').mean()
    print("[INFO] Averaged beta values for training data")

    #save a np array for the fmri averaged betas and one for the png filenames (maybe change later to already include the stimulus)
    fmri_avg_train = avg_betas_train.to_numpy()
    np.save(SUB_DIR / f"laion_train_fmriavg_laiongeneral_sub{subj:02d}.npy", fmri_avg_train)
    print("[INFO] Saved averaged fMRI activity for training data")

    unique_dictionary, shared_dictionary = make_img_dictionaries(sub=sub, sessions=sessions, trials_info=trials_info)

    labels_order = avg_betas_train.index.to_numpy()
    img_data = []
    for name in labels_order:
        img_array = unique_dictionary[name]
        img_data.append(img_array)
    img_data = np.stack(img_data, axis = 0)

    np.save(SUB_DIR / f"laion_train_stim_sub{subj:02d}.npy", img_data)
    print("[INFO] Saved image data for train set in correct order")

    #test set
    #same exact steps but with shared stim across subjects
    print("[INFO] Getting betas for shared stimuli")
    betas_test =  sub.get_betas(session=sessions, roi="laiongeneral", stimuli="shared")
    print("[INFO] Collected all Betas")
    betas_test  = np.concatenate(list(betas_test.values()), axis=0)
    print("[INFO] Concatenated Betas for all sessions")

    trials_test = trials_info[trials_info['label'].str.startswith('shared')]
    print("[INFO] Concatenated test data trial info for all sessions")

    betas_test_df = pd.DataFrame(betas_test)
    betas_test_df['label'] = trials_test['label'].values
    avg_betas_test = betas_test_df.groupby('label').mean()
    print("[INFO] Averaged beta values for test data")

    fmri_avg_test = avg_betas_test.to_numpy()
    np.save(SUB_DIR / f"laion_test_fmriavg_laiongeneral_sub{subj:02d}.npy", fmri_avg_test)
    print("[INFO] Saved averaged fMRI activity for test data")

    labels_order = avg_betas_test.index.to_numpy()
    img_data = []
    for name in labels_order:
        img_array = shared_dictionary[name]
        img_data.append(img_array)
    img_data = np.stack(img_data, axis = 0)

    np.save(SUB_DIR / f"laion_test_stim_sub{subj:02d}.npy", img_data)
    print("[INFO] Saved image data for train set in correct order")


save_avg_fmri_data(subj=1)


