
import numpy as np
import pandas as pd
from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_train_test_ids
from pathlib import Path

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
OUT_DIR = Path("/share/klab/labstudents/jmihatsch/processed_data")



def save_avg_fmri_data(subj=1):
    #set up subdirectory inside loop 
    SUB_DIR = OUT_DIR / f"subj{subj:02d}"
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    # Load betas and trial info as usual (see laion_fmri_package/load).
    print(f"Loading fMRI subject {subj}")
    sub = load_subject(f"sub-{subj:02d}")
    sessions = "ses-01", "ses-02"


    #train set
    #get betas for all sessions and concatenate
    print("getting train betas all sessions")
    betas_train  = sub.get_betas(session=sessions, roi="laiongeneral", stimuli="unique")
    print("got betas all session, starting concatentation")
    betas_train  = np.concatenate(list(betas_train.values()), axis=0)
    print("Concatenated betas for all sessions")

    #get all trial info, filter by those starting name with "unique", and concatenate
    trials_train = sub.get_trial_info(session=sessions)
    trials_train = pd.concat(list(trials_train.values()), ignore_index=True)
    trials_train_filtered = trials_train[trials_train['label'].str.startswith('unique')]
    print("Concatenated trial info for all sessions")

    # make a dataframe out of betas and png names to use group.by to then mean the beta
    betas_df = pd.DataFrame(betas_train)
    betas_df['label'] = trials_train_filtered['label'].values
    avg_betas = betas_df.groupby('label').mean()
    print("Averaged beta values for train set")

    #save a np array for the fmri averaged betas and one for the png filenames (maybe change later to already include the stimulus)
    fmri_avg = avg_betas.to_numpy()
    np.save(SUB_DIR / f"laion_train_fmriavg_laiongeneral_sub{subj:02d}.npy", fmri_avg)
    print("Saved fMRI average for train set")

    img_all = []
    for ses in sessions:
        img_ses = sub.images.array(session=ses)   
        img_all.append(img_ses)
    #img_all = np.stack(img_all, axis = 0)
    img_all = np.concatenate(img_all, axis = 0)
    print("testtest", img_all.shape)
    #print(trials_train)
    images_df = pd.DataFrame(img_all)
    images_df['label'] = trials_train['label'].values
    images_df = images_df[~images_df.index.duplicated(keep='first')]

    labels = avg_betas.index.to_numpy()

    train_stim =[]
    for label in labels:
        img = images_df.loc[label]
        train_stim.append(img)
    np.save(SUB_DIR / f"laion_train_stim_sub{subj:02d}.npy", train_stim)
    print("Saved fMRI png data for train set")

'''
    #test set
    #same exact steps but with shared stim across subjects
    print("getting test betas all sessions")
    betas_test =  sub.get_betas(session=sessions, roi="laiongeneral", stimuli="shared")
    print("got betas all session, starting concatentation")
    betas_test  = np.concatenate(list(betas_test.values()), axis=0)
    print("Concatenated beta values for test set")

    trials_test = sub.get_trial_info(session=sessions)
    trials_test = pd.concat(list(trials_test.values()), ignore_index=True)
    trials_test = trials_test[trials_test['label'].str.startswith('shared')]
    print("Concatenated trial data for test set")

    betas_df_test = pd.DataFrame(betas_test)
    betas_df_test['label'] = trials_test['label'].values
    avg_betas_test = betas_df_test.groupby('label').mean()
    print("Averaged beta values for test set")

    fmri_avg_test = avg_betas_test.to_numpy()
    np.save(SUB_DIR / f"laion_test_fmriavg_laiongeneral_sub{subj:02d}.npy", fmri_avg_test)
    print("Saved fMRI average test set")
    labels_test = avg_betas_test.index.to_numpy()
    np.save(SUB_DIR / f"laion_test_stim_sub{subj:02d}.npy", labels_test)
    print("saved stim label for test set")
'''
save_avg_fmri_data(subj=1)


