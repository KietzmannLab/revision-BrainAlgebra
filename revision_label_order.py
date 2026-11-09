import numpy as np
import pandas as pd
from laion_fmri.config import dataset_initialize
from laion_fmri.subject import load_subject
from pathlib import Path
from skimage.transform import resize
from datetime import datetime
import pickle
from config import *


subj=7
#set up directories
DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
dataset_initialize(DATA_DIR)

sub = load_subject(f"sub-{subj:02d}")
sessions = sub.get_sessions()
trials_info = sub.get_trial_info(session=sessions)
trials_info = pd.concat(list(trials_info.values()), ignore_index=True)

shared = trials_info[trials_info["label"].str.startswith("shared")]["label"].unique()
ordered = np.array(sorted(shared))         # your saved order

# save
with open(processed_data_path(subj) / "labels_shared.pkl", "wb") as f:
    pickle.dump({
        "originalorder": shared,
        "ordered": ordered
    }, f)


orig_idx_map = {label: i for i, label in enumerate(shared)}
img_idx = 114
label = ordered[img_idx]
orig_idx = orig_idx_map[label]
print("label:", label)
print("original index:", orig_idx)