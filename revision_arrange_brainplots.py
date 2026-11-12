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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from config import *

subj=1
plot_dir = subject_path(ROI_PLOTS, subj)

concepts = ["man","woman","crowded","empty", "summer","winter","happy" ,"sad", "indoor scene","outdoor scene","day","night"]

fig, axes = plt.subplots(6, 2, figsize=(12, 15))
axes = axes.ravel(order="F")

for ax, concept in zip(axes, concepts):
    img = mpimg.imread(plot_dir / f"pertubation_img_{concept}.png")
    ax.imshow(img)
    ax.axis("off")

fig.tight_layout()
fig.savefig(plot_dir / "ALL_perturbations.png", dpi=200, bbox_inches="tight")
plt.close(fig)