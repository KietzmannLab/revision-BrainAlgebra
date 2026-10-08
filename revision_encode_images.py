import laion_fmri
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_train_test_ids
from pathlib import Path
import pickle
import seaborn as sns
import string
import numpy as np
import os
import glob
from os.path import join as opj
import h5py  
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
import torch
from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_train_test_ids
import numpy as np
import pandas as pd
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import laion_fmri
import pickle
import sys
sys.path.append("../")
sys.path.append("../../")


import seaborn as sns
import string
import numpy as np
import os
import glob
from os.path import join as opj
import h5py  
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
import torch

from torch.utils.data import Dataset, Subset, DataLoader
import json
from PIL import Image

from autoencoder import *
#from torchsummary import summary
#import torchvision
#import tqdm
#from sklearn.linear_model import Ridge
import pickle
#import wandb
from pathlib import Path
import tqdm


DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
dataset_initialize(DATA_DIR)
data_path = Path("/share/klab/labstudents/jmihatsch/processed_data")

subj = 1   #making subject into variable to possibly use later as function
subject = f"sub-0{subj}"
processed_data = data_path / f"subj{subj:02d}"
labels_train = opj(processed_data, f"laion_train_stim_sub{subj:02d}.npy")
train_imgs = np.load(labels_train)
print("check 1")

#device="cuda:1"  #"CUDA is a parallel computing platform and programming model developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU" (docs.nvidia.com)
discriminator = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") #.device
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("check 2")
# train_imgs=np.load(opj(processed_data,f"nsd_train_stim_sub{sub_idx}.npy")).astype(np.uint8)
# train_fmri= np.load(fmri_train_data)

#stim = laion_fmri.load_stimuli()
## encode all the images in batch with clip
BS=2  #128

img_embeds=[]
with torch.no_grad():
    for batch_idx in tqdm.trange(0,len(train_imgs)//BS+1,1):
        print("check 3")
        batch = train_imgs[batch_idx*BS:(batch_idx+1)*BS]
        batch = [Image.fromarray(i) for i in batch]
        print("check 4")
        # apply processor
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k:v for k,v in inputs.items()}   #inputs = {k:v.to(device) for k,v in inputs.items()}
        print("check 5")
        emb = discriminator.get_image_features(inputs["pixel_values"]).cpu()
        img_embeds.append(emb)
        print("check 6")
 
       
img_embeds = torch.cat(img_embeds,0)


## Source - https://stackoverflow.com/a/76218591
# Posted by Timbus Calin
# Retrieved 2026-10-08, License - CC BY-SA 4.0
# pip install requests==2.27.1   
