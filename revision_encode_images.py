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
import pandas as pd
import nibabel as nib
from scipy.io import loadmat
import torch
import sys
sys.path.append("../")
sys.path.append("../../")
from torch.utils.data import Dataset, Subset, DataLoader
import json
from autoencoder import *
import tqdm

#DATA_DIR = "/share/klab/datasets/optimized_datasets/laion_fmri_data"
#dataset_initialize(DATA_DIR)

data_path = Path("/share/klab/labstudents/jmihatsch/processed_data")
out_dir = Path("/share/klab/labstudents/jmihatsch/embeds")
#data_path = Path("/home/student/j/jmihatsch/revision-BrainAlgebra/data/processed_data")
#out_dir = Path("/home/student/j/jmihatsch/revision-BrainAlgebra/data/embeds")
out_dir.mkdir(parents=True, exist_ok=True)

subj = 1   #making subject into variable to possibly use later as function
processed_data = data_path / f"subj{subj:02d}"

labels_train = opj(processed_data, f"laion_train_stim_sub{subj:02d}.npy")
train_imgs = np.load(labels_train)
train_imgs = train_imgs[0:3]   #xxx

#device="cuda:1"  #"CUDA is a parallel computing platform and programming model developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU" (docs.nvidia.com)
discriminator = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") #.device
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

## encode all the images in batch with clip
BS=128

img_embeds=[]
with torch.no_grad():
    for batch_idx in tqdm.trange(0,len(train_imgs)//BS+1,1):
        print("check 1")
        batch = train_imgs[batch_idx*BS:(batch_idx+1)*BS]
        batch = [Image.fromarray(i) for i in batch]
        print("check 2")
        # apply processor
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k:v for k,v in inputs.items()}   #inputs = {k:v.to(device) for k,v in inputs.items()}
        print("check 3")
        emb = discriminator.get_image_features(inputs["pixel_values"]).cpu()
        img_embeds.append(emb)
        print("check 4")
 

img_embeds = torch.cat(img_embeds,0)
torch.save(img_embeds, out_dir / f"image_embeds_subject-{subj:02d}.pt")

## Source - https://stackoverflow.com/a/76218591
# Posted by Timbus Calin
# Retrieved 2026-10-08, License - CC BY-SA 4.0
# pip install requests==2.27.1   
