from PIL import Image
from transformers import AutoProcessor, CLIPModel
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.append("../")
sys.path.append("../../")
from autoencoder import *
import tqdm

project = PROJ_ROOT = Path(__file__).resolve().parents[1]
username = project.name
data_path = Path(f"/share/klab/labstudents/{username}")


def encode_images(subj):
    processed_data_path = data_path / "processed_data" 
    processed_data_subject = processed_data_path / f"subj{subj:02d}"

    embeds_path = data_path / "embeds"
    embeds_path.mkdir(parents=True, exist_ok=True)

    train_stim = processed_data_subject / f"laion_train_stim_sub{subj:02d}.npy"
    train_imgs = np.load(train_stim)
    #train_imgs = train_imgs[0:10]   #xxx

    device="cuda"  
    discriminator = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ## encode all the images in batch with clip
    BS=128

    img_embeds=[]
    with torch.no_grad():
        for batch_idx in tqdm.trange(0,len(train_imgs)//BS+1,1):
            batch = train_imgs[batch_idx*BS:(batch_idx+1)*BS]
            batch = [Image.fromarray(i) for i in batch]
            # apply processor
            inputs = processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k:v.to(device) for k,v in inputs.items()}  #inputs = {k:v for k,v in inputs.items()}   
            emb = discriminator.get_image_features(inputs["pixel_values"]).cpu()
            img_embeds.append(emb)
    
    img_embeds = torch.cat(img_embeds,0)
    torch.save(img_embeds, embeds_path / f"image_embeds_subject-{subj:02d}.pt")

# for subj in (1,3,5,6,7):
#     encode_images(subj)

encode_images(subj=1)

## Source - https://stackoverflow.com/a/76218591
# Posted by Timbus Calin
# Retrieved 2026-10-08, License - CC BY-SA 4.0
# pip install requests==2.27.1   

#salloc -p klab-gpu -x klab-8 -n 1 -c 1 --mem 300G --gres=gpu:1 -t 04:00:00 srun --pty bash
#salloc -p klab-gpu -x klab-7,klab-8 -n 1 -c 4 --mem 32G --gres=gpu:1 -t 04:00:00 srun --pty bash