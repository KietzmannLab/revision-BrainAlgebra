import pickle
import sys
sys.path.append("../")
from hparams import HParams
from hps import Hyperparams
from vae import VAE

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.manifold import TSNE
import seaborn as sns
import nltk
from nltk.corpus import stopwords
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
# import clip
from PIL import Image
# from diffusers import VersatileDiffusionPipeline
# from diffusers import VersatileDiffusionDualGuidedPipeline
from diffusers.models import AutoencoderKL, DualTransformer2DModel, Transformer2DModel, UNet2DConditionModel
from versatile_diffusion_dual_guided import VersatileDiffusionDualGuidedPipeline
from versatile_diffusion_dual_guided_fake_images import VersatileDiffusionDualGuidedFromCLIPEmbeddingPipeline
import accelerate
from autoencoder import *
from torchsummary import summary
import torchvision
import tqdm
from sklearn.linear_model import Ridge
import pickle
import wandb
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from encoding_models import *
import subprocess
import sys

try:
    import accelerate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'accelerate'])
finally:
    import accelerate
    

class BrainDiffuserPretrainedDecoder:
    def __init__(self,vae_weights="/home/matteo/models/vdvae/vae2.pt",
                 vae_hyper='/home/matteo/models/vdvae/H.sav', 
                 pretrained=True,
                 subj_path=None,
                 device="cpu"):
        super().__init__()
        self.keep=31
        self.device=device
        self.pretrained=pretrained
        self.subj_path=subj_path

        print("Loading pretrained deep learning backbones")

        with open(vae_hyper, 'rb') as fp:
            d = pickle.load(fp)

        H=Hyperparams()
        for k,v in d.items():
            H[k]=v
            
        vae=VAE(H)    
        state_dict = torch.load(vae_weights)
        new_state_dict = {}
        l = len('module.')
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
        vae.load_state_dict(state_dict)


        self.vae=vae.to(device)


        self.pipe_embed= VersatileDiffusionDualGuidedFromCLIPEmbeddingPipeline.from_pretrained("shi-labs/versatile-diffusion",)

        self.pipe_embed.remove_unused_weights()
        self.pipe_embed.to(self.device)

        if self.pretrained:
            assert self.subj_path is not None, "Please provide a valid subject path, whith decoding dir and related files"
            print("Loading pretrained brain to feature models")

            keys=np.arange(self.keep)
            # filename='brain_to_latent_ridge.sav'
            self.brain_to_latent = {}
        #     pickle.load(open(opj(f"models/{sub}/decoding",filename), 'rb'))


            self.brain_to_img_emb=[]
            self.brain_to_txt_emb=[]          
            
            print("loading brain to latent models")
            for k in keys:
                filename = f'brain_to_vdvae_latent_ridge_{k}.sav'
                p=pickle.load(open(opj(self.subj_path,"decoding",filename), 'rb'))
                self.brain_to_latent[k]=p

            print("loading brain to img embeddings models")
            for i in range(257):
                filename = f'brain_to_img_emb_ridge_{i}.sav'
                p=pickle.load(open(opj(self.subj_path,"decoding",filename), 'rb'))
                self.brain_to_img_emb.append(p)
            
            print("loading brain to txt embeddings models")
            for i in range(77):
                filename = f'brain_to_txt_emb_ridge_{i}.sav'
                p=pickle.load(open(opj(self.subj_path,"decoding",filename), 'rb'))
                self.brain_to_txt_emb.append(p)

            print("loading adjust values")
            filename = f'latent_adjust_values.sav'
            with open(opj(self.subj_path,filename), 'rb') as f:
                self.latent_adjust_values=pickle.load(f)

            self.clip_img_embeds_mean=torch.load(opj(self.subj_path,"clip_img_embeds_mean.pt"))
            self.clip_img_embeds_std=torch.load(opj(self.subj_path,"clip_img_embeds_std.pt"))


            self.clip_txt_embeds_mean=torch.load(opj(self.subj_path,"clip_txt_embeds_mean.pt"))
            self.clip_txt_embeds_std=torch.load(opj(self.subj_path,"clip_txt_embeds_std.pt"))
            
            print("loading predicted values for adjusting")
            
            img_emb_mean_path = opj(self.subj_path,"predicted_img_emb_mean.pt")
            img_emb_std_path = opj(self.subj_path,"predicted_img_emb_std.pt")
            txt_emb_mean_path = opj(self.subj_path,"predicted_txt_emb_mean.pt")
            txt_emb_std_path = opj(self.subj_path,"predicted_txt_emb_std.pt")

            # Load the tensors
            self.predicted_img_emb_mean = torch.load(img_emb_mean_path)
            self.predicted_img_emb_std = torch.load(img_emb_std_path)
            self.predicted_txt_emb_mean = torch.load(txt_emb_mean_path)
            self.predicted_txt_emb_std = torch.load(txt_emb_std_path)
            
            
            with open(opj(self.subj_path,"predicted_latent_stats.sav"),"rb") as f:
                self.predicted_latent_stats=pickle.load(f)
            

    def get_latents(self,data):
        
        
        shapes={0:(16,1,1),
                1: (16, 1, 1),
                 2: (16, 4, 4),
                 3: (16, 4, 4),
                 4: (16, 4, 4),
                 5: (16, 4, 4),
                 6: (16, 8, 8),
                 7: (16, 8, 8),
                 8: (16, 8, 8),
                 9: (16, 8, 8),
                 10: (16, 8, 8),
                 11: (16, 8, 8),
                 12: (16, 8, 8),
                 13: (16, 8, 8),
                 14: (16, 16, 16),
                 15: (16, 16, 16),
                 16: (16, 16, 16),
                 17: (16, 16, 16),
                 18: (16, 16, 16),
                 19: (16, 16, 16),
                 20: (16, 16, 16),
                 21: (16, 16, 16),
                 22: (16, 16, 16),
                 23: (16, 16, 16),
                 24: (16, 16, 16),
                 25: (16, 16, 16),
                 26: (16, 16, 16),
                 27: (16, 16, 16),
                 28: (16, 16, 16),
                 29: (16, 16, 16),
                 30: (16, 32, 32)}
        
        
        adjust=self.latent_adjust_values
        latents={}
        bs=data.shape[0]
        for k,v in self.brain_to_latent.items():
            s=shapes[k]
            z=torch.tensor(v.predict(data)).reshape(-1,*s)


            if adjust is not None and bs>1:
                #compute actual mean and std
                                
                z_mean=self.predicted_latent_stats[k]["mean"]  
                z_std=self.predicted_latent_stats[k]["std"] 
                
                
                
                #standardize 
                z = (z - z_mean)/(1e-9+z_std)

                #replace with latent mean and std
                z = z*adjust[k]["std"]+adjust[k]["mean"]

            latents[k]=z

        return latents
    
    def decode_with_partial_sampling(self,latents,keep=None):
        xs = {a.shape[2]: a for a in self.vae.decoder.bias_xs}
        
        decoder=self.vae.decoder.to(self.device)
        out=decoder.forward_manual_latents(keep,latents.values(),t=None)

        xs=decoder.out_net.sample(out)
        xs=torch.tensor(xs).permute(0,3,1,2)/255
        return xs
                                             
    def decode_features(self,fmri):
        
        #get latents
        z=self.get_latents(fmri.numpy())
        
        adjust=self.latent_adjust_values
        
        img_emb=[]
        txt_emb=[]
        for i in tqdm.tqdm(range(257)):
            emb=torch.tensor(self.brain_to_img_emb[i].predict(fmri.numpy()))
            # print(emb.shape)
            if adjust and len(fmri)>1:
                #compute actual mean and std
                emb_mean=self.predicted_img_emb_mean[i]
                emb_std=self.predicted_img_emb_std[i]

                emb= (emb-emb_mean)/emb_std
                emb = emb*self.clip_img_embeds_std[i]+self.clip_img_embeds_mean[i]

            img_emb.append(emb)

        for i in tqdm.tqdm(range(77)):


            emb=torch.tensor(self.brain_to_txt_emb[i].predict(fmri.numpy()))

            if adjust and len(fmri)>1:
                #compute actual mean and std
                
                emb_mean=self.predicted_txt_emb_mean[i]
                emb_std=self.predicted_txt_emb_std[i]
                
                emb= (emb-emb_mean)/emb_std

                emb = emb*self.clip_txt_embeds_std[i]+self.clip_txt_embeds_mean[i]
            txt_emb.append(emb)
                                             
        img_emb=torch.stack(img_emb,1)
        txt_emb=torch.stack(txt_emb,1)
        
        return z, img_emb, txt_emb
        
        
    def reconstruct_guess(self,fmri):
        upsample=torchvision.transforms.Resize(512,interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        
        z, img_emb, txt_emb = self.decode_features(fmri)
        
        with torch.no_grad():

            latents={k:v.to(self.device).float() for k,v in z.items()}
            # guess_img=upsample(autoencoder.decoder.double()(z.to(device)).cpu())
            guess_img=self.decode_with_partial_sampling(latents=latents,keep=len(fmri))
            # img_out=pipe_embed.vae.float().decode(z.float().to(device)).sample.cpu()
            print(guess_img.max())
            guess_img=upsample(guess_img).clamp(0,1)
        
        
        return guess_img, z, img_emb, txt_emb
    
    
    def decode(self,fmri,strength=7.5,text_to_image_strength=0.4, num_inference_steps=37,how_many=1, use_latents=True, fix_seed = 42):
        
        if fix_seed is not None:
            seed = 42
            torch.manual_seed(seed)  # For CPU
            torch.cuda.manual_seed(seed)  # For CUDA devices
            torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups
        
        to_pil=torchvision.transforms.ToPILImage()

        
        # decode initial guess and featuers
        guess_img, z, img_emb, txt_emb=self.reconstruct_guess(fmri)
        
        
        # encode null img and null prompt
        null_prompt=""
        null_img=Image.fromarray(np.zeros((425,425,3),dtype=np.uint8))
        uimg=self.pipe_embed._encode_image_prompt([null_img],device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False).cpu()
        utxt=self.pipe_embed._encode_text_prompt([null_prompt],device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False).cpu()
        
        
        #decode the final images
        
        scale=self.pipe_embed.vae.config.scaling_factor
        images=[]
        for i in range(len(fmri)):
            with torch.no_grad():
                print(f"[INFO] Final reconstrution {i+1}/{len(fmri)}")
                encoded_latents=scale*self.pipe_embed.vae.encode((2*guess_img[i:i+1]-1).to(self.device)).latent_dist.sample()
                noise = randn_tensor((how_many,encoded_latents.shape[1],encoded_latents.shape[2],encoded_latents.shape[3]), device=self.device, dtype=encoded_latents.dtype)
                encoded_latents_norm=(encoded_latents-encoded_latents.mean())//(1e-8+encoded_latents.std())
                #final_latents=pipe_embed.scheduler.add_noise(0.0*(encoded_latents_norm.clamp(-3,3)),noise,torch.tensor(50).long().to(device))

                #final_latents=noise+0.18*encoded_latents_norm.clamp(-3,3)
                final_latents=noise+scale*encoded_latents.clamp(-3,3)
                final_latents = (final_latents - final_latents.mean())/final_latents.std()
                
                if use_latents:
                    final_latents=noise+scale*encoded_latents.clamp(-3,3)
                    final_latents = (final_latents - final_latents.mean())/final_latents.std()
                 
                else:
                    final_latents=noise
                

                if strength>1:
                    txt_cond=torch.cat([utxt.repeat(how_many,1,1),txt_emb[i:i+1].float().repeat(how_many,1,1)],0)

                    img_cond=torch.cat([uimg.repeat(how_many,1,1),img_emb[i:i+1].float().repeat(how_many,1,1)],0)
                else:
                    txt_cond=txt_emb[i:i+1].float().repeat(how_many,1,1)
                    img_cond=img_emb[i:i+1].float().repeat(how_many,1,1)

                # print(txt_emb[i:i+1].float().repeat(how_many,1,1).shape,img_emb[i:i+1].float().repeat(how_many,1,1).shape,final_latents.shape)

                # image_generated = pipe_embed([null_prompt]*bs,guessed,txt_cond.to(device), img_cond.to(device), text_to_image_strength=0.4,num_inference_steps=37,guidance_scale=strength,latents=final_latents).images
                image_generated = self.pipe_embed([null_prompt]*how_many,[null_img]*how_many,txt_cond.to(self.device), img_cond.to(self.device), text_to_image_strength=text_to_image_strength,num_inference_steps=num_inference_steps,guidance_scale=strength,latents=final_latents).images
                images+=image_generated
    
        guessed=[to_pil(i) for i in guess_img]
        
        
        return images, guessed
            
                                            
                                            