import numpy as np
from pathlib import Path
import pickle
from PIL import Image   
import matplotlib.pyplot as plt
import tqdm
from os.path import join as opj
import glob
from transformers import AutoProcessor, CLIPModel
import torch
import tqdm
import numpy as np
from PIL import ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor
discriminator = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

N = 100
subjs = [1]  #,"subj03","subj05","subj06", "subj07"
difference_pairs=[("man","woman"),("indoor scene","outdoor scene"),("day","night"),("summer","winter"),("sad","happy"), ("crowded","empty")]  #,("cat","dog"), ("still", "moving"), ("left","right"), ("top","down"), 
unrolled_concepts=  [item for pair in difference_pairs for item in pair]
thrs = [50]  #[0, 25, 50, 75, 90, 95]
out_dir = Path(f"/share/klab/labstudents/jmihatsch/similarity_data")
out_dir.mkdir(parents=True,exist_ok=True )

print("v1")

all_results_subv1={}

for sub in subjs:
    # Thresholds and coordinate values
    thrs = thrs
    coords_list = {
        "minus_seven": 0,
        "minus_five": 1,
        "minus_three": 2,
        "minus_one": 3,
        "original": 4,  # Include original coordinates
        "plus_one": 5,
        "plus_three": 6,
        "plus_five": 7,
        "plus_seven": 8
    }

    T = 0.1 

    # Initialize results dictionary
    all_results = {}
    with torch.no_grad():
        for thr in thrs:
            print(f"Processing threshold: {thr}")
            all_results[thr] = dict()
            for pair in unrolled_concepts:  # You should have this defined with your concept pairs
                print(pair)
                all_results[thr][pair] = dict()

                algebra_dir = Path(f"/share/klab/labstudents/jmihatsch/diffuser_img/subj{sub:02d}/{pair}_{thr}")
                images = [Image.open(algebra_dir / f"diffuser_{i}.png") for i in range(N * 9)]
                # Encode the original image
                original_img = images[4]
                inputs = processor(images=[original_img], return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                original_img_embeds = discriminator.get_image_features(inputs["pixel_values"]).cpu()
                original_img_embeds_norm = torch.nn.functional.normalize(original_img_embeds, p=2, dim=1)

                # Crop and process images for each coordinate, including the original
                for key, offset in coords_list.items():  #instead of coordinate its every 9th img
                    cropped_images = images[offset::9]

                    # Encode images with CLIP
                    img_embeds = []
                    BS = 128
                    with torch.no_grad():
                        for batch_idx in tqdm.trange(0, len(cropped_images)//BS+1, 1):
                            batch = cropped_images[batch_idx*BS:(batch_idx+1)*BS]
                            inputs = processor(images=batch, return_tensors="pt", padding=True)
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            emb = discriminator.get_image_features(inputs["pixel_values"]).cpu()
                            img_embeds.append(emb)

                    img_embeds = torch.cat(img_embeds, 0)
                    all_results[thr][pair][key] = img_embeds

                # Compute text embeddings
                inputs = processor(text=[pair], return_tensors="pt", padding=True)
                txt_embeds = discriminator.get_text_features(inputs["input_ids"].to(device)).cpu()

                # Calculate similarities and store results
                for key in coords_list.keys():
                    img_embeds = all_results[thr][pair][key]

                    # Normalize the embeddings
                    img_embeds_norm = torch.nn.functional.normalize(img_embeds, p=2, dim=1)
                    txt_embeds_norm = torch.nn.functional.normalize(txt_embeds, p=2, dim=1)

                    # Compute cosine similarity with text
                    cosine_sim = img_embeds_norm @ txt_embeds_norm.T

                    # Compute cosine similarity with the original image
                    original_img_cosine_sim = img_embeds_norm @ original_img_embeds_norm.T

                    # Concatenate similarities horizontally
                    combined_cosine_sim = torch.cat((cosine_sim, original_img_cosine_sim), dim=1)

                    # Apply softmax to obtain probabilities
                    # probs = torch.nn.functional.softmax(combined_cosine_sim/T, dim=1)

                    all_results[thr][pair][key] = combined_cosine_sim.numpy()

                    
    all_results_subv1[sub] = all_results

with open(out_dir / "similarity_v1.pkl", "wb") as f:
    pickle.dump(all_results_subv1, f)



print("v2")

all_results_subv2={}

for sub in subjs:
    random.seed(42)
    data_path = Path(f"/share/klab/labstudents/jmihatsch/processed_data/subj{sub:02d}")
    test_img_path = data_path / f"laion_test_stim_sub{sub:02d}.npy"
    test_imgs = np.load(test_img_path)
    indices = sorted(random.sample(range(len(test_imgs)), 100))
    stimuli = [Image.fromarray(test_imgs[k]) for k in indices]
    inputs = processor(images=stimuli, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        stim_embeds = discriminator.get_image_features(inputs["pixel_values"]).cpu()
    stim_embeds_norm = torch.nn.functional.normalize(stim_embeds, p=2, dim=1)

    # Thresholds and coordinate values
    thrs = thrs
    coords_list = {
        "minus_seven": 0,
        "minus_five": 1,
        "minus_three": 2,
        "minus_one": 3,
        "original": 4,  # Include original coordinates
        "plus_one": 5,
        "plus_three": 6,
        "plus_five": 7,
        "plus_seven": 8
    }

    T = 0.1 

    stimuli = [Image.fromarray(test_imgs[k]) for k in indices]

    # Initialize results dictionary
    all_results = {}
    with torch.no_grad():
        for thr in thrs:
            print(f"Processing threshold: {thr}")
            all_results[thr] = dict()
            for pair in unrolled_concepts:  # You should have this defined with your concept pairs
                print(pair)
                all_results[thr][pair] = dict()

                algebra_dir = Path(f"/share/klab/labstudents/jmihatsch/diffuser_img/subj{sub:02d}/{pair}_{thr}")
                images = [Image.open(algebra_dir / f"diffuser_{i}.png") for i in range(N * 9)]
                # Encode the original image
                # original_img = stimulus
                # inputs = processor(images=[original_img], return_tensors="pt", padding=True)
                # inputs = {k: v.to(device) for k, v in inputs.items()}
                # original_img_embeds = discriminator.get_image_features(inputs["pixel_values"]).cpu()
                # original_img_embeds_norm = torch.nn.functional.normalize(original_img_embeds, p=2, dim=1)

                # Crop and process images for each coordinate, including the original
                for key, offset in coords_list.items():  #instead of coordinate its every 9th img
                    cropped_images = images[offset::9]

                    # Encode images with CLIP
                    img_embeds = []
                    BS = 128
                    with torch.no_grad():
                        for batch_idx in tqdm.trange(0, len(cropped_images)//BS+1, 1):
                            batch = cropped_images[batch_idx*BS:(batch_idx+1)*BS]
                            inputs = processor(images=batch, return_tensors="pt", padding=True)
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            emb = discriminator.get_image_features(inputs["pixel_values"]).cpu()
                            img_embeds.append(emb)

                    img_embeds = torch.cat(img_embeds, 0)
                    all_results[thr][pair][key] = img_embeds

                # Compute text embeddings
                inputs = processor(text=[pair], return_tensors="pt", padding=True)
                txt_embeds = discriminator.get_text_features(inputs["input_ids"].to(device)).cpu()

                # Calculate similarities and store results
                for key in coords_list.keys():
                    img_embeds = all_results[thr][pair][key]

                    # Normalize the embeddings
                    img_embeds_norm = torch.nn.functional.normalize(img_embeds, p=2, dim=1)
                    txt_embeds_norm = torch.nn.functional.normalize(txt_embeds, p=2, dim=1)

                    # Compute cosine similarity with text
                    cosine_sim = img_embeds_norm @ txt_embeds_norm.T

                    # Compute cosine similarity with the original image
                    original_img_cosine_sim = (img_embeds_norm * stim_embeds_norm).sum(1, keepdim=True)

                    # Concatenate similarities horizontally
                    combined_cosine_sim = torch.cat((cosine_sim, original_img_cosine_sim), dim=1)

                    # Apply softmax to obtain probabilities
                    # probs = torch.nn.functional.softmax(combined_cosine_sim/T, dim=1)

                    all_results[thr][pair][key] = combined_cosine_sim.numpy()

                    
    all_results_subv2[sub] = all_results
with open(out_dir / "similarity_v2.pkl", "wb") as f:
    pickle.dump(all_results_subv2, f)