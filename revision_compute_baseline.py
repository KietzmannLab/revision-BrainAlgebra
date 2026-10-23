import random
import pickle
from pathlib import Path
import numpy as np
import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor, CLIPModel

sub = 1
N = 100
base_thr = 50
data_path = Path("/share/klab/labstudents/jmihatsch")
difference_pairs=[("man","woman"),("indoor scene","outdoor scene"),("day","night"),("summer","winter"),("sad","happy"), ("crowded","empty")]  #,("cat","dog"), ("still", "moving"), ("left","right"), ("top","down"), 
unrolled_concepts=  [item for pair in difference_pairs for item in pair]
# from encode images
train_img_embeds = torch.load(data_path / f"embeds/image_embeds_subject-{sub:02d}.pt").float().cpu()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

random.seed(42) 

baselines = {}
with torch.no_grad():
    for pair in tqdm.tqdm(unrolled_concepts):
        algebra_dir = Path(f"/share/klab/labstudents/jmihatsch/diffuser_img/subj{sub:02d}/{pair}_{base_thr}")
        baselines[pair] = dict()

        original_img = [Image.open(algebra_dir / f"diffuser_{9 * k + 4}.png") for k in range(N)]
        inputs = processor(images=original_img, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        original_img_embeds = discriminator.get_image_features(inputs["pixel_values"]).cpu()
        original_img_embeds_norm = torch.nn.functional.normalize(original_img_embeds, p=2, dim=1)

        ## sample 100 random images from train_img_embeds
        indices = random.sample(range(len(train_img_embeds)), 100)
        sampled_embeds = train_img_embeds[indices]
        sampled_embeds_norm = torch.nn.functional.normalize(sampled_embeds, p=2, dim=1)

        # just do cosine similarity
        baseline_sim = (original_img_embeds_norm @ sampled_embeds_norm.T)
        baselines[pair]["baseline_img_similarities_mean"] = baseline_sim.mean(1)
        baselines[pair]["baseline_img_similarities_std"] = baseline_sim.std(1)

        # Compute text embeddings
        inputs = processor(text=[pair], return_tensors="pt", padding=True)
        txt_embeds = discriminator.get_text_features(inputs["input_ids"].to(device)).cpu()
        txt_embeds_norm = torch.nn.functional.normalize(txt_embeds, p=2, dim=1)

        # compute text sim with embeddings
        concept_sim = (txt_embeds_norm @ sampled_embeds_norm.T)
        baselines[pair]["baseline_concept_similarities_mean"] = concept_sim.mean(1)
        baselines[pair]["baseline_concept_similarities_std"] = concept_sim.std(1)


keys = ["baseline_img_similarities_mean", "baseline_img_similarities_std",
        "baseline_concept_similarities_mean", "baseline_concept_similarities_std"]

baselines_aggregated = {}
for key in keys:
    values = []
    for pair in baselines:
        values.append(baselines[pair][key].numpy())   # one array per concept
    baselines_aggregated[key] = np.concatenate(values)   # all concepts in one array

with open(data_path / "similarity_data/baselines.pkl", "wb") as f:
    pickle.dump(baselines_aggregated, f)