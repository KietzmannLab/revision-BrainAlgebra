import numpy as np
from transformers import AutoProcessor, CLIPModel
import os
from os.path import join as opj
import matplotlib.pyplot as plt
import torch
from autoencoder import *
from pathlib import Path
import random
from revision_run_algebra_on_brain_space import run_algebra_experiment_on_brain_space_single_pert
#from decoding import BrainDiffuserPretrainedDecoder

project = PROJ_ROOT = Path(__file__).resolve().parents[1]
username = project.name

data_path = Path(f"/share/klab/labstudents/{username}")
processed_data = data_path / "processed_data"
algebra_dir = Path("/share/klab/labstudents/jmihatsch/fmri_algebra")
algebra_dir.mkdir(parents=True, exist_ok=True)

def brain_algebra(subj):
    
    #subj_path=opj("/home/matteo/explore_NSD","models",sub)
    #device = "cpu"
    #brain_decoder=BrainDiffuserPretrainedDecoder(subj_path=subj_path,device=device)

    processed_data_subject = processed_data / f"subj{subj:02d}"

    fmri_train_data_path = opj(processed_data_subject,f"laion_train_fmriavg_laiongeneral_sub{subj:02d}.npy")
    fmri_test_data_path = opj(processed_data_subject,f"laion_test_fmriavg_laiongeneral_sub{subj:02d}.npy")
    train_img_path = opj(processed_data_subject, f"laion_train_stim_sub{subj:02d}.npy")
    test_img_path = opj(processed_data_subject, f"laion_test_stim_sub{subj:02d}.npy")
    embeds_path = data_path / "embeds"

    train_fmri = np.load(fmri_train_data_path)
    #train_fmri = (train_fmri-mean)/std
    test_fmri = np.load(fmri_test_data_path)
    #test_fmri = (test_fmri - mean)/std
    img_embeds = torch.load(embeds_path / f"image_embeds_subject-{subj:02d}.pt")
    train_imgs = np.load(train_img_path)
    test_imgs = np.load(test_img_path)

    # The things the authors of the original paper did to load the CLIP stuff

    discriminator = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    difference_pairs=[("man","woman"),("indoor scene","outdoor scene"),("day","night"),("summer","winter"),("sad","happy"),("cat","dog"), ("still", "moving"), ("left","right"), ("top","down"), ("crowded","empty")]
    modality="fmri"
    N=100
    thr_outputs=[]

    random.seed(42)
    indices = sorted(random.sample(range(len(test_fmri)), 100))
    print(indices)

    prompts = test_fmri[indices]
    stims = test_imgs[indices]

    #copied from the original paper's code, some of their pre-defined stuff

    #copying the original paper's code with slight changes. seems like it should work. I'll figure out how to check it later.
    for thr in [0, 25, 50, 75, 90, 95]: #going over the same experiment with different thresholds
        print(f"INFO thr: {thr}") #printing out the current threshold
        out_dir= data_path / f"models/sub-0{subj}/encoding/algebra_l2_SINGLE_thr_{thr}" #creating a folder for the output of the treshlold (thr) and subject (sub) so at the end ther should be 4 subjects * 3 thresholds = 12 folders
        os.makedirs(out_dir,exist_ok=True) #makes the folder if it doesnt exist already, goes on if it does

        outputs = {} #to collect the outputs

        for diff in difference_pairs: #taking each pair of concepts one by one

            positive, negative = diff #unpacking the pair into a "positive" and a "negative" concept (that's a funny way to name it no problems whatsoever have occured historically with defining binary categories like man and woman into positive and negative)
            
            inputs = processor(text=[positive, negative],return_tensors="pt", padding=True) #this processes text into some other format (ids?)
            txt_embeds=discriminator.get_text_features(inputs["input_ids"]) #This ueses the ids from the line before to get the CLIP embeddings.

            #here it seems like they are checking to which word of the pair the image is closer and sorting them to the pos and neg versions of the category (difference pair)
            probs = (img_embeds@txt_embeds.T).softmax(1) #gets a matrix of embedded images and text
            pos_indices = probs[:,0].argsort()[-N:].detach().numpy() #retures the N most similar rows to the positive word (row number, so it can be used to find images)
            neg_indices = probs[:,1].argsort()[-N:].detach().numpy() #returnes the N most similar rows to the negative word
            fmri_positive=train_fmri[pos_indices][:N].mean(0) #avarages the brain activation pattern for all the positive concept corresponding images
            fmri_negative=train_fmri[neg_indices][:N].mean(0) # same for negative
            # for replication purposes and to make sure the images actually look like they are close to the concepts, here is plotting the images. That's gonna be a lot. 
            for pert,fmri_pert in zip(diff,[fmri_positive,fmri_negative]): #gets the name of the perturbation (the concept) and the perturbation vector (avareged betas)
                print("[INFO] Running", pert) #prints out which pertrubation concept is being delt with
                target_dir = opj(out_dir, modality, f"{pert}_thr_{thr}") #creates a folder name for the current perturbation and threshhold to store the results
                os.makedirs(target_dir, exist_ok=True) #creates the folder if it doesnt exist yet, goes on if it does
                outputs[f"{pert}"] = {} #creates a structure to store the results under the current pertrubation name
                
                    # Plotting the grid of images
                if pert == positive: # for the positive concept
                    grid_images = train_imgs[pos_indices][::-1]
                    #create a grid withh the corresponding images
                    title = f"Positive Images for {positive} (thr={thr})" #title the grid with the concept and the threshold in the name
                else: #same steps but for the negative concept
                    grid_images = train_imgs[neg_indices][::-1]
                    title = f"Negative Images for {negative} (thr={thr})"

                fig, axes = plt.subplots(10, 10, figsize=(20, 20)) #creates a plot with 10*10 (100) images in a 20*20 size figure (each image is 2*2)
                fig.suptitle(title, fontsize=20) #titels the figure wth the title from above (including the concept and threshold) in a 20 size font
                for img, ax in zip(grid_images[:100], axes.flat): #goes through each image and the corresponding axes (.flat to make it a vector instead of a matrix)
                    ax.imshow(img)  # Assuming images are in a format plt.imshow can display -- original author's comment
                    ax.axis('off') #does not show the axes -- for pretty?
                plt.savefig(opj(target_dir, f"{pert}_images_grid_new.png")) #saves the figure with the pertrubation name
                #plt.show() #shows the figure

                
                all_algebra_fmri = []
                for i, (prompt, stim) in enumerate(list(zip(prompts, stims))):

                    #plain_decoded, algebra_imgs, fmri_difference, fig = 
                    fmri_algebra = run_algebra_experiment_on_brain_space_single_pert(
                        fmri_inputs=prompt,
                        kind=modality,
                        pert=pert,
                        fmri_pert= fmri_pert,
                        fmri_stim=stim,
                        thr_perc=thr,
                        subj=subj
                    )
                    all_algebra_fmri.append(fmri_algebra.numpy())
                all_algebra_fmri = np.concatenate(all_algebra_fmri)
                print(all_algebra_fmri.shape)
                fmri_algebra_dir = algebra_dir / f"subj{subj:02d}"
                np.save(fmri_algebra_dir / f"fmri_brain_algebra_sub{subj:02d}_{pert}_{thr}.npy", all_algebra_fmri)

                
                    # Save stuff

                #     torch.save(fmri_difference, opj(target_dir, f"diff_{i}.pt"))
                #     fig.savefig(opj(target_dir, f"algebra_{i}.png"))
                #     svg_filename = opj(target_dir, f"algebra_{i}.svg")
                #     fig.savefig(svg_filename, format='svg', dpi=300)

                #     # Store results in the dictionary
                #     outputs[f"{pert}"][i]= {
                #         "fmri_difference": fmri_difference,
                #         "algebra_imgs": algebra_imgs,
                #         "plain_decoded": plain_decoded,
                #         "figure": fig,
                #         "pos_num": len(pos_indices),
                #         "neg_num": len(neg_indices)
                #     }


                # thr_outputs.append(outputs)

#for subj in (1,3,5,6,7):
#    clip_encode(subj)

brain_algebra(subj=1)

#python3 revision_brain_algebra.py