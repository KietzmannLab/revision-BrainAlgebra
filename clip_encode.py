from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_train_test_ids
import numpy as np
import pandas as pd

# Load betas and trial info as usual (see laion_fmri_package/load).
sessions = sub.get_sessions()
betas_per_ses  = sub.get_betas(session=sessions, roi="laiongeneral")
trials_per_ses = sub.get_trial_info(session=sessions)

# Concatenate across sessions (standard idiom).
betas  = np.concatenate(list(betas_per_ses.values()), axis=0)
trials = pd.concat(list(trials_per_ses.values()), ignore_index=True)
label = trials['label'] 

# The things the authors of the original paper did to load the CLIP stuff
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
discriminator = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

subject = "sub-01" #making subject into variable to possibly use later as function
sub = load_subject(subject) 
#lodaing the full subject info - probably when converting into a function after doing the split train/test and avareging for each picture not needed, instead the function sould get the whole preprocessed data of a subject
images_with_embeddings = np.concatenate([sub.metadata.image_name, sub.embeddings.all("CLIP")], axis=1) 
#not sure if "image_name" would work, and the embeddings maybe better done before the preprocessing to not get confused about order, image names etc., so if doing a function that sould not be needed and the function should get data already with the embeddings 
#this should thechnically make a matrix where each image name is paired with the embedded CLIP, so then it should be possible to find the embedding using the name
#really unsure about what im doing here, and the stuff for some reason won't run probably because I messed up something in the terminal

difference_pairs=[("man","woman"),("indoor scene","outdoor scene"),("day","night"),("summer","winter"),("sad","happy"),("cat","dog"), ("still", "moving"), ("left","right"), ("top","down"), ("crowded","empty")]
modality="fmri"
N=100
thr_outputs=[]
#copied from the original paper's code, some of their pre-defined stuff

#copying the original paper's code with slight changes. seems like it should work. I'll figure out how to check it later.
for thr in [75,90,95]: #going over the same experiment with different thresholds
    print(f"INFO thr: {thr}") #printing out the current threshold
    out_dir=f"models/{sub}/encoding/algebra_l2_SINGLE_thr_{thr}" #creating a folder for the output of the treshlold (thr) and subject (sub) so at the end ther should be 4 subjects * 3 thresholds = 12 folders
    os.makedirs(out_dir,exist_ok=True) #makes the folder if it doesnt exist already, goes on if it does

    outputs = {} #to collect the outputs

    for diff in difference_pairs: #taking each pair of concepts one by one
        

        positive, negative = diff #unpacking the pair into a "positive" and a "negative" concept (that's a funny way to name it no problems whatsoever have occured historically with defining binary categories like man and woman into positive and negative)
        
        inputs = processor(text=[positive, negative],return_tensors="pt", padding=True) #this processes text into some other format (ids?)
        txt_embeds=discriminator.get_text_features(inputs["input_ids"]) #This ueses the ids from the line before to get the CLIP embeddings.

        #here it seems like they are checking to which word of the pair the image is closer and sorting them to the pos and neg versions of the category (difference pair)
        probs = (images_with_embeddings@txt_embeds.T).softmax(1) #gets a matrix of embedded images and text
        pos_indices = probs[:,0].argsort()[-N:].detach().numpy() #retures the N most similar rows to the positive word (row number, so it can be used to find images)
        neg_indices = probs[:,1].argsort()[-N:].detach().numpy() #returnes the N most similar rows to the negative word
        fmri_positive=betas[pos_indices][:N].mean(0) #avarages the brain activation pattern for all the positive concept corresponding images
        fmri_negative=betas[neg_indices][:N].mean(0) # same for negative
        # for replication purposes and to make sure the images actually look like they are close to the concepts, here is plotting the images. That's gonna be a lot. 
        for pert,fmri_pert in zip(diff,[fmri_positive,fmri_negative]): #gets the name of the perturbation (the concept) and the perturbation vector (avareged betas)
            print("[INFO] Running", pert) #prints out which pertrubation concept is being delt with
            target_dir = opj(out_dir, modality, f"{pert}_thr_{thr}") #creates a folder name for the current perturbation and threshhold to store the results
            os.makedirs(target_dir, exist_ok=True) #creates the folder if it doesnt exist yet, goes on if it does
            outputs[f"{pert}"] = {} #creates a structure to store the results under the current pertrubation name
            
                # Plotting the grid of images
            if pert == positive: # for the positive concept
                grid_images = sub.images.get(pos_indices) #create a grid withh the corresponding images
                title = f"Positive Images for {positive} (thr={thr})" #title the grid with the concept and the threshold in the name
            else: #same steps but for the negative concept
               grid_images = sub.images.get(neg_indices)
               title = f"Negative Images for {negative} (thr={thr})"

            fig, axes = plt.subplots(10, 10, figsize=(20, 20)) #creates a plot with 10*10 (100) images in a 20*20 size figure (each image is 2*2)
            fig.suptitle(title, fontsize=20) #titels the figure wth the title from above (including the concept and threshold) in a 20 size font
            for img, ax in zip(grid_images[:100], axes.flat): #goes through each image and the corresponding axes (.flat to make it a vector instead of a matrix)
                ax.imshow(img)  # Assuming images are in a format plt.imshow can display -- original author's comment
                ax.axis('off') #does not show the axes -- for pretty?
            plt.savefig(opj(target_dir, f"{pert}_images_grid.png")) #saves the figure with the pertrubation name
            plt.show() #shows the figure

        

