import matplotlib.pyplot as plt
import numpy as np
import torch
from autoencoder import *
from torch.nn import functional as F
from decoding import BrainDiffuserPretrainedDecoder

def run_algebra_experiment_on_brain_space_single_pert(fmri_inputs=None, kind="text", pert="",fmri_pert=None, fmri_stim=None,thr_perc=None,l2_norm=True, N=100):
    
    ## look for concepts fMRI
    # inputs = processor(text=[positive, negative],return_tensors="pt", padding=True)
    #     txt_embeds=discriminator.get_text_features(inputs["input_ids"].to(device)).cpu()
    
    #     probs = (img_embeds@txt_embeds.T).softmax(1)
    #     pos_indices = np.where(probs[:,0]>0.95)[0]
    #     neg_indices = np.where(probs[:,1]>0.95)[0]
    
    #     fmri_positive=train_fmri[pos_indices][:N].mean(0)
    #     fmri_negative=train_fmri[neg_indices][:N].mean(0)
    # fmri_positive = positive
    # fmri_negative = negative
    
    fmri_encoded=torch.tensor(fmri_inputs).unsqueeze(0)
    
    
    
    ## what if they have different shapes?
        
    # plain_decoded,_=brain_decoder.decode(fmri_encoded,num_inference_steps=37)   

    fmri_difference = torch.tensor(fmri_pert)
    
    n4=5.7445
    n3=4.3588
    n2=3.
    n1=1.7320
    
    c1=0.25
    c2=0.5
    c3=0.75
    c4=1
    
    if thr_perc is not None:
        print(f"thresholding concept at {thr_perc}")
        #trheshold the difference vector in brain space
        if thr_perc>1:
        
            thr=torch.quantile(fmri_difference.abs(),thr_perc/100)
        else:
            thr=torch.quantile(fmri_difference.abs(),thr_perc)

        mask = (fmri_difference.abs() <= thr)

        # Set values below the threshold to zero
        fmri_difference[mask] = 0
    
        n1=1
        n2=1
        n3=1
        n4=1
        
        c1=1
        c2=2
        c3=3
        c4=4
    
    fmri_algebra=torch.cat([(fmri_encoded-c4*fmri_difference)/n4,
                            (fmri_encoded-c3*fmri_difference)/n3, 
                              (fmri_encoded-c2*fmri_difference)/n2,
                              (fmri_encoded-c1*fmri_difference)/n1,
                              fmri_encoded,
                              (fmri_encoded+c1*fmri_difference)/n1,
                              (fmri_encoded+c2*fmri_difference)/n2,
                              (fmri_encoded+c3*fmri_difference)/n3,
                              (fmri_encoded+c4*fmri_difference)/n4
                           ],0)
    
    
    if l2_norm:
        norm=fmri_encoded.norm(dim=1) #norm before
        fmri_algebra=F.normalize(fmri_algebra,dim=1)
        #reset the previous norm
        fmri_algebra=norm.unsqueeze(1)*fmri_algebra
        
    # print(fmri_algebra.shape)
    
    algebra_imgs,_=brain_decoder.decode(fmri_algebra,num_inference_steps=37)
    

    if kind=="fmri":
        fig, axs = plt.subplots(1,10, figsize=(30,6),subplot_kw={'aspect': 1})

        # titles=[f"Original {i}({positive}-{negative})" for i in ["-3","-2","-1","+1","+2","+3"]]
        
        axs[0].imshow(fmri_stim)
        axs[0].axis("off")
        axs[0].set_title("Stimulus shown")


    #         axs[1].imshow(plain_decoded[0])
    #         axs[1].axis("off")
    #         axs[1].set_title("Original decoded")

        titles=[f"Original -{i}({pert})" for i in [c1,c2,c3,c4][::-1]]+["Original decoded"]+[f"Original +{i}({pert})" for i in  [c1,c2,c3,c4]]
        for i in range(9):
            axs[i+1].imshow(algebra_imgs[i])
            axs[i+1].axis("off")
            axs[i+1].set_title(wrap_text(titles[i]))
            
            
        # Add a border around the image
        border_color = 'black'  # You can specify the border color
        border_width = 3      # You can specify the border width in pixels

        # Get the dimensions of the image
        height, width, _ = np.array(algebra_imgs[0]).shape

        # Create a rectangle patch and add it to the axes
        border_rect = patches.Rectangle((0, 0), width, height, linewidth=border_width, edgecolor=border_color, fill=False)
        axs[5].add_patch(border_rect)
        
    
    plain_decoded=algebra_imgs[4]
    #save difference
    
    return plain_decoded, algebra_imgs, fmri_difference,fig
    
    