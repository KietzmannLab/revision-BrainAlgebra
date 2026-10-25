import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join as opj
import pickle
from pathlib import Path


def plot_similarity_original_image(subjs, thrs, version):
    scaling_values = [0,1,2,3,4]
    difference_pairs=[("man","woman"),("indoor scene","outdoor scene"),("day","night"),("summer","winter"),("sad","happy"), ("crowded","empty")]
    #all_keys = ["minus_seven", "minus_five", "minus_three", "minus_one", "original","plus_one", "plus_three", "plus_five", "plus_seven"]
    all_keys =  ["original","plus_one", "plus_three", "plus_five", "plus_seven"]
    unrolled_concepts = [item for pair in difference_pairs for item in pair]

    v = f"v{version}"

    with open(f"/share/klab/labstudents/jmihatsch/similarity_data/similarity_{v}.pkl", "rb") as f:
        all_results_sub = pickle.load(f)


    with open("/share/klab/labstudents/jmihatsch/similarity_data/baselines.pkl", "rb") as f:
        baselines_aggregated = pickle.load(f)
     # Define the number of columns and rows for subplots
    n_cols = 3
    n_rows = 2
    
    # Initialize dictionaries to store results across subjects
    group_y1_means = {thr: {pair: {key: [] for key in all_keys} for pair in unrolled_concepts} for thr in thrs}
    group_y2_means = {thr: {pair: {key: [] for key in all_keys} for pair in unrolled_concepts} for thr in thrs}
    group_y1_stds = {thr: {pair: {key: [] for key in all_keys} for pair in unrolled_concepts} for thr in thrs}
    group_y2_stds = {thr: {pair: {key: [] for key in all_keys} for pair in unrolled_concepts} for thr in thrs}
    
    # Collect data across all subjects
    for sub in subjs:
        all_results = all_results_sub[sub]
            
        for pair in unrolled_concepts:
            for thr in thrs:
                for key in all_keys:
                    img_embeds = all_results[thr][pair][key]
                    group_y1_means[thr][pair][key].append(img_embeds[:, 0].mean())
                    group_y2_means[thr][pair][key].append(img_embeds[:, 1].mean())
                    group_y1_stds[thr][pair][key].append(img_embeds[:, 0].std())
                    group_y2_stds[thr][pair][key].append(img_embeds[:, 1].std())
    
    # Calculate group means and average of STDs
    group_means_y1 = {thr: {pair: {key: np.mean(group_y1_means[thr][pair][key]) for key in all_keys} for pair in unrolled_concepts} for thr in thrs}
    group_means_y2 = {thr: {pair: {key: np.mean(group_y2_means[thr][pair][key]) for key in all_keys} for pair in unrolled_concepts} for thr in thrs}
    average_stds_y1 = {thr: {pair: {key: np.mean(group_y1_stds[thr][pair][key]) for key in all_keys} for pair in unrolled_concepts} for thr in thrs}
    average_stds_y2 = {thr: {pair: {key: np.mean(group_y2_stds[thr][pair][key]) for key in all_keys} for pair in unrolled_concepts} for thr in thrs}
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10), sharex=True)
    
    fig.suptitle(f"Average Similarity between decoded perturbed image and ORIGINAL IMAGE")
    
    for idx, thr in enumerate(thrs):
        row = idx // n_cols
        col = idx % n_cols
    
        y1_means = np.array([[group_means_y1[thr][pair][key] for key in all_keys] for pair in unrolled_concepts]).mean(0) ## all concpets
        y2_means = np.array([[group_means_y2[thr][pair][key] for key in all_keys] for pair in unrolled_concepts]).mean(0)
        y1_avg_stds = np.array([[average_stds_y1[thr][pair][key] for key in all_keys] for pair in unrolled_concepts]).mean(0)
        y2_avg_stds = np.array([[average_stds_y2[thr][pair][key] for key in all_keys] for pair in unrolled_concepts]).mean(0)
    
        axs[row, col].plot(scaling_values, y2_means, label=f"decoded image similarity",color="tab:green")
        axs[row, col].scatter(scaling_values, y2_means,color="tab:green")
        axs[row, col].fill_between(scaling_values, 
                                np.array(y2_means) - np.array(y2_avg_stds), 
                                np.array(y2_means) + np.array(y2_avg_stds), alpha=0.2,color="tab:green")
        axs[row,col].set_xticks(scaling_values)
    
        axs[row,col].hlines(baselines_aggregated["baseline_img_similarities_mean"].mean(),xmin=0,xmax=4,linestyle="--",color="tab:red",label="baseline")
        axs[row, col].fill_between(x=scaling_values, y1=baselines_aggregated["baseline_img_similarities_mean"].mean()-baselines_aggregated["baseline_img_similarities_std"].mean(), 
                                y2=baselines_aggregated["baseline_img_similarities_mean"].mean()+baselines_aggregated["baseline_img_similarities_std"].mean(),color="tab:orange",alpha=0.2)
    
        axs[row, col].set_ylabel("Similarity")
        axs[row, col].set_title(f"Threshold {thr}")
        axs[row, col].legend()
    
        for ax in axs.flat:
            ax.label_outer() 
    
        axs[-1, 0].set_xlabel("Scaling Value")
        axs[-1, 1].set_xlabel("Scaling Value")
        axs[-1, 2].set_xlabel("Scaling Value")
    
        figures_dir = Path(f"/home/student/i/igarasimyuk/revision-BrainAlgebra/figures")
        out_dir = figures_dir / "similarity_plots"
        out_dir.mkdir(parents=True,exist_ok=True )
    
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the main title
        plt.savefig(out_dir / f"group_decoded_perturbed_and_og_image_{v}.png")
    



plot_similarity_original_image(
    subjs = [1],
    thrs = [0, 25, 50, 75, 90, 95],
    version=2
)