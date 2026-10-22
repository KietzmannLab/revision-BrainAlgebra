import random
import textwrap
from pathlib import Path
import pickle
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sub = 1
thr = 50
#pert = "outdoor scene"
concepts = ["man","woman","indoor scene", "outdoor scene","day", "night","summer","winter","sad","happy","crowded","empty"]
figures_dir = Path(f"/share/klab/labstudents/jmihatsch/figures/")
# for thr 75 need "sad","happy","crowded","empty"

def plot_diffusion_decoded(sub, thr, concepts, figures_dir=figures_dir):
    print(f"[INFO] Plotting diffusion images for sub {sub}, threshold {thr}")
    random.seed(42)
    N = 100
    c1, c2, c3, c4 = 1, 2, 3, 4

    data_path = Path(f"/share/klab/labstudents/jmihatsch/processed_data/subj{sub:02d}")
    fmri_test_data_path = data_path / f"laion_test_fmriavg_laiongeneral_sub{sub:02d}.npy"
    test_fmri = np.load(fmri_test_data_path)
    test_img_path = data_path / f"laion_test_stim_sub{sub:02d}.npy"
    test_imgs = np.load(test_img_path)

    indices = sorted(random.sample(range(len(test_fmri)), 100))
    stims = test_imgs[indices]

    for pert in concepts:
        diffusion_img_path = Path(f"/share/klab/labstudents/jmihatsch/diffuser_img/subj{sub:02d}/{pert}_{thr}")
        plot_dir = figures_dir / f"diffusion_plots/subj{sub:02d}/{pert}_{thr}"
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True,exist_ok=True )

        print(f"[INFO] Plotting figure for sub {sub}, concept {pert}, and threshold {thr}")
        last_img = diffusion_img_path / "diffuser_899.png" 

        if not last_img.exists():
            print(f"ERROR: decoding for {pert}_{thr} is incomplete, skip")
            continue

        start = 0
        for stim in range(N):
            paths = []  
            for i in range(start, start + 9):
                paths.append(diffusion_img_path / f"diffuser_{i}.png")
            fig, axs = plt.subplots(1,10, figsize=(30,6),subplot_kw={'aspect': 1})
            axs[0].imshow(stims[stim])
            axs[0].axis("off")
            axs[0].set_title("Stimulus shown")
            titles=[f"Original -{i}({pert})" for i in [c1,c2,c3,c4][::-1]]+["Original decoded"]+[f"Original +{i}({pert})" for i in  [c1,c2,c3,c4]]
            for i in range(9):
                axs[i+1].imshow(Image.open(paths[i]))
                axs[i+1].axis("off")
                axs[i+1].set_title(textwrap.fill(titles[i],14))
            border_color = 'black'  # You can specify the border color
            border_width = 3      # You can specify the border width in pixels
            width, height = Image.open(paths[4]).size
            border_rect = patches.Rectangle((0, 0), width, height, linewidth=border_width, edgecolor=border_color, fill=False)
            axs[5].add_patch(border_rect)
            fig.savefig(plot_dir / f"sub{sub:02d}_{pert}_{thr}_stim{indices[stim]}.png", bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved Image {stim}")
            start += 9


def plot_compare_original_vdvae_decoded(sub, thr, concepts, figures_dir=figures_dir):
    print(f"[INFO] Plotting comparison figure for sub {sub}, threshold {thr}")
    random.seed(42)
    N = 100
    data_path = Path(f"/share/klab/labstudents/jmihatsch/processed_data/subj{sub:02d}")
    test_fmri = np.load(data_path / f"laion_test_fmriavg_laiongeneral_sub{sub:02d}.npy")
    test_imgs = np.load(data_path / f"laion_test_stim_sub{sub:02d}.npy")
    indices = sorted(random.sample(range(len(test_fmri)), 100))
    stims = test_imgs[indices]
    del test_fmri

    diffusion_img_path = Path(f"/share/klab/labstudents/jmihatsch/diffuser_img/subj{sub:02d}/woman_{thr}")
    vdvae_img_path = Path(f"/share/klab/labstudents/jmihatsch/vdvae_img/subj{sub:02d}/woman_{thr}")
    plot_dir = figures_dir / f"comparison_plots/subj{sub:02d}/threshold_{thr}"
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True,exist_ok=True )

    start = 0
    for stim in range(N):
        img = stim*9+4
        vdvae_img = vdvae_img_path / f"vdvae_image_{img}.png"
        diffuser_img = diffusion_img_path / f"diffuser_{img}.png"
        fig, axs = plt.subplots(1,3, figsize=(12,5), subplot_kw={'aspect': 1})
                
        axs[0].imshow(stims[stim])
        axs[0].set_title("Stimulus shown")
        axs[1].imshow(Image.open(vdvae_img))
        axs[1].set_title("VDVAE reconstructed image")
        axs[2].imshow(Image.open(diffuser_img))
        axs[2].set_title("Diffuser reconstructed image")
        for ax in axs:
            ax.axis("off")      

        fig.savefig(plot_dir / f"sub{sub:02d}_{thr}_stim{indices[stim]}_comparison.png", bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved Image {stim}")


def get_alex_index(sub, stim, shared, ordered):
    orig_idx_map = {label: i for i, label in enumerate(shared)}
    label = ordered[stim]
    orig_idx = orig_idx_map[label]
    return orig_idx


def plot_compare_ours_with_alex(sub, thr, concepts):
    print(f"[INFO] Plotting comparison with alex figure for sub {sub}, threshold {thr}")
    random.seed(42)
    N = 100

    save_path=Path(f"/share/klab/labstudents/jmihatsch/processed_data/subj{sub:02d}")
    data_path = Path(f"/share/klab/labstudents/jmihatsch/processed_data/subj{sub:02d}")
    test_fmri = np.load(data_path / f"laion_test_fmriavg_laiongeneral_sub{sub:02d}.npy")
    test_imgs = np.load(data_path / f"laion_test_stim_sub{sub:02d}.npy")
    indices = sorted(random.sample(range(len(test_fmri)), 100))
    stims = test_imgs[indices]
    del test_fmri

    with open(save_path / "labels_shared.pkl", "rb") as f:
        data = pickle.load(f)
    shared = data["originalorder"]
    ordered = data["ordered"]

    img_alex_path = Path(f"/share/klab/alkroner/brain-diffuser/data/nsddata_stimuli/test_images/subj{sub:02d}")
    vdvae_alex_path = Path(f"/share/klab/alkroner/brain-diffuser/results/vdvae/subj{sub:02d}")
    diff_alex_path = Path(f"/share/klab/alkroner/brain-diffuser/results/versatile_diffusion/subj{sub:02d}")


    diffusion_img_path = Path(f"/share/klab/labstudents/jmihatsch/diffuser_img/subj{sub:02d}/woman_{thr}")
    vdvae_img_path = Path(f"/share/klab/labstudents/jmihatsch/vdvae_img/subj{sub:02d}/woman_{thr}")

    plot_dir = figures_dir / f"comparison_plots_alex/subj{sub:02d}/threshold_{thr}"
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True,exist_ok=True )

    start = 0
    for stim in range(N):
        img = stim*9+4
        vdvae_img = vdvae_img_path / f"vdvae_image_{img}.png"
        diffuser_img = diffusion_img_path / f"diffuser_{img}.png"

        orig_idx = get_alex_index(sub=sub, stim=indices[stim], shared=shared, ordered=ordered)

        orig_img_alex =  img_alex_path / f"{orig_idx}.png"
        vdvae_img_alex = vdvae_alex_path / f"{orig_idx}.png"
        diffuser_img_alex = diff_alex_path / f"{orig_idx}.png"

        fig, axs = plt.subplots(2,3, figsize=(12,8), subplot_kw={'aspect': 1})                    
        axs[0,0].imshow(stims[stim])
        axs[0,0].set_title("Stimulus shown")
        axs[0, 1].imshow(Image.open(vdvae_img))
        axs[0, 1].set_title("VDVAE reconstructed image")
        axs[0, 2].imshow(Image.open(diffuser_img))
        axs[0, 2].set_title("Diffuser reconstructed image")
        axs[1, 0].imshow(Image.open(orig_img_alex))
        axs[1, 1].imshow(Image.open(vdvae_img_alex))
        axs[1, 2].imshow(Image.open(diffuser_img_alex))
        #row labels
        axs[0,0].set_ylabel("Ours", fontsize=14, rotation=90)
        axs[1,0].set_ylabel("Alex", fontsize=14, rotation=90)
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.savefig(plot_dir / f"sub{sub:02d}_{thr}_stim{indices[stim]}_comparison.png", bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved Image {stim}")


def plot_star(sub, thr, concepts):
    print(f"[Info] Plotting star plot for subject {sub}, threshold {thr}")
    random.seed(42)
    N = 100
    c1, c2, c3, c4 = 1, 2, 3, 4

    data_path = Path(f"/share/klab/labstudents/jmihatsch/processed_data/subj{sub:02d}")
    fmri_test_data_path = data_path / f"laion_test_fmriavg_laiongeneral_sub{sub:02d}.npy"
    test_fmri = np.load(fmri_test_data_path)
    test_img_path = data_path / f"laion_test_stim_sub{sub:02d}.npy"
    test_imgs = np.load(test_img_path)

    indices = sorted(random.sample(range(len(test_fmri)), 100))
    stims = test_imgs[indices]

    plot_dir = figures_dir / f"star_plots/subj{sub:02d}/threshold_{thr}"
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True,exist_ok=True )
    
    for stim in range(N):
        img_endpoints = []  
        img_center = None
        start = stim*9
        for pert in concepts:
        
            diffusion_img_path = Path(f"/share/klab/labstudents/jmihatsch/diffuser_img/subj{sub:02d}/{pert}_{thr}")

            paths = []
            for i in range(start, start + 9):
                paths.append(diffusion_img_path / f"diffuser_{i}.png")

            if img_center is None:
                img_center = Image.open(paths[4])

            img_endpoints.append(Image.open(paths[6]))  # c=+2

        fig, ax = plt.subplots(figsize=(24,24))

        theta = np.linspace(0, 2 * np.pi, num=12, endpoint=False)
        arrow_length = 1.2
        arrow_dx = arrow_length * np.cos(theta)
        arrow_dy = arrow_length * np.sin(theta)

        for i in range(12):
            ax.arrow(
                0, 0, arrow_dx[i], arrow_dy[i],
                head_width=0.05, head_length=0.1, fc='k', ec='k', length_includes_head=True,
                zorder=1  # Lower zorder to place arrows behind images
            )
        text_distance = 2*arrow_length / 3
        offset = 0.1

        for i in range(12):
            x_label = (text_distance + offset) * np.cos(theta[i])
            y_label = (text_distance + offset) * np.sin(theta[i])
            angle = np.degrees(theta[i])
            angle = np.degrees(theta[i])
            if (angle > 180 and angle <= 270) or (angle > 90 and angle <= 180) :
                angle -=180
            ax.text(
                x_label, y_label, concepts[i],
                rotation=angle,
                rotation_mode='anchor',  # Keep text aligned with the arrow
                ha='center', va='center',
                fontsize=18, zorder=6,
                bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=1)  # Optional: background for better visibility
            )

        ax.imshow(img_center, extent=[-0.5, 0.5, -0.5, 0.5], zorder=3)  # Higher zorder

        R_image = 1.5
        x_image = R_image * np.cos(theta)
        y_image = R_image * np.sin(theta)

        colors = [
            'red', 'green', 'blue', 'yellow', 'cyan', 'magenta',
            'orange', 'purple', 'brown', 'pink', 'gray', 'olive'
        ]

        image_size = 0.5
        for i in range(12):
            x = x_image[i]
            y = y_image[i]
            extent = [
                x - image_size / 2, x + image_size / 2,
                y - image_size / 2, y + image_size / 2
            ]
            ax.imshow(img_endpoints[i], extent=extent, zorder=3)  # Set zorder appropriately

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis('off')
        plt.savefig(plot_dir / f"sub{sub:02d}_{pert}_{thr}_stim{indices[stim]}_starplot.png", bbox_inches="tight")
        print(f"[INFO] Saved Image {stim}")
        

plot_diffusion_decoded(sub=sub, thr=thr, concepts=concepts)
plot_compare_original_vdvae_decoded(sub=sub, thr=thr, concepts=concepts)
plot_compare_ours_with_alex(sub=sub, thr=thr, concepts=concepts)
#plot_star(sub=sub, thr=thr, concepts=concepts)

