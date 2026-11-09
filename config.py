from pathlib import Path

BASE_PATH = Path("/share/klab/labstudents/jmihatsch")
SUBJECT_PREFIX = "subj{subject:02d}"


# Data
PROCESSED_DATA = BASE_PATH / "processed_data"
EMBEDS_PATH = BASE_PATH / "embeds"
ROI_PATH = BASE_PATH / "roi_mask"
FMRI_ALGEBRA_PATH = BASE_PATH / "fmri_algebra"
SIMILARITY_DATA_PATH = PROCESSED_DATA / "similarity_analysis_data"

# Weights
WEIGHTS_PATH = BASE_PATH / "regression_weights"

VDVAE_WEIGHTS = WEIGHTS_PATH / "pred_vdvae"
CLIPVISION_WEIGHTS = WEIGHTS_PATH / "pred_vision"
CLIPTEXT_WEIGHTS = WEIGHTS_PATH / "pred_text"

# Figures
FIGURES_PATH = BASE_PATH / "figures"

COMPARISON_PLOTS = FIGURES_PATH / "comparison_plots"
COMPARISON_PLOTS_ALEX = FIGURES_PATH / "comparison_plots_alex"
DIFFUSION_PLOTS = FIGURES_PATH / "diffusion_plots"
SIMILARITY_ANALYSIS_PLOTS = FIGURES_PATH / "similarity_analysis_plots"
STAR_PLOTS = FIGURES_PATH / "star_plots"
CONCEPT_IMAGES = FIGURES_PATH / "concept_images"
ROI_PLOTS = FIGURES_PATH / "roi_plots"

# Images
IMAGES_PATH = BASE_PATH / "reconstructed_images"

VDVAE_IMAGES = IMAGES_PATH / "vdvae_images"
DIFFUSER_IMAGES = IMAGES_PATH / "diffuser_images"


#formating functions

def subject_path(root, subject):
    path = root / SUBJECT_PREFIX.format(subject=subject)
    path.mkdir(parents=True, exist_ok=True)
    return path

def processed_data_path(subject):
    return subject_path(PROCESSED_DATA, subject)


def difference_path(subj, pert=None, thr=None):
    path = subject_path(FMRI_ALGEBRA_PATH, subj)

    if pert is not None:
        path = path / f"{pert}_thr_{thr}"

    path.mkdir(parents=True, exist_ok=True)
    return path