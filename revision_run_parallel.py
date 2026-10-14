import sys
from revision_load_data import save_avg_fmri_data
from revision_clip_encode import clip_encode

subj = int(sys.argv[1])

#save_avg_fmri_data(subj=subj)
clip_encode(subj=subj)