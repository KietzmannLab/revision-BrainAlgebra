import sys
from revision_load_data import save_avg_fmri_data

subj = int(sys.argv[1])
save_avg_fmri_data(subj=subj)