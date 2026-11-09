import sys
from revision_load_data import save_avg_fmri_data
from revision_brain_algebra import brain_algebra
from revision_plot import plot_all

subj = int(sys.argv[1])

#save_avg_fmri_data(subj=subj)
#brain_algebra(subj=subj)
plot_all(subj)