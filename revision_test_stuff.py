import torch
from pathlib import Path
import pickle


path = Path("/share/klab/alkroner/brain-diffuser/data/regression_weights/subj01/vdvae_regression_weights.pkl")

with open(path, 'rb') as file:
    data = pickle.load(file)

print(type(data))

print(data)