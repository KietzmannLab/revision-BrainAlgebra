from pathlib import Path
import torch

'''subj = 1
path = Path("/share/klab/labstudents/jmihatsch/embeds")
embeds = torch.load(path / f"image_embeds_subject-{subj:02d}.pt")
print(embeds)'''

project = PROJ_ROOT = Path(__file__).resolve().parents[1]
username = project.name
#log.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

import socket
sock = socket.gethostname() #.startswith("cipppy"):
print("proj", username)
print("sock", sock)