import torch
from pathlib import Path

path = Path("/share/klab/labstudents/jmihatsch/embeds")

embeds = torch.load(path / "image_embeds_subject-01.pt")
print(embeds.shape)



