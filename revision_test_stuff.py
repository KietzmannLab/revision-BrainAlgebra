from pathlib import Path
import torch

subj = 1
path = Path("/share/klab/labstudents/jmihatsch/embeds")
embeds = torch.load(path / f"image_embeds_subject-{subj:02d}.pt")
print(embeds)
