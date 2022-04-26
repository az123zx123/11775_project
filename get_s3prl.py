import torch
import pandas as pd
from pathlib import Path
from numpy import genfromtxt
import numpy as np
import torch
import s3prl.hub as hub
import soundfile as sf
import gc

model = getattr(hub, "mockingjay")()

speaker = 'f2'
speaker_path = Path('./data/' + speaker)
sound_path = speaker_path / "wav"
feature_path = speaker_path / "mockingjay"
for f in sound_path.iterdir():
    data, sr = sf.read(f)
    data = torch.from_numpy(data).float()
    h = model([data])["hidden_states"]
    np.save(feature_path / f.parts[-1], h[-1][0].detach().numpy())
    del h
    del data
    gc.collect()