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

#for speaker in ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]:
for speaker in ["F1", "F2", "F3", "F4", "F5"]:
    speaker_path = Path('./data/' + speaker)
    sound_path = speaker_path / "wav"
    feature_path = speaker_path / "mockingjay"
    if not feature_path.exists():
        feature_path.mkdir()
    for f in sound_path.iterdir():
        data, sr = sf.read(f)
        data = torch.from_numpy(data).float()
        h = model([data])["hidden_states"]
        filename = f.parts[-1][:-4]
        np.save(feature_path / filename, h[-1][0].detach().numpy())
        del h
        del data
        gc.collect()

# each wave file has a feature of (l, 768), where l is related to the length of the waveform
# the feature is extracted from the 13th layer of mockingjay model