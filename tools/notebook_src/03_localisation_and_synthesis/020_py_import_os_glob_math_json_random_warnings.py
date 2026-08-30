import os, glob, math, json, random, warnings, time, zlib, shutil, csv, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)
def IMAGE_TO_TENSOR(im):
    """PIL RGB -> C x H x W float tensor in [0,1].

    This is all torchvision's ToTensor did here, and importing torchvision for one call makes
    the notebook fail on any environment that has torch but not torchvision.
    """
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()

N_CPU = max(1, (os.cpu_count() or 2))
print(f"torch {torch.__version__} | cv2 {cv2.__version__} | seed {SEED}")
print(f"cuda: {torch.cuda.is_available()} "
      f"({torch.cuda.device_count()} device(s))" if torch.cuda.is_available() else "cuda: no")
print(f"cpu workers available: {N_CPU}")
