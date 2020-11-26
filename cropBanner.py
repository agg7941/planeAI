import os
import glob
from PIL import Image

path = 'fgvcDataset/'

files = [f for f in glob.glob(path + "**/*.jpg", recursive=True)]

for f in files:
    print(f)
    with Image.open(f) as img:
        w, h = img.size
        img.crop((0, 0, w, h - 20)).save(f)