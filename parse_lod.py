import os
from pathlib import Path


data_dir = Path('/home/ubuntu/data/LOD')
save_dir = Path('LOD_train')
save_dir.mkdir(exist_ok=True, parents=True)

meta_file = data_dir / 'train.txt'

with open(meta_file, 'r') as f:
    meta_lines = f.readlines()

for line in meta_lines:
    line = line.strip().split()
    filename = Path(line[0]).name
    image_path = data_dir / 'RGB_Dark' / filename
    os.symlink(image_path, save_dir / filename)

