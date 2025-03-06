import os
from pathlib import Path


data_dir = Path('/home/ubuntu/data/DarkFace_Train_2021')
save_dir = Path('DarkFace_Train')
save_dir.mkdir(exist_ok=True, parents=True)

meta_file = data_dir / 'mf_dsfd_dark_face_train_5500.txt'

with open(meta_file, 'r') as f:
    meta_lines = f.readlines()

for line in meta_lines:
    line = line.strip().split()
    filename = Path(line[0]).name
    image_path = data_dir / 'image' / filename
    os.symlink(image_path, save_dir / filename)

