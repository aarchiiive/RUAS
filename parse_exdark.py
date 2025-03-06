import os
from pathlib import Path


data_dir = Path('Exdark')
save_dir = Path('Exdark_Train')
save_dir.mkdir(exist_ok=True, parents=True)

meta_file = data_dir / 'main/train.txt'

with open(meta_file, 'r') as f:
    meta_lines = f.readlines()

for line in meta_lines:
    line = line.strip().split()
    filename = Path(line[0]).name
    image_path = data_dir / 'JPEGImages' / 'IMGS_dark' / filename
    os.symlink(image_path.resolve(), save_dir / filename)

