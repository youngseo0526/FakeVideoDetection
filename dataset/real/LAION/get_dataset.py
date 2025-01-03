import os

from dagshub.streaming import DagsHubFilesystem
from PIL import Image

# Setup data streaming from DagsHub
fs = DagsHubFilesystem('.', repo_url='https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus')
fs.install_hooks()

# Get all images + labels.tsv file
files = fs.listdir('data/')

# Get the data for the first 5 images in the labels.tsv file
with fs.open('data/labels.tsv') as tsv:
    for row in tsv.readlines()[3140:5140]:
        row = row.strip()
        img_file, caption, score, url = row.split('\t')

        # Load the image file
        img_path = os.path.join('data', img_file)
        img = Image.open(img_path)
        print(f'{img_file} has a size of {img.size} and an aesthetics score of {score}')