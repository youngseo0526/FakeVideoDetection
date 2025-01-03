# LAION-Aesthetics V2 (6.5+)

[LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) is a subset of the [LAION 5B dataset](https://laion.ai/blog/laion-5b/) with high visual quality. 

The dataset in this repo is specifically the images that scored 6.5 or higher via aesthetics prediction models. These models were trained to predict the rating people gave when asked “How much do you like this image on a scale from 1 to 10?”.

The data is composed of image-caption pairs along with an aesthetics score for each image and the original URL, from which it was downloaded.

## Example usage

All of the images can be found under the [data/](https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus/src/main/data) folder. 

A TSV-formatted labels file can also be found there ([data/labels.tsv](https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus/src/main/data/labels.tsv).

This TSV file contains 4 columns: image file name, caption, aesthetics score, url


```python
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
    for row in tsv.readlines()[:5]:
        row = row.strip()
        img_file, caption, score, url = row.split('\t')

        # Load the image file
        img_path = os.path.join('data', img_file)
        img = Image.open(img_path)
        print(f'{img_file} has a size of {img.size} and an aesthetics score of {score}')
```


## Further information

For further information, see the [LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/) project page. This dataset was created by [LAION](https://laion.ai/).
