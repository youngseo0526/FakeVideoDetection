import os
os.environ['TMPDIR'] = '/var/tmp'  # 임시 파일 저장 경로 변경
os.environ['HF_DATASETS_CACHE'] = '/var/tmp/hf_datasets'  # Hugging Face 데이터셋 캐시 경로 변경
os.environ['HF_HOME'] = '/var/tmp/hf_home'  # Hugging Face 모델 및 기타 파일 경로 변경

import tqdm
import datasets

dataset = datasets.load_dataset("OpenDatasets/dalle-3-dataset", ignore_verifications=True)

ds = dataset['train'].shard(num_shards=5, index=0)

os.makedirs("dalle3", exist_ok=True)
for i, row in tqdm.tqdm(enumerate(ds.select(range(3_000))), total=3_000):
    im = row['image']
    im.save(os.path.join("dalle3", "{:05d}.png".format(i)))

print("Data saved successfully in 'dalle3' directory.")
