import datasets

dataset = datasets.load_dataset("wanng/midjourney-v5-202304-clean", split="train").filter(lambda example: example["upscaled"]).shuffle(seed=42)

prompts = [row['clean_prompts'] for row in dataset.select(range(5_000))]

with open("fake_inversion_prompts.txt", "w") as f:
    f.write("\n".join(prompts))
