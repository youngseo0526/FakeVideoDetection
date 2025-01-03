import diffusers, json, os, glob, tqdm

json_files = list(sorted(glob.glob("url_files/*.jsonl")))

total = 0
failed = 0

for file in json_files:
    dataset = os.path.basename(file)[:-6]
    print("Fetching {}...".format(dataset))
    save_dir = "fetched_images/{}".format(dataset)
    os.makedirs(save_dir, exist_ok=True)
    with open(file, 'r') as json_file:
        json_list = list(json_file)

    json_list = list(sorted(json_list, key=lambda x: int(json.loads(x)["prompt_index"])))

    for json_str in tqdm.tqdm(json_list):
        result = json.loads(json_str)
        total += 1
        url = result["url"]
        prompt_index = int(result["prompt_index"])
        save_path = os.path.join(save_dir, "{:05d}.png".format(prompt_index))
        if os.path.exists(save_path):
            print("Skipping existing image at {}".format(save_path))

        try:
            im = diffusers.utils.load_image(url)
            im.verify()
        except:
            failed += 1
            print("Falied to download image {} from {}".format(prompt_index, url))
            print("Failure rate: {}".format(failed / total))
            continue

        im.save(save_path)
