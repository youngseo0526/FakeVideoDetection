import os
from glob import glob
import argparse
import numpy as np
import torch
import natsort
from PIL import Image

from lavis.models import load_model_and_preprocess

from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/yskim/CNNDetection/dataset')
    parser.add_argument('--results_folder', type=str, default='output')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--use_float_16', action='store_true')
    args = parser.parse_args()

    real_results_folder= os.path.join(args.results_folder, "real_lsun_5k")
    fake_results_folder= os.path.join(args.results_folder, "fake_progan_5k")

    os.makedirs(os.path.join(real_results_folder, "inversion"), exist_ok=True)
    os.makedirs(os.path.join(real_results_folder, "prompt"), exist_ok=True)
    os.makedirs(os.path.join(fake_results_folder, "inversion"), exist_ok=True)
    os.makedirs(os.path.join(fake_results_folder, "prompt"), exist_ok=True)

    #import pdb; pdb.set_trace()
    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(device)) 
    
    pipe = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch_dtype, use_auth_token=True).to(device)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    train_dir = os.path.join(args.data_dir, "train")
    class_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        real_dir = os.path.join(class_dir, "0_real")
        fake_dir = os.path.join(class_dir, "1_fake")

        # 0_real 처리
        if os.path.exists(real_dir):
            real_images = natsort.natsorted(glob(os.path.join(real_dir, "*.png")))[:250] # 20 class * 250 images = 5,000 for real 
            for img_path in real_images:
                bname = f"{class_name}_real_{os.path.basename(img_path).split('.')[0]}"
                img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)

                _image = vis_processors["eval"](img).unsqueeze(0).to(device)
                prompt_str = model_blip.generate({"image": _image})[0]

                x_inv, x_inv_image, x_dec_img = pipe(
                    prompt_str, 
                    guidance_scale=1,
                    num_inversion_steps=args.num_ddim_steps,
                    img=img,
                    torch_dtype=torch_dtype
                )

                with open(os.path.join(real_results_folder, f"prompt/{bname}.txt"), "w") as f:
                    f.write(prompt_str)

                img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)
                img_tensor = (img_tensor / 255.0 - 0.5) * 2  # Normalize to [-1, 1]
                x_inv_image_tensor = torch.from_numpy(np.array(x_inv_image[0]).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)
                x_dec_img_tensor = torch.from_numpy(np.array(x_dec_img[0]).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)
                concatenated = torch.cat((img_tensor, x_inv_image_tensor, x_dec_img_tensor), dim=1)

                torch.save(concatenated, os.path.join(real_results_folder, f"inversion/{bname}_concat.pt"))
                print(f"Concatenated tensor saved at {os.path.join(real_results_folder, f'inversion/{bname}_concat.pt')}.")

        # 1_fake 처리
        if os.path.exists(fake_dir):
            fake_images = natsort.natsorted(glob(os.path.join(fake_dir, "*.png")))[:250] # 20 class * 250 images = 5,000 for fake 
            for img_path in fake_images:
                bname = f"{class_name}_fake_{os.path.basename(img_path).split('.')[0]}"
                img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)

                _image = vis_processors["eval"](img).unsqueeze(0).to(device)
                prompt_str = model_blip.generate({"image": _image})[0]

                x_inv, x_inv_image, x_dec_img = pipe(
                    prompt_str, 
                    guidance_scale=1,
                    num_inversion_steps=args.num_ddim_steps,
                    img=img,
                    torch_dtype=torch_dtype
                )

                with open(os.path.join(fake_results_folder, f"prompt/{bname}.txt"), "w") as f:
                    f.write(prompt_str)

                img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)
                img_tensor = (img_tensor / 255.0 - 0.5) * 2  # Normalize to [-1, 1]
                x_inv_image_tensor = torch.from_numpy(np.array(x_inv_image[0]).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)
                x_dec_img_tensor = torch.from_numpy(np.array(x_dec_img[0]).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)
                concatenated = torch.cat((img_tensor, x_inv_image_tensor, x_dec_img_tensor), dim=1)

                torch.save(concatenated, os.path.join(fake_results_folder, f"inversion/{bname}_concat.pt"))
                print(f"Concatenated tensor saved at {os.path.join(fake_results_folder, f'inversion/{bname}_concat.pt')}.")
