import os
from glob import glob
import argparse
import numpy as np
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input_image', type=str, default='/data/yskim/easy-diffusion-generation/generated_images/fake_inversion_prompts/sd-15_det-seed')
    parser.add_argument('--input_image', type=str, default='/data/yskim/LAION-Aesthetics-V2-6.5plus/data')
    parser.add_argument('--results_folder', type=str, default='output/real_laion')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--use_float_16', action='store_true')
    args = parser.parse_args()

    # make the output folders
    os.makedirs(os.path.join(args.results_folder, "inversion"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "prompt"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # load the BLIP model
    model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(device))
    # make the DDIM inversion pipeline    
    
    pipe = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch_dtype, use_auth_token=True).to(device)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    # if the input is a folder, collect all the images as a list
    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        l_img_paths = [args.input_image]

    for img_path in l_img_paths:
        bname = os.path.basename(img_path).split(".")[0]
        #img = Image.open(img_path).resize((512,512), Image.Resampling.LANCZOS)
        img = Image.open(img_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        
        # generate the caption
        _image = vis_processors["eval"](img).unsqueeze(0).to(device)
        prompt_str = model_blip.generate({"image": _image})[0]
        x_inv, x_inv_image, x_dec_img = pipe(
            prompt_str, 
            guidance_scale=1,
            num_inversion_steps=args.num_ddim_steps,
            img=img,
            torch_dtype=torch_dtype
        )
    
        # save the inversion
        #torch.save(x_inv[0], os.path.join(args.results_folder, f"inversion/{bname}.pt"))
        # save the prompt string
        with open(os.path.join(args.results_folder, f"prompt/{bname}.txt"), "w") as f:
            f.write(prompt_str)

        #import pdb; pdb.set_trace()
        # Convert img to tensor
        img_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)
        img_tensor = (img_tensor / 255.0 - 0.5) * 2  # Normalize to [-1, 1]

        # Concatenate img, x_inv_image, x_dec_img along channel dimension
        x_inv_image_tensor = torch.from_numpy(np.array(x_inv_image[0]).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)
        x_dec_img_tensor = torch.from_numpy(np.array(x_dec_img[0]).transpose(2, 0, 1)).unsqueeze(0).to(torch_dtype).to(device)

        concatenated = torch.cat((img_tensor, x_inv_image_tensor, x_dec_img_tensor), dim=1)

        # Save concatenated tensor
        torch.save(concatenated, os.path.join(args.results_folder, f"inversion/{bname}_concat.pt"))
        print(f"Concatenated tensor saved at {os.path.join(args.results_folder, f'inversion/{bname}_concat.pt')}.")
