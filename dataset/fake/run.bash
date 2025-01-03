#!/bin/bash

for m in kandinsky2 kandinsky3 pixart-alpha playground-25 sd-15 sd-21 sdxl-dpo sdxl ssd1b stable-cascade vega wurstchen2; do
    python gen.py --prompts=fake_inversion --det_seed --model=${m}
done
