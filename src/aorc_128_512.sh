#!/usr/bin/env bash

# aorc 128 to 512 model
python main.py --model EDSR --scale 4 --patch_size 128 --save aorc_128_512 --save_results --save_gt --reset --dir_data J:\Mississippi_design_storm\Super_resolution\SR_diffusion\datasets\aorc_128_512_small_test --data_train aorc512_train --data_test aorc512_test --ext npy --n_colors 1 --n_resblocks 16 --n_feats 64 --res_scale 0.1 --batch_size 4


