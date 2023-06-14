#!/usr/bin/env bash

python E:/Code/EDSR-PyTorch/src/main.py --model EDSR --skip_psnr \
--dir_data J:\Mississippi_design_storm\Super_resolution\cesm_sr_prediction\1251_11 \
--data_test era128_test \
--save era5_32_128_test --save_results --scale 4 --patch_size 128 --n_feats 256 --res_scale 0.1 --n_resblocks 32 \
 --pre_train J:\Mississippi_design_storm\Super_resolution\EDSR_model\prediction\trained_models\era5_32_128/exp_7709_era_32_128_E300.pt --test_only