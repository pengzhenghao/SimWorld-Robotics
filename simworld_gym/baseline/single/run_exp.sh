#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

# UE_PORT=9000 python single_exp.py --backend=gemini --model=gemini-2.5-flash --map=20_0


# python evaluate.py --run_dir gemini-2.5-flash_simple --env simpleenv --split easy

# cd external/SimWorld-Robotics/simworld_gym/baseline/single
# export GEMINI_API_KEY=...

UE_PORT=9000 python sweep_exp.py --backend gemini --model gemini-2.5-flash --env simple \
  --split easy \
  --segment \
  --out_dir gemini-2.5-flash_easy_simple_0107 \
  --tasks task_dist_11_0_1 \
  --record_video \
  --max_maps 1
