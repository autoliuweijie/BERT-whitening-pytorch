#!/bin/bash

# CUDA_VISIBLE_DEVICES='0' nohup python3 -u ./eval_without_whitening.py > ./tmp/eval_without_whitening.log &
# CUDA_VISIBLE_DEVICES='0' nohup python3 -u ./eval_with_whitening\(target\).py > ./tmp/eval_with_whitening\(target\).log &
CUDA_VISIBLE_DEVICES='0' nohup python3 -u ./eval_with_whitening\(nli\).py > ./tmp/eval_with_whitening\(nli\).log &

