#!/usr/bin/env bash
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 evaluate.py --rec-path ./data/OCCLUSION/val.rec --network resnet50 --batch-size 64 --epoch 45 --data-shape 512 --class-names 'obj_01, obj_02, obj_05, obj_06, obj_08, obj_09, obj_11, obj_12' --prefix ./output/resnet50-512-exp4/ssd --gpu 0 --num-class 8


