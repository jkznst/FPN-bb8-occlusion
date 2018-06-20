#!/usr/bin/env bash
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 train.py --train-path ./data/OCCLUSION/train.rec --val-path ./data/OCCLUSION/val.rec --network resnet50m --batch-size 8 --pretrained ./model/resnet-50 --epoch 0 --data-shape 300 --lr 0.001 --class-names 'obj_01, obj_02, obj_05, obj_06, obj_08, obj_09, obj_11, obj_12' --prefix ./output/OCCLUSION/resnet50m-300-lr0.001-alpha10-wd0.0005/ssd --gpu 1 --alpha-bb8 10.0 --end-epoch 45 --lr-steps '30,40' --wd 0.0005  --num-class 8


