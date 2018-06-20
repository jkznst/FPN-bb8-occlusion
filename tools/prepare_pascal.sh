#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python $DIR/prepare_dataset.py --dataset pascal --year 2012 --set train --target $DIR/../data/train.lst
python $DIR/prepare_dataset.py --dataset pascal --year 2012 --set val --target $DIR/../data/val.lst --shuffle False
