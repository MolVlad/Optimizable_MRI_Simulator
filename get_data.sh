#!/bin/bash
gdown 1GOQIUB5otuXCwBkj6uNABu90nJ2e6tyB
mkdir data data/train data/test
unzip -q data.zip -d data
mv data/batch{0..6}.dat data/train
mv data/batch{7..8}.dat data/test
