#!/bin/bash
mkdir data_16 data_16/train data_16/test
unzip -q phantoms_16.zip -d data_16
mv data_16/data/phantoms_16/batch{0..6}.dat data_16/train
mv data_16/data/phantoms_16/batch{7..8}.dat data_16/test
