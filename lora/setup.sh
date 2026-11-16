#!/bin/bash
# Link the data into this folder (the scripts read train.csv, dev.csv, ... from the current directory).
cd "$(dirname "$0")"
for f in train.csv dev.csv devtest.csv train_r1.csv train_r2.csv dev_split_group.json minpair_slice.json; do ln -sf ../data/$f $f; done
ln -sf dev_split_group.json dev_split.json   # common2.dev_split() reads dev_split.json
(cd ../data && sha256sum -c SHA256SUMS)
