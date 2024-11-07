#!/bin/bash
for dataset in "train" "test" "val"
do
  for i in 0 1
  do
   echo "Processing $dataset $i"
    python preprocess.py \
      --class_label $i \
      --tag pt80_eta60_phi60_lower001 \
      --nBins 40 30 30 \
      --input_file /hpcwork/rwth0934/top_benchmark/$dataset.h5 \
      --lower_q 0.001 \
      --upper_q 1.0
  done
done
