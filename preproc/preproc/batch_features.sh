#!/bin/bash

config_dir="/Users/dbernardo/Documents/pyres/eeggpt/preproc/preproc/config/"
search_dirs=("/Users/dbernardo/Documents/pyres/eeggpt/preproc/data/train")
# search_dirs=("/Users/dbernardo/Documents/pyres/eeggpt/preproc/data/eval" "/Users/dbernardo/Documents/pyres/eeggpt/preproc/data/train")

for search_dir in "${search_dirs[@]}"
do
  datatype=$(basename "$search_dir")
  for entry in "$config_dir"*.yml
  do
    echo "config file $entry : subject $subject"
    python generate_features_TUH.py -s "$subject" -n 8 -c "$entry" -m 1 -d "$datatype"
    # python generate_features_TUH.py -s "$subject" -n 8 -c "$entry" -d "$datatype" -r 1

  done
done
