#!/bin/bash
# This script is used to create multiple zip files from multiple folders in the same directory. Multiple processes are used to speed up the process.

# MAC path
#path="/Users/wanzelin/办公/Drone-DRL-HT/data/data_sensitivity_analysis"
# Server path
path=$(pwd)"/../data"
folder_set=$path"/*"

mkdir $path/zip_folder

for folder in $folder_set
do
  folder_name=$(basename $folder)
  echo $folder_name
  tar -cvzf $path/zip_folder/$folder_name.tar.gz $folder &
done
