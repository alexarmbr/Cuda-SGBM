#!/bin/bash

wget -r -np --no-check-certificate -A png,pfm,txt http://vision.middlebury.edu/stereo/data/scenes2014/datasets/Backpack-perfect/
if [ ! -d "data" ]
then
    mkdir data
    echo "created data directory"
fi

mv vision.middlebury.edu/stereo/data/scenes2014/datasets ./data
mv  data/datasets/Backpack-perfect data/
rm -r vision.middlebury.edu
rm -r data/datasets
