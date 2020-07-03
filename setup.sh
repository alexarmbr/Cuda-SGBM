#!/bin/bash

wget -r -np -A png,pfm,txt -I "/stereo/data/scenes2014/datasets/Backpack-perfect/,/stereo/data/scenes2014/datasets/Bicycle1-perfect/,/stereo/data/scenes2014/datasets/Adirondack-perfect/" http://vision.middlebury.edu/stereo/data/scenes2014/datasets/
if [ ! -d "data" ]
then
    mkdir data
    echo "created data directory"
fi

mv vision.middlebury.edu/stereo/data/scenes2014/datasets ./data
rm -r vision.middlebury.edu
