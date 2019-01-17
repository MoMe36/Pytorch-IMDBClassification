#!/bin/bash 


path=$1

ffmpeg -y -framerate 5 -i "${path}"%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "${path}"output.mp4
# rm "${path}"*.png