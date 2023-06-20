#!/bin/bash

#ffmpeg -i $1 -filter_complex "color=c=black:s=640x480:d=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $1),format=yuv420p[v]" -map "[v]" -map 0:a -c:v mpeg4 -c:a aac $2

ffmpeg -i $1 -filter_complex "[0:a]showwaves=s=640x480:mode=line,format=yuv420p[v]" -map "[v]" -map 0:a -c:v mpeg4 -c:a aac $2

