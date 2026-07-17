#! /bin/bash

# The output dir to store the generated results
outputs="outputs/demo_song" 
# Please specify your own music directory
music_dir="music/demo_song" 

# Test (Please set dance_num argument to a even number)
python3 predict.py --test_dir ${music_dir}  \
                   --output_dir ${outputs} \
                   --model checkpoints/epoch_best.pt \
                   --dance_num 8

files=$(ls ${outputs})
for filename in $files
do
    IDs=$(ls ${outputs}/${filename})
    for ID in $IDs
    do
        # Make dance video from generated pose data
        ffmpeg -r 30 -i ${outputs}/${filename}/${ID}/img/frame%06d.jpg -vb 20M -vcodec mpeg4 -y ${outputs}/${filename}/${ID}/${ID}.mp4

        # Add background music to dance video
        ffmpeg -i ${outputs}/${filename}/${ID}/${ID}.mp4 -i ${music_dir}/${filename}.m4a -c copy -map 0:v:0 -map 1:a:0 ${outputs}/${filename}/${ID}.mp4
        echo "make video ${filename} ${ID}"

        # Remove dance video without audio track
        rm ${outputs}/${filename}/${ID}/${ID}.mp4
    done
done

echo "Finish!"

