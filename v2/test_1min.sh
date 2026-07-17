#! /bin/bash

audio_dir="../data/audio"
version="layers1_win900_schedule100_condition10_detach"
e="5500"
random="_random"
synthesized_video_dir="visualizations/${version}/videos_epoch_${e}${random}"

mkdir -p ${synthesized_video_dir}

# Test
python3 test.py --train_dir ../data/train_1min \
                --test_dir ../data/test_1min \
                --output_dir outputs/${version}/epoch_${e}${random} \
                --visualize_dir visualizations/${version}/epoch_${e}${random} \
                --model checkpoints/${version}/epoch_${e}.pt

files=$(ls visualizations/${version}/epoch_${e}${random})
for filename in $files
do
    ffmpeg -r 15 -i visualizations/${version}/epoch_${e}${random}/${filename}/frame%06d.jpg -vb 20M -vcodec mpeg4 \
        -y visualizations/${version}/epoch_${e}${random}/${filename}.mp4

    audio_type=`echo ${filename} | cut -d \_ -f 1,2`
    audio_filename=`echo ${filename} | cut -d \_ -f 3,4`

    ffmpeg -i visualizations/${version}/epoch_${e}${random}/${filename}.mp4 -i ${audio_dir}/${audio_type}/${audio_filename}.m4a \
        -c copy -map 0:v:0 -map 1:a:0 ${synthesized_video_dir}/${filename}.mp4

    echo "make video ${filename}"
done

