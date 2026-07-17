#! /bin/bash
image_dir=outputs_alpha0.01_images

# Generate pose sequences of json format and visualize them into image
python test.py --input_dir data/test \
               --model checkpoints_windowsize100_hidden1024_fixedsteps10_alpha0.01/epoch_3000.pt \
               --json_dir outputs_alpha0.01_epoch3000 \
               --image_dir ${image_dir} \
               --batch_size 1


# Merge visualized images into dance videos
dances=$(ls ${image_dir})
for dance in ${dances}
do
    ffmpeg -r 15 -i ${image_dir}/${dance}/frame%06d.jpg -vb 20M -vcodec mpeg4 -y ${image_dir}/${dance}.mp4
    echo "make video {images/${dance}.mp4}"
done
