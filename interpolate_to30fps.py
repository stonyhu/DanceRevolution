# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from keypoint2img import read_keypoints

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, 
                    default='v2/outputs/layers2_win100_schedule100_condition10_detach/epoch_3000_random',
                    help='the input directory of generated pose sequences under 15 FPS')
parser.add_argument('--output_dir', type=str, 
                    default='v2/outputs/layers2_win100_schedule100_condition10_detach/epoch_3000_random_30fps',
                    help='the output directory of interpolated pose sequences under 30 FPS')
args = parser.parse_args()


def interpolate(frames, stride=10):
    new_frames = []
    for i in range(len(frames) - 1):
        inter_points = np.zeros((25,3))
        left_points = frames[i]
        right_points = frames[i + 1]
        for j in range(len(inter_points)):
            inter_points[j][0] = (left_points[j][0] + right_points[j][0])/2
            inter_points[j][1] = (left_points[j][1] + right_points[j][1])/2
            inter_points[j][2] = (left_points[j][2] + right_points[j][2])/2
        new_frames.append(left_points)
        new_frames.append(inter_points)
    new_frames.append(frames[-1])
    new_frames.append(frames[-1])

    return new_frames


def scale_back(points, width=1280, height=720):
    for point in points:
        point[0] = (point[0] + 1) * 0.5 * width
        point[1] = (point[1] + 1) * 0.5 * height


def store(frames, out_dance_path):
    for i, pose_points in enumerate(frames):
        
        # scale_back(pose_points)
        # scale_back(face_points)
        # scale_back(hand_points_l)
        # scale_back(hand_points_r)

        people_dicts = []
        people_dict = {'pose_keypoints_2d': np.array(pose_points).reshape(-1).tolist(),
                       'face_keypoints_2d': [],
                       'hand_left_keypoints_2d': [],
                       'hand_right_keypoints_2d': [],
                       'pose_keypoints_3d': [],
                       'face_keypoints_3d': [],
                       'hand_left_keypoints_3d': [],
                       'hand_right_keypoints_3d': []}
        people_dicts.append(people_dict)
        frame_dict = {'version': 1.2}
        frame_dict['people'] = people_dicts
        frame_json = json.dumps(frame_dict)
        with open(os.path.join(out_dance_path, f'frame{i:06d}_keypoints.json'), 'w') as f:
            f.write(frame_json)


def modify(data_dir, output_dir):
    sub_dirs = sorted(os.listdir(data_dir))
    for sub_dir in sub_dirs:
        dance_path = os.path.join(data_dir, sub_dir)

        out_dance_path = os.path.join(output_dir, sub_dir)
        if not os.path.exists(out_dance_path):
            os.mkdir(out_dance_path)

        frames = []
        filenames = sorted(os.listdir(dance_path))
        for i, filename in enumerate(filenames):
            json_file = os.path.join(dance_path, filename)
            with open(json_file) as f:
                keypoint_dicts = json.loads(f.read())['people']
               
                keypoint_dict = keypoint_dicts[0]
                pose_points = np.array(keypoint_dict['pose_keypoints_2d']).reshape(25, 3).tolist()
                
                frames.append(pose_points)
               
        # Recover the missing key points
        frames = interpolate(frames)
        # Store the corrected frames
        store(frames, out_dance_path)


if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    modify(args.input_dir, args.output_dir)
