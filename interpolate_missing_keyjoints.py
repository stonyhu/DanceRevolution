# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import os
import json
import argparse
import numpy as np
from PIL import Image
from keypoint2img import read_keypoints
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='data/json/ballet_1min')
parser.add_argument('--output_dir', type=str, default='data/json/ballet_1min_inter')
args = parser.parse_args()


def interpolate(frames, stride=10):
    for i in range(len(frames)):
        pose_points = frames[i][0]

        for j, point in enumerate(pose_points):
            if point[0] == -1 and point[1] == -1:
                k1 = i
                while k1 > i - stride and k1 >= 0:
                    tmp_point = frames[k1][0][j]
                    if tmp_point[0] != -1 or tmp_point[1] != -1:
                        break
                    k1 -= 1

                k2 = i
                while k2 < i + stride and k2 <= len(frames) - 1:
                    tmp_point = frames[k2][0][j]
                    if tmp_point[0] != -1 or tmp_point[1] != -1:
                        break
                    k2 += 1

                if k1 == -1 and k2 < i + stride:
                    target_right_point = frames[k2][0][j]
                    point[0] = target_right_point[0]
                    point[1] = target_right_point[1]
                    point[2] = target_right_point[2]
                if k1 > i - stride and k2 == len(frames):
                    target_left_point = frames[k1][0][j]
                    point[0] = target_left_point[0]
                    point[1] = target_left_point[1]
                    point[2] = target_left_point[2]

                if (k1 > i - stride and k1 >= 0) and (k2 < i + stride and k2 <= len(frames) - 1):
                    target_left_point = frames[k1][0][j]
                    target_right_point = frames[k2][0][j]
                    point[0] = (target_left_point[0] + target_right_point[0]) / 2
                    point[1] = (target_left_point[1] + target_right_point[1]) / 2
                    point[2] = (target_left_point[2] + target_right_point[2]) / 2

                if (k1 > i - stride and k1 >= 0) and (k2 == i + stride and k2 <= len(frames) - 1):
                    target_left_point = frames[k1][0][j]
                    point[0] = target_left_point[0]
                    point[1] = target_left_point[1]
                    point[2] = target_left_point[2]

                if (k1 == i - stride and k1 >= 0) and (k2 < i + stride and k2 <= len(frames) - 1):
                    target_right_point = frames[k2][0][j]
                    point[0] = target_right_point[0]
                    point[1] = target_right_point[1]
                    point[2] = target_right_point[2]

                # print('interpolate')


def scale_back(points, width=1280, height=720):
    for point in points:
        point[0] = (point[0] + 1) * 0.5 * width
        point[1] = (point[1] + 1) * 0.5 * height


def store(frames, out_dance_path):
    for i, frame in enumerate(frames):
        pose_points, face_points, hand_points_l, hand_points_r = frame

        # scale_back(pose_points)
        # scale_back(face_points)
        # scale_back(hand_points_l)
        # scale_back(hand_points_r)

        people_dicts = []
        people_dict = {'pose_keypoints_2d': np.array(pose_points).reshape(-1).tolist(),
                       'face_keypoints_2d': np.array(face_points).reshape(-1).tolist(),
                       'hand_left_keypoints_2d': np.array(hand_points_l).reshape(-1).tolist(),
                       'hand_right_keypoints_2d': np.array(hand_points_r).reshape(-1).tolist(),
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


def modify(input_dir, output_dir):
    sub_dirs = sorted(os.listdir(input_dir))
    for sub_dir in sub_dirs:
        dance_path = os.path.join(input_dir, sub_dir)

        out_dance_path = os.path.join(output_dir, sub_dir)
        if not os.path.exists(out_dance_path):
            os.mkdir(out_dance_path)

        frames = []
        filenames = sorted(os.listdir(dance_path))
        for i, filename in enumerate(filenames):
            json_file = os.path.join(dance_path, filename)
            with open(json_file) as f:
                keypoint_dicts = json.loads(f.read())['people']
                if len(keypoint_dicts) > 0:
                    keypoint_dict = keypoint_dicts[0]
                    pose_points = np.array(keypoint_dict['pose_keypoints_2d']).reshape(25, 3).tolist()
                    face_points = np.array(keypoint_dict["face_keypoints_2d"]).reshape(70, 3).tolist()
                    hand_points_l = np.array(keypoint_dict["hand_left_keypoints_2d"]).reshape(21, 3).tolist()
                    hand_points_r = np.array(keypoint_dict["hand_right_keypoints_2d"]).reshape(21, 3).tolist()
                    frames.append((pose_points, face_points, hand_points_l, hand_points_r))
                    
                else:
                    pose_points=np.array([-1,-1,0]*25)
                    pose_points = pose_points.reshape(25,3).tolist()
                    face_points = np.zeros((70,3)).tolist()
                    hand_points_l = np.zeros((21,3)).tolist()
                    hand_points_r = np.zeros((21,3)).tolist()
                    frames.append((pose_points, face_points, hand_points_l, hand_points_r))
                   
        # Recover the missing key points
        interpolate(frames)
        # Store the corrected frames
        store(frames, out_dance_path)


if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    modify(args.input_dir, args.output_dir)
