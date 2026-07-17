# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import os
import json
import argparse
import numpy as np


# def load_data(data_dir, interval=900, data_type='2D'):
#     music_data, dance_data = [], []
#     fnames = sorted(os.listdir(data_dir))
#     # fnames = fnames[:10]  # For debug
#     if ".ipynb_checkpoints" in fnames:
#         fnames.remove(".ipynb_checkpoints")
#     for fname in fnames:
#         path = os.path.join(data_dir, fname)
#         with open(path) as f:
#             sample_dict = json.loads(f.read())
#             np_music = np.array(sample_dict['music_array'])
#             np_dance = np.array(sample_dict['dance_array'])
#             if data_type == '2D':
#                 # Only use 25 keypoints skeleton (basic bone) for 2D
#                 np_dance = np_dance[:, :50]
#                 root = np_dance[:, 2*8:2*9]  # Use the hip keyjoint as the root
#                 np_dance = np_dance - np.tile(root, (1, 25))  # Calculate relative offset with respect to root
#                 np_dance[:, 2*8:2*9] = root
#
#             seq_len, dim = np_music.shape
#             for i in range(0, seq_len, interval):
#                 music_sub_seq = np_music[i: i + interval]
#                 dance_sub_seq = np_dance[i: i + interval]
#                 if len(music_sub_seq) == interval:
#                     music_data.append(music_sub_seq)
#                     dance_data.append(dance_sub_seq)
#
#     return music_data, dance_data


def printer(args):
    for arg in vars(args):
        print(arg, '->', getattr(args, arg))


def load_data(data_dir, interval=900, data_type='2D'):
    music_data, dance_data = [], []
    fnames = sorted(os.listdir(data_dir))
    # print(fnames)
    # fnames = fnames[:10]  # For debug
    if ".ipynb_checkpoints" in fnames:
        fnames.remove(".ipynb_checkpoints")
    for fname in fnames:
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'])
            np_dance = np.array(sample_dict['dance_array'])
            if data_type == '2D':
                # Only use 25 keypoints (x,y) skeleton (basic bone) for 2D
                np_dance = np_dance[:, :50]
                root = np_dance[:, 2*8:2*9]  # Use the hip keyjoint as the root
                np_dance = np_dance - np.tile(root, (1, 25))  # Calculate relative offset with respect to root
                np_dance[:, 2*8:2*9] = root

            if interval is not None:
                seq_len, dim = np_music.shape
                for i in range(0, seq_len, interval):
                    music_sub_seq = np_music[i: i + interval]
                    dance_sub_seq = np_dance[i: i + interval]
                    if len(music_sub_seq) == interval:
                        music_data.append(music_sub_seq)
                        dance_data.append(dance_sub_seq)
            else:
                music_data.append(np_music)
                dance_data.append(np_dance)

    return music_data, dance_data, [fn.replace('.json', '') for fn in fnames]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
