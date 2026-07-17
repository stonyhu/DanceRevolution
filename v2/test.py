# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the test process. """
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from dataset import DanceDataset, paired_collate_fn
from utils.functional import str2bool, load_test_data
from generator import Generator
from PIL import Image
from keypoint2img import read_keypoints
from multiprocessing import Pool
from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='data/train_1min',
                    help='the directory of train data')
parser.add_argument('--test_dir', type=str, default='data/test_1min',
                    help='the directory of test data')
parser.add_argument('--data_type', type=str, default='2D',
                    help='the type of training data')
parser.add_argument('--output_dir', type=str, default='outputs',
                    help='the directory of generated result')
parser.add_argument('--visualize_dir', type=str,
                    default='visualizations/layers2_win100_schedule100_condition10_detach',
                    help='the output directory of visualization')
parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument('--model', type=str, metavar='PATH',
                    default='checkpoints/layers2_win100_schedule100_condition10_detach/epoch_3000.pt')
parser.add_argument('--batch_size', type=int, metavar='N', default=1)
parser.add_argument('--seq_len', type=int, metavar='N', default=900)
parser.add_argument('--width', type=int, default=1280,
                    help='the width pixels of target image')
parser.add_argument('--height', type=int, default=720,
                    help='the height pixels of target image')

args = parser.parse_args()
print(args)

pose_keypoints_num = 25
face_keypoints_num = 70
hand_left_keypoints_num = 21
hand_right_keypoints_num = 21


def visualize_json(fname, output_dir, dance_dir, dance_path):
    j, fname = fname
    json_file = os.path.join(dance_path, fname)
    img = Image.fromarray(read_keypoints(json_file, (args.width, args.height),
                                         remove_face_labels=False, basic_point_only=False))
    img.save(os.path.join(f'{output_dir}/{dance_dir}', f'frame{j:06d}.jpg'))


def visualize(data_dir, output_dir, worker_num=16):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dance_dirs = sorted(os.listdir(data_dir))
    for i, dance_dir in enumerate(dance_dirs):
        dance_path = os.path.join(data_dir, dance_dir)
        fnames = sorted(os.listdir(dance_path))
        if not os.path.exists(f'{output_dir}/{dance_dir}'):
            os.makedirs(f'{output_dir}/{dance_dir}')

        # Visualize json in parallel
        pool = Pool(worker_num)
        partial_func = partial(visualize_json, output_dir=output_dir,
                               dance_dir=dance_dir, dance_path=dance_path)
        pool.map(partial_func, enumerate(fnames))
        pool.close()
        pool.join()

        print(f'visualize {dance_dir}')


def write2json(dances, dir_names):
    assert len(dances) == len(dir_names),\
        "number of generated dance != number of dir_names"
    for i in range(len(dances)):
        num_poses = dances[i].shape[0]
        dances[i] = dances[i].reshape(num_poses, pose_keypoints_num, 2)
        dance_path = os.path.join(args.output_dir, dir_names[i])
        if not os.path.exists(dance_path):
            os.makedirs(dance_path)

        for j in range(num_poses):
            frame_dict = {'version': 1.2}
            # 2-D key points
            pose_keypoints_2d = []
            # Random values for the below key points
            face_keypoints_2d = []
            hand_left_keypoints_2d = []
            hand_right_keypoints_2d = []
            # 3-D key points
            pose_keypoints_3d = []
            face_keypoints_3d = []
            hand_left_keypoints_3d = []
            hand_right_keypoints_3d = []

            keypoints = dances[i][j]
            for k, keypoint in enumerate(keypoints):
                x = (keypoint[0] + 1) * 0.5 * args.width
                y = (keypoint[1] + 1) * 0.5 * args.height
                score = 0.8
                if k < pose_keypoints_num:
                    pose_keypoints_2d.extend([x, y, score])
                elif k < pose_keypoints_num + face_keypoints_num:
                    face_keypoints_2d.extend([x, y, score])
                elif k < pose_keypoints_num + face_keypoints_num + hand_left_keypoints_num:
                    hand_left_keypoints_2d.extend([x, y, score])
                else:
                    hand_right_keypoints_2d.extend([x, y, score])

            people_dicts = []
            people_dict = {'pose_keypoints_2d': pose_keypoints_2d,
                           'face_keypoints_2d': face_keypoints_2d,
                           'hand_left_keypoints_2d': hand_left_keypoints_2d,
                           'hand_right_keypoints_2d': hand_right_keypoints_2d,
                           'pose_keypoints_3d': pose_keypoints_3d,
                           'face_keypoints_3d': face_keypoints_3d,
                           'hand_left_keypoints_3d': hand_left_keypoints_3d,
                           'hand_right_keypoints_3d': hand_right_keypoints_3d}
            people_dicts.append(people_dict)
            frame_dict['people'] = people_dicts
            frame_json = json.dumps(frame_dict)
            with open(os.path.join(dance_path, f'frame{j:06d}_keypoints.json'), 'w') as f:
                f.write(frame_json)
        print(f'finished writing to json {i:02d}')


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    music_data, dance_data, dir_names = load_test_data(
        args.test_dir, data_type=args.data_type
    )

    device = torch.device('cuda' if args.cuda else 'cpu')

    test_loader = torch.utils.data.DataLoader(
        DanceDataset(music_data, dance_data),
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn
    )

    generator = Generator(args, device)

    results = []
    for batch in tqdm(test_loader, desc='Generating dance poses'):
        # Prepare data
        src_seq, src_pos, tgt_seq = map(lambda x: x.to(device), batch)
        # Choose the first 10 frames as the beginning
        tgt_seq = tgt_seq[:, :10, :]
        # Change the way to pass tgt_seq
        poses = generator.generate(src_seq, src_pos, tgt_seq=tgt_seq)
        results.append(poses)

    # Visualize the generated dance poses
    np_dances = []
    for i in range(len(results)):
        np_dance = results[i][0].data.cpu().numpy()
        root = np_dance[:, 2*8:2*9]
        np_dance = np_dance + np.tile(root, (1, 25))
        np_dance[:, 2*8:2*9] = root

        np_dances.append(np_dance)
    write2json(np_dances, dir_names)

    visualize(args.output_dir, args.visualize_dir)


if __name__ == '__main__':
    main()
