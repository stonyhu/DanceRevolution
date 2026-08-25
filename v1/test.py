# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from dataset import DanceDataset, paired_collate_fn
from utils.functional import str2bool, load_data
from generator import Generator
from PIL import Image
from keypoint2img import read_keypoints
from multiprocessing import Pool
from functools import partial


pose_keypoints_num = 25
face_keypoints_num = 70
hand_left_keypoints_num = 21
hand_right_keypoints_num = 21


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='data/test')
    parser.add_argument('--data_type', type=str, default='2D', help='the type of pose data')
    parser.add_argument('--model', type=str, metavar='PATH', default='checkpoints/epoch_3000.pt')
    parser.add_argument('--json_dir', metavar='PATH', default='outputs/',
                        help='the generated pose data of json format')
    parser.add_argument('--image_dir', type=str, default='images',
                        help='the directory of visualization image')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--width', type=int, default=1280, help='the width pixels of image')
    parser.add_argument('--height', type=int, default=720, help='the height pixels of image')

    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')

    return parser.parse_args()


def visualize_json(fname_iter, image_dir, dance_name, dance_path, args):
    j, fname = fname_iter
    json_file = os.path.join(dance_path, fname)
    img = Image.fromarray(read_keypoints(json_file, (args.width, args.height),
                                         remove_face_labels=False, basic_point_only=False))
    img.save(os.path.join(f'{image_dir}/{dance_name}', f'frame{j:06d}.jpg'))


def visualize(args, worker_num=16):
    json_dir = args.json_dir
    image_dir = args.image_dir

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    dance_names = sorted(os.listdir(json_dir))
    for i, dance_name in enumerate(dance_names):
        dance_path = os.path.join(json_dir, dance_name)
        fnames = sorted(os.listdir(dance_path))
        if not os.path.exists(f'{image_dir}/{dance_name}'):
            os.makedirs(f'{image_dir}/{dance_name}')

        # Visualize json in parallel
        pool = Pool(worker_num)
        partial_func = partial(visualize_json, image_dir=image_dir,
                               dance_name=dance_name, dance_path=dance_path, args=args)
        pool.map(partial_func, enumerate(fnames))
        pool.close()
        pool.join()
        
        print(f'Visualize {dance_name}')


def write2json(dances, dance_names, args):
    assert len(dances) == len(dance_names),\
        "number of generated dance != number of dance_names"
    for i in range(len(dances)):
        num_poses = dances[i].shape[0]
        dances[i] = dances[i].reshape(num_poses, pose_keypoints_num, 2)
        dance_path = os.path.join(args.json_dir, dance_names[i])
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
        print(f'Writing json to -> {dance_path}')


def main():
    args = get_args()

    if not os.path.exists(args.json_dir):
        os.makedirs(args.json_dir)

    music_data, dance_data, dance_names = load_data(
        args.input_dir, interval=None, data_type=args.data_type)

    device = torch.device('cuda' if args.cuda else 'cpu')

    test_loader = torch.utils.data.DataLoader(
        DanceDataset(music_data, dance_data),
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn
    )

    generator = Generator(args.model, device)

    results = []
    for batch in tqdm(test_loader, desc='Generating dance poses'):
        # Prepare data
        src_seq, src_pos, _ = map(lambda x: x.to(device), batch)
        pose_seq = generator.generate(src_seq, src_pos)
        results.append(pose_seq)

    # Visualize generated dance poses
    np_dances = []
    for i in range(len(results)):
        np_dance = results[i][0].data.cpu().numpy()
        root = np_dance[:, 2*8:2*9]
        np_dance = np_dance + np.tile(root, (1, 25))
        np_dance[:, 2*8:2*9] = root
        np_dances.append(np_dance)
    write2json(np_dances, dance_names, args)
    visualize(args)


if __name__ == '__main__':
    main()
