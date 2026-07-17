# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import os
import json
import time
from tqdm import tqdm
import argparse
import torch
import torch.utils.data
import numpy as np
from multiprocessing import Pool
from functools import partial
from random import shuffle
from v2.dataset import DanceDataset #, paired_collate_fn
from v2.utils.functional import str2bool
from v2.generator import Generator
from v2.keypoint2img import read_keypoints
import essentia.streaming
from essentia.standard import *
from extractor import FeatureExtractor
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, default='music/test')
parser.add_argument('--output_dir', type=str, default='generation_dances')
parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument('--dance_num', type=int, default=10,
                    help='the number of generated dances for one audio')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--model', type=str, metavar='PATH',
                    default='checkpoints/epoch_best.pt')
parser.add_argument('--batch_size', type=int, metavar='N', default=1)
parser.add_argument('--worker_num', type=int, default=16)
parser.add_argument('--width', type=int, default=1280,
                    help='the width pixels of target image')
parser.add_argument('--height', type=int, default=720,
                    help='the height pixels of target image')
parser.add_argument('--pose_dim', type=int, default=50)
parser.add_argument('--sample_rate', type=int, default=15360)
parser.add_argument('--fps',type=str, default='15')
args = parser.parse_args()

extractor = FeatureExtractor()

pose_keypoints_num = 25
face_keypoints_num = 70
hand_left_keypoints_num = 21
hand_right_keypoints_num = 21

# only using src_seq and src_pos at inference time
def paired_collate_fn(insts):
    src_seq = insts
    src_pos = np.array([[
        pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in src_seq])

    src_seq = torch.FloatTensor(src_seq)
    src_pos = torch.LongTensor(src_pos)

    return src_seq, src_pos

def visualize_json(fname, json_path, img_path):
    fname_prefix = fname.split('_')[0]
    json_file = os.path.join(json_path, fname)
    img = Image.fromarray(read_keypoints(json_file, (args.width, args.height),
                          remove_face_labels=False, basic_point_only=False))
    img.save(os.path.join(img_path, f'{fname_prefix}.jpg'))

def visualize(data_dir, worker_num=16):
    # t1 = time.time()
    audio_fnames = sorted(os.listdir(data_dir))
    for i, audio_fname in enumerate(audio_fnames):
        dance_ids = sorted(os.listdir(os.path.join(data_dir,audio_fname)))
        for json_id in dance_ids:
            if args.fps=='15':
                json_path = os.path.join(data_dir, audio_fname, json_id,"json_30fps")
            else:
                json_path = os.path.join(data_dir, audio_fname, json_id,"json")
            fnames = sorted(os.listdir(json_path))
            img_path = os.path.join(data_dir, audio_fname, json_id,"img")
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            # Visualize json in parallel
            pool = Pool(worker_num)
            partial_func = partial(visualize_json, json_path=json_path, img_path=img_path)
            pool.map(partial_func, fnames)
            pool.close()
            pool.join()
        print(f'visualize {audio_fname}')
    # t2 = time.time()
    # print(f'parallel time cost: {int(t2 - t1)}s')


def interpolate(frames, stride=10):
    new_frames=[]
    for i in range(len(frames)-1):

        inter_points=np.zeros((25,3))
        left_points = frames[i]
        right_points = frames[i+1]
        for j in range(len(inter_points)):
            inter_points[j][0] = (left_points[j][0] + right_points[j][0])/2
            inter_points[j][1] = (left_points[j][1] + right_points[j][1])/2
            inter_points[j][2] = (left_points[j][2] + right_points[j][2])/2
        new_frames.append(left_points)
        new_frames.append(inter_points)
    new_frames.append(frames[-1])
    new_frames.append(frames[-1])

    return new_frames

def store(frames, out_dance_path):
    for i, pose_points in enumerate(frames):


        people_dicts = []
        people_dict = {'pose_keypoints_2d': np.array(pose_points).reshape(-1).tolist(),
                       'face_keypoints_2d': np.zeros((70, 3)).tolist(),
                       'hand_left_keypoints_2d': np.zeros((21, 3)).tolist(),
                       'hand_right_keypoints_2d': np.zeros((21, 3)).tolist(),
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

def convert_to_30fps(dance_path, out_dance_path):

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
    frames=interpolate(frames)
    # Store the corrected frames
    store(frames, out_dance_path)


def write2json(dances,audio_fnames):
    assert len(audio_fnames)*args.dance_num == len(dances), "Dance datas mismatch the file names"

    for i in range(len(dances)):
        num_poses = dances[i].shape[0]
        dances[i] = dances[i].reshape(num_poses, pose_keypoints_num, 2)
        dance_path = os.path.join(args.output_dir, audio_fnames[i//args.dance_num].split(".m4a")[0],
            f'{i%args.dance_num:02d}',"json")

        if not os.path.exists(dance_path):
            os.makedirs(dance_path)

        for j in range(num_poses):
            frame_dict = {'version': 1.2}
            # 2-D key points
            pose_keypoints_2d = []
            # Random values for the below key points
            face_keypoints_2d = np.zeros((70, 3)).tolist()
            hand_left_keypoints_2d = np.zeros((21, 3)).tolist()
            hand_right_keypoints_2d = np.zeros((21, 3)).tolist()
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
        print(f'finished writing to json {i%args.dance_num:02d}')

        if args.fps == '15':
            out_dance_path = os.path.join(args.output_dir, audio_fnames[i//args.dance_num].split(".m4a")[0],
            f'{i%args.dance_num:02d}',"json_30fps")
            if not os.path.exists(out_dance_path):
                os.makedirs(out_dance_path)
            convert_to_30fps(dance_path,out_dance_path)


def extract_acoustic_feature(input_audio_dir, fps='30',sr=15360):
    #assert (fps == '15') or (fps == '30') or (fps == '20'), "illegal fps"
    assert (fps == '15') or (fps == '30'), "illegal fps"
    print('---------- Extract features from raw audio ----------')
    music_data = []
    # onset_beats = []
    audio_fnames = sorted(os.listdir(input_audio_dir))

    if ".ipynb_checkpoints" in audio_fnames:
        audio_fnames.remove(".ipynb_checkpoints")
    for audio_fname in audio_fnames:
        audio_file = os.path.join(input_audio_dir, audio_fname)
        print(f'Process -> {audio_file}')
        # Load audio
        # sr = args.sampling_rate
        # sr = 48000
        loader = essentia.standard.MonoLoader(filename=audio_file, sampleRate=sr)
        audio = loader()
        audio = np.array(audio).T

        melspe_db = extractor.get_melspectrogram(audio, sr)
        mfcc = extractor.get_mfcc(melspe_db)
        mfcc_delta = extractor.get_mfcc_delta(mfcc)
        # mfcc_delta2 = extractor.get_mfcc_delta2(mfcc)

        # audio_harmonic, audio_percussive = librosa.effects.hpss(audio)
        audio_harmonic, audio_percussive = extractor.get_hpss(audio)
        # harmonic_melspe_db = extractor.get_harmonic_melspe_db(audio_harmonic, sr)
        # percussive_melspe_db = extractor.get_percussive_melspe_db(audio_percussive, sr)
        chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr)
        # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

        onset_env = extractor.get_onset_strength(audio_percussive, sr)
        tempogram = extractor.get_tempogram(onset_env, sr)
        onset_beat = extractor.get_onset_beat(onset_env, sr)
        # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # onset_beats.append(onset_beat)

        onset_env = onset_env.reshape(1, -1)

        feature = np.concatenate([
            mfcc,
            mfcc_delta,
            chroma_cqt,
            onset_env,
            onset_beat,
            tempogram
        ], axis=0)

        feature = feature.transpose(1, 0)
        print(f'acoustic feature -> {feature.shape}')
        music_data.append(feature.tolist())

    if fps == '30':
        new_musics = music_data
    else:
        new_musics=[]
        for i in range(len(music_data)):
            seq_len = (len(music_data[i])//2)*2
            new_musics.append([music_data[i][j] for j in range(seq_len) if j%2==0])
            new_musics.append([music_data[i][j] for j in range(seq_len) if j%2==1])

    return new_musics, audio_fnames


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    music_data, audio_fnames = extract_acoustic_feature(args.test_dir,args.fps)
    # print(audio_fnames)

    test_loader = torch.utils.data.DataLoader(
        DanceDataset(music_data),
        batch_size=args.batch_size,
        collate_fn=paired_collate_fn
    )

    device = torch.device('cuda' if args.cuda else 'cpu')

    generator = Generator(args, device)

    results = []
    for batch in tqdm(test_loader, desc='Generating dance poses'):
        src_seq, src_pos = map(lambda x: x.to(device), batch)
        generate_num = args.dance_num//2 if args.fps=='15' else args.dance_num
        for _ in range(generate_num):
            poses = generator.generate(src_seq, src_pos)
            results.append(poses)

    np_dances = []
    for i in range(len(results)):
        np_dance = results[i][0].data.cpu().numpy()
        root = np_dance[:,2*8:2*9]
        np_dance = np_dance + np.tile(root,(1,25))
        np_dance[:,2*8:2*9] = root

        np_dances.append(np_dance)

    write2json(np_dances,audio_fnames)

    visualize(args.output_dir, worker_num=args.worker_num)

if __name__ == '__main__':
    main()

