# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from extractor import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--input_audio_dir', type=str, default='data/audio')
parser.add_argument('--input_dance_dir', type=str, default='data/json')

parser.add_argument('--train_dir', type=str, default='data/train_1min')
parser.add_argument('--test_dir', type=str, default='data/test_1min')

parser.add_argument('--sampling_rate', type=int, default=15360)
args = parser.parse_args()

extractor = FeatureExtractor()

if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)
if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)


def extract_acoustic_feature(input_audio_dir):
    print('---------- Extract features from raw audio ----------')
    musics = []
    # onset_beats = []
    audio_fnames = sorted(os.listdir(input_audio_dir))
    # audio_fnames = audio_fnames[:20]  # for debug
    print(f'audio_fnames: {audio_fnames}')

    for audio_fname in audio_fnames:
        audio_file = os.path.join(input_audio_dir, audio_fname)
        print(f'Process -> {audio_file}')
        ### load audio ###
        sr = args.sampling_rate
        loader = essentia.standard.MonoLoader(filename=audio_file, sampleRate=sr)
        audio = loader()
        audio = np.array(audio).T

        melspe_db = extractor.get_melspectrogram(audio, sr)
        mfcc = extractor.get_mfcc(melspe_db)
        mfcc_delta = extractor.get_mfcc_delta(mfcc)
        # mfcc_delta2 = get_mfcc_delta2(mfcc)

        audio_harmonic, audio_percussive = extractor.get_hpss(audio)
        # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
        # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
        chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr)
        # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

        onset_env = extractor.get_onset_strength(audio_percussive, sr)
        tempogram = extractor.get_tempogram(onset_env, sr)
        onset_beat = extractor.get_onset_beat(onset_env, sr)
        # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # onset_beats.append(onset_beat)

        onset_env = onset_env.reshape(1, -1)

        feature = np.concatenate([
            # melspe_db,
            mfcc,
            mfcc_delta,
            # mfcc_delta2,
            # harmonic_melspe_db,
            # percussive_melspe_db,
            # chroma_stft,
            chroma_cqt,
            onset_env,
            onset_beat,
            tempogram
        ], axis=0)

        feature = feature.transpose(1, 0)
        print(f'acoustic feature -> {feature.shape}')
        musics.append(feature.tolist())

    return musics

def load_dance_data(dance_dir, remove_hand_keypoints=True, remove_face_keypoints=True):
    print('---------- Loading pose keypoints ----------')
    dances = []
    dir_names = sorted(os.listdir(dance_dir))
    # dir_names = dir_names[:20]  # for debug
    print(f'dance dir names: {dir_names}')

    for dir_name in dir_names:
        poses = []
        prev_key_points = None
        pose_path = os.path.join(dance_dir, dir_name)
        print(f'Process -> {pose_path}')
        json_files = sorted(os.listdir(pose_path))

        for i, json_file in enumerate(json_files):
            with open(os.path.join(pose_path, json_file)) as f:
                keypoint_dicts = json.loads(f.read())['people']
                if len(keypoint_dicts) > 0:
                    keypoint_dict = keypoint_dicts[0]
                    pose_points = np.array(keypoint_dict['pose_keypoints_2d']).reshape(25, 3)[:, :-1].reshape(-1)
                    face_points = np.array(keypoint_dict["face_keypoints_2d"]).reshape(70, 3)[:, :-1].reshape(-1) \
                        if not remove_face_keypoints else []
                    hand_points_l = np.array(keypoint_dict["hand_left_keypoints_2d"]).reshape(21, 3)[:, :-1].reshape(-1) \
                        if not remove_hand_keypoints else []
                    hand_points_r = np.array(keypoint_dict["hand_right_keypoints_2d"]).reshape(21, 3)[:, :-1].reshape(-1) \
                        if not remove_hand_keypoints else []
                    key_points = np.concatenate([pose_points, face_points, hand_points_l, hand_points_r], 0)
                    # if i == 0:
                    #     relative_points = np.zeros(key_points.size)
                    # else:
                    #     relative_points = key_points - prev_key_points
                    # prev_key_points = key_points
                    # poses.append(relative_points.tolist())
                    poses.append(key_points.tolist())
        dances.append(poses)
        print(np.shape(poses))

    return dances



def align(musics, dances):
    print('---------- Align the frames of music and dance ----------')
    assert len(musics) == len(dances), \
        'the number of audios should be equal to that of videos'
    new_musics=[]
    new_dances=[]
    for i in range(len(musics)):
        min_seq_len = min(len(musics[i]), len(dances[i]))
        print(f'music -> {np.array(musics[i]).shape}, ' +
              f'dance -> {np.array(dances[i]).shape}, ' +
              f'min_seq_len -> {min_seq_len}')
        del musics[i][min_seq_len:]
        del dances[i][min_seq_len:]
        new_musics.append([musics[i][j] for j in range(min_seq_len) if j%2==0])
        new_musics.append([musics[i][j] for j in range(min_seq_len) if j%2==1])
        
        new_dances.append([dances[i][j] for j in range(min_seq_len) if j%2==0])
        new_dances.append([dances[i][j] for j in range(min_seq_len) if j%2==1])
    return new_musics, new_dances



def split_data(fnames):
    print('---------- Split data into train and test ----------')
    indices = list(range(len(fnames)))
    random.shuffle(indices)
    train = indices[10:]
    test = indices[:10]


    return train, test


def save(args, musics, dances, inner_dir):
    print('---------- Save to text file ----------')
    fnames = sorted(os.listdir(os.path.join(args.input_dance_dir,inner_dir)))
    # fnames = fnames[:20]  # for debug
    assert len(fnames)*2 == len(musics) == len(dances), 'alignment'

    train_idx, test_idx = split_data(fnames)
    train_idx = sorted(train_idx)
    print(f'train ids: {[fnames[idx] for idx in train_idx]}')
    test_idx = sorted(test_idx)
    print(f'test ids: {[fnames[idx] for idx in test_idx]}')

    print('---------- train data ----------')

    for idx in train_idx:
        for i in range(2):
            with open(os.path.join(args.train_dir, f'{inner_dir+"_"+fnames[idx]+"_"+str(i)}.json'), 'w') as f:
                sample_dict = {
                    'id': fnames[idx]+"_"+str(i),
                    'music_array': musics[idx*2+i],
                    'dance_array': dances[idx*2+i]
                }
                json.dump(sample_dict, f)

    print('---------- test data ----------')
    for idx in test_idx:
        for i in range(2):
            with open(os.path.join(args.test_dir, f'{inner_dir+"_"+fnames[idx]+"_"+str(i)}.json'), 'w') as f:
                sample_dict = {
                    'id': fnames[idx]+"_"+str(i),
                    'music_array': musics[idx*2+i],
                    'dance_array': dances[idx*2+i]
                }
                json.dump(sample_dict, f)



if __name__ == '__main__':
    music_dirs = os.listdir(args.input_audio_dir)
    dance_dirs = os.listdir(args.input_dance_dir)
    for music_dir in music_dirs:
        if music_dir in dance_dirs and music_dir != ".DS_Store":
      
            musics = extract_acoustic_feature(os.path.join(args.input_audio_dir, music_dir))
            dances = load_dance_data(os.path.join(args.input_dance_dir, music_dir),
                             remove_face_keypoints=False,
                             remove_hand_keypoints=False)

            musics, dances = align(musics, dances)
            save(args, musics, dances, music_dir)


