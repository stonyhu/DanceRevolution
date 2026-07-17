# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import DanceDataset, paired_collate_fn
from model import Encoder, Decoder, Model
from utils.log import Logger
from utils.functional import str2bool, load_data
import warnings
warnings.filterwarnings('ignore')


def train(model, training_data, optimizer, device, args, log):
    """ Start training """
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    updates = 0  # global step

    for epoch_i in range(1, args.epoch + 1):
        log.set_progress(epoch_i, len(training_data))
        model.train()

        for batch_i, batch in enumerate(training_data):
            # prepare data
            src_seq, src_pos, tgt_seq = map(lambda x: x.to(device), batch)
            gold_seq = tgt_seq[:, 1:]
            src_seq = src_seq[:, :-1]
            src_pos = src_pos[:, :-1]
            tgt_seq = tgt_seq[:, :-1]

            hidden, out_frame, out_seq = model.module.init_decoder_hidden(tgt_seq.size(0))

            # forward
            optimizer.zero_grad()

            output = model(src_seq, src_pos, tgt_seq, hidden, out_frame, out_seq, epoch_i)

            # backward
            loss = criterion(output, gold_seq)
            loss.backward()

            # update parameters
            optimizer.step()

            stats = {
                'updates': updates,
                'loss': loss.item()
            }
            log.update(stats)
            updates += 1

        checkpoint = {
            'model': model.state_dict(),
            'args': args,
            'epoch': epoch_i
        }

        if epoch_i % args.save_per_epochs == 0 or epoch_i == 1:
            filename = os.path.join(args.output_dir, f'epoch_{epoch_i}.pt')
            torch.save(checkpoint, filename)


def main():
    """ Main function """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='data/train_1min',
                        help='the directory of dance data')
    parser.add_argument('--test_dir', type=str, default='data/test_1min',
                        help='the directory of music feature data')
    parser.add_argument('--data_type', type=str, default='2D',
                        help='the type of training data')
    parser.add_argument('--output_dir', metavar='PATH',
                        default='checkpoints/layers2_win100_schedule100_condition10_detach')

    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_per_epochs', type=int, metavar='N', default=500)
    parser.add_argument('--log_per_updates', type=int, metavar='N', default=1,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument('--tensorboard', action='store_false')

    parser.add_argument('--d_frame_vec', type=int, default=438)
    parser.add_argument('--frame_emb_size', type=int, default=200)
    parser.add_argument('--d_pose_vec', type=int, default=50)
    parser.add_argument('--pose_emb_size', type=int, default=50)

    parser.add_argument('--d_inner', type=int, default=1024)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--seq_len', type=int, default=900)
    parser.add_argument('--max_seq_len', type=int, default=4500)
    parser.add_argument('--condition_step', type=int, default=10)
    parser.add_argument('--sliding_windown_size', type=int, default=100)
    parser.add_argument('--lambda_v', type=float, default=0.01)

    parser.add_argument('--cuda', type=str2bool, nargs='?', metavar='BOOL', const=True,
                        default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')

    args = parser.parse_args()
    args.d_model = args.frame_emb_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    global log
    log = Logger(args)
    print(args)
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Loading training data
    train_music_data, train_dance_data = load_data(
        args.train_dir, 
        interval=args.seq_len,
        data_type=args.data_type)
    training_data = prepare_dataloader(train_music_data, train_dance_data, args)

    device = torch.device('cuda' if args.cuda else 'cpu')

    encoder = Encoder(max_seq_len=args.max_seq_len,
                      input_size=args.d_frame_vec,
                      d_word_vec=args.frame_emb_size,
                      n_layers=args.n_layers,
                      n_head=args.n_head,
                      d_k=args.d_k,
                      d_v=args.d_v,
                      d_model=args.d_model,
                      d_inner=args.d_inner,
                      dropout=args.dropout)

    decoder = Decoder(input_size=args.d_pose_vec,
                      d_word_vec=args.pose_emb_size,
                      hidden_size=args.d_inner,
                      encoder_d_model=args.d_model,
                      dropout=args.dropout)

    model = Model(encoder, decoder,
                  condition_step=args.condition_step,
                  sliding_windown_size=args.sliding_windown_size,
                  lambda_v=args.lambda_v,
                  device=device)

    print(model)

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    # Data Parallel to use multi-gpu
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(filter(
        lambda x: x.requires_grad, model.module.parameters()), lr=args.lr)

    train(model, training_data, optimizer, device, args, log)


def prepare_dataloader(music_data, dance_data, args):
    data_loader = torch.utils.data.DataLoader(
        DanceDataset(music_data, dance_data),
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=paired_collate_fn,
        pin_memory=True
    )

    return data_loader


if __name__ == '__main__':
    main()
