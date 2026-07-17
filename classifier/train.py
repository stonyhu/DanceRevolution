# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn.functional as F
from classifier import Classifier


class DanceDataset(Dataset):
    def __init__(self, dances, labels=None):
        if labels is not None:
            assert (len(labels) == len(dances)), \
                'the number of dances should be equal to the number of labels'
        self.dances = dances
        self.labels = labels

    def __len__(self):
        return len(self.dances)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.dances[index], self.labels[index]
        else:
            return self.dances[index]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_prefix)
    torch.save(model, save_path)


def eval(dev_data, classifier, device, args):
    classifier.eval()
    corrects, avg_loss, size = 0, 0, 0
    for i, batch in enumerate(dev_data):
        dance, label = map(lambda x: x.to(device), batch)
        dance = dance.type(torch.cuda.FloatTensor)
        logits, _ = classifier(dance)
        loss = F.cross_entropy(logits, label)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(label.size()).data == label.data).sum()
        size += batch[0].shape[0]

    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def train(train_data, dev_data, classifier, device, args):
    optimizer = optim.Adam(filter(lambda x: x.requires_grad,
                                  classifier.parameters()), lr=args.lr)
    best_acc = 0
    steps = 0
    classifier.train()
    for epoch in range(1, args.epochs+1):
        for i, batch in enumerate(train_data):
            dance, label = map(lambda x: x.to(device), batch)
            dance = dance.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            logits, _ = classifier(dance)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            steps += 1

            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
                train_acc = 100.0 * corrects / batch[0].shape[0]
                print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                               loss.item(),
                                                                               train_acc,
                                                                               corrects,
                                                                               batch[0].shape[0]))


        # evaluate the model on test set at each epoch
        dev_acc = eval(dev_data, classifier, device, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
            save(classifier, args.save_dir, args.save_model, steps)


def load_data(data_dir, interval):
    dance_data, labels = [], []
    fnames = sorted(os.listdir(data_dir))
    if ".ipynb_checkpoints" in fnames:
        fnames.remove(".ipynb_checkpoints")
    # fnames = fnames[:60]  # For debug
    for fname in fnames:
        path = os.path.join(data_dir, fname)
        label = -1
        if "hiphop" in fname:
            label = 0
        elif "ballet" in fname:
            label = 1
        elif "manako" in fname:
            label = 2

        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_dance = np.array(sample_dict['dance_array'])
            # Only use 25 keypoints skeleton (basic bone) for 2D
            np_dance = np_dance[:, :50]


            seq_len, dim = np_dance.shape
            for i in range(0, seq_len, interval):
                dance_sub_seq = np_dance[i: i + interval]
                if len(dance_sub_seq) == interval and label != -1:
                    labels.append(label)
                    dance_data.append(dance_sub_seq)

    return dance_data, labels


def prepare_dataloader(dance_data, labels, args):
    data_loader = torch.utils.data.DataLoader(
        DanceDataset(dance_data, labels),
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=True,
        # collate_fn=paired_collate_fn,
        pin_memory=True
    )

    return data_loader


def main():
    """ Main function """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='data/train_1min',
                        help='the directory of dance data')
    parser.add_argument('--valid_dir', type=str, default='data/valid_1min',
                        help='the directory of music feature data')
    parser.add_argument('--save_model', type=str, default='best_model_200dim_50interval_rnn',
                        help='model name')
    parser.add_argument('--save_dir', metavar='PATH', default='checkpoints/')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=50)
    parser.add_argument('--interval', type=int, default=50)

    args = parser.parse_args()

    

    # Loading data
    dance_data, labels = load_data(args.train_dir, args.interval)
    z = list(zip(dance_data, labels))
    random.shuffle(z)
    dance_data, labels = zip(*z)
   

    train_data = prepare_dataloader(dance_data[1000:], labels[1000:], args)
    dev_data = prepare_dataloader(dance_data[:1000], labels[:1000], args)

    device = torch.device('cuda')

    classifier = Classifier()

    for name, parameters in classifier.named_parameters():
        print(name, ':', parameters.size())

    classifier = classifier.to(device)
    train(train_data, dev_data, classifier, device, args)


if __name__ == '__main__':
    main()
