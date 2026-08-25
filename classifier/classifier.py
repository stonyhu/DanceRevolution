# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class Classifier(nn.Module):
    def __init__(self,
                 max_seq_len=900, input_size=50, emb_dim=200,
                 num_classes=3, hidden_size=200, dropout=0.1):
        super(Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.src_emb = nn.Linear(input_size, emb_dim)
        self.lstm1 = nn.LSTMCell(emb_dim, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)



    def forward(self, src_seq):
        device = torch.device('cuda')
        x = self.src_emb(src_seq)

        bsz, seq_len, _ = src_seq.size()
        vec_h = torch.zeros(bsz, self.hidden_size).to(device)
        vec_c = torch.zeros(bsz, self.hidden_size).to(device)

        for i in range(seq_len):
            vec_h, vec_c = self.lstm1(x[:, i], (vec_h, vec_c))

        features = vec_h

        x = self.fc(vec_h)
        return x, features
