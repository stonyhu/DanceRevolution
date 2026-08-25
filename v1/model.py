# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pose import BOS_POSE
from longformer.longformer import LongformerSelfAttention, LongformerConfig


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


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    """ Compose with two layers """
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        # self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        config = LongformerConfig()
        config.num_attention_heads = args.num_heads
        config.hidden_size = args.encoder_hidden_size
        config.attention_probs_dropout_prob = args.dropout
        config.attention_window = [args.window_size]
        config.attention_dilation = [1]  # No dilation
        config.attention_mode = 'tvm'
        config.output_attentions = True
        config.autoregressive = False

        self.self_attn = LongformerSelfAttention(config=config, layer_id=0)
        self.pos_ffn = PositionwiseFeedForward(
            args.encoder_hidden_size, args.encoder_hidden_size, dropout=args.dropout)
        
        self.layer_norm = nn.LayerNorm(args.encoder_hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        
        self.fc = nn.Linear(args.encoder_hidden_size, args.encoder_hidden_size)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, inputs):
        residual = inputs
        
        outputs, attn = self.self_attn(inputs)
        
        outputs = self.dropout(self.fc(outputs))
        outputs = self.layer_norm(outputs + residual)

        # print('---------------- longformer output ----------------')
        # print(enc_outputs.size())
        # print(enc_self_attn.size())
        outputs = self.pos_ffn(outputs)

        return outputs, attn


class Encoder(nn.Module):
    """ Music encoder with local self attention mechanism. """
    def __init__(self, args):
        super().__init__()
        num_position = args.max_seq_len + 1

        self.src_emb = nn.Linear(args.frame_dim, args.encoder_hidden_size)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(num_position, args.encoder_hidden_size, padding_idx=0), freeze=True)
        
        self.dropout = nn.Dropout(args.dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(args) for _ in range(args.num_layers)])

    def forward(self, src_seq, src_pos, mask=None, return_attns=False):
        enc_self_attn_list = []
        
        embedding = self.dropout(self.src_emb(src_seq))
        enc_outputs = embedding + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_outputs, enc_self_attn = enc_layer(enc_outputs)

            if return_attns:
                enc_self_attn_list += [enc_self_attn]

        if return_attns:
            return enc_outputs, enc_self_attn_list
        return enc_outputs,


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_size = args.decoder_hidden_size

        self.tgt_emb = nn.Linear(args.pose_dim, self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        self.lstm1 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)

    def init_state(self, bsz, device):
        c0 = torch.randn(bsz, self.hidden_size).to(device)
        c1 = torch.randn(bsz, self.hidden_size).to(device)
        c2 = torch.randn(bsz, self.hidden_size).to(device)
        h0 = torch.randn(bsz, self.hidden_size).to(device)
        h1 = torch.randn(bsz, self.hidden_size).to(device)
        h2 = torch.randn(bsz, self.hidden_size).to(device)

        vec_h = [h0, h1, h2]
        vec_c = [c0, c1, c2]

        bos = BOS_POSE
        bos = np.tile(bos, (bsz, 1))
        root = bos[:, 2 * 8:2 * 9]
        bos = bos - np.tile(root, (1, 25))
        bos[:, 2 * 8:2 * 9] = root
        out_pose = torch.from_numpy(bos).float().to(device)

        return (vec_h, vec_c), out_pose

    def forward(self, in_frame, vec_h, vec_c):
        in_frame = self.tgt_emb(in_frame)
        in_frame = self.dropout(in_frame)

        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]
        return vec_h2, vec_h_new, vec_c_new


class Model(nn.Module):
    def __init__(self, encoder, decoder, args, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(args.decoder_hidden_size + args.encoder_hidden_size, args.pose_dim)  # for luong attention

        self.n = args.fixed_steps
        self.alpha = args.alpha
        self.device = device

    def init_decoder_hidden(self, bsz):
        return self.decoder.init_state(bsz, self.device)

    # Dynamic auto-condition training
    def forward(self, src_seq, src_pos, tgt_seq, hidden, dec_output, epoch_i):
        seq_len = tgt_seq.size(1)

        groundtruth_mask = torch.ones(seq_len, self.n)
        prediction_mask = torch.zeros(seq_len, int(epoch_i * self.alpha))
        mask = torch.cat([prediction_mask, groundtruth_mask], 1).view(-1)[:seq_len]

        enc_outputs, *_ = self.encoder(src_seq, src_pos)
        vec_h, vec_c = hidden

        preds = []
        for i in range(seq_len):
            dec_input = tgt_seq[:, i] if mask[i] == 1 else dec_output.detach()  # dec_output
            dec_output, vec_h, vec_c = self.decoder(dec_input, vec_h, vec_c)
            dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
            dec_output = self.linear(dec_output)
            preds.append(dec_output)

        outputs = [z.unsqueeze(1) for z in preds]
        outputs = torch.cat(outputs, dim=1)
        # print(f'outputs -> {outputs.size()}')
        return outputs
