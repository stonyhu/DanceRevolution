# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


""" Define the Seq2Seq Generation Network """
import numpy as np
import torch
import torch.nn as nn
from utils.pose import BOS_POSE
from layers import MultiHeadAttention, PositionwiseFeedForward


def get_non_pad_mask(seq):
    assert seq.dim() == 3
    non_pad_mask = torch.abs(seq).sum(2).ne(0).type(torch.float)
    return non_pad_mask.unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = torch.abs(seq_k).sum(2).eq(0)  # sum the vector of last dim and then judge
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq, sliding_windown_size):
    """ For masking out the subsequent info. """
    batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8)
    mask = torch.triu(mask, diagonal=-sliding_windown_size)
    mask = torch.tril(mask, diagonal=sliding_windown_size)
    mask = 1 - mask
    # print(mask)
    return mask.bool()


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


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, non_pad_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, max_seq_len=1800, input_size=20, d_word_vec=10,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            d_model=10, d_inner=256, dropout=0.1):

        super().__init__()

        self.d_model = d_model
        n_position = max_seq_len + 1

        self.src_emb = nn.Linear(input_size, d_word_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
    
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    def __init__(self, input_size=274, d_word_vec=150, hidden_size=200,
                 dropout=0.1, encoder_d_model=200):
        super().__init__()

        self.input_size = input_size
        self.d_word_vec = d_word_vec
        self.hidden_size = hidden_size
      
        self.tgt_emb = nn.Linear(input_size, d_word_vec)
        self.dropout = nn.Dropout(dropout)
        self.encoder_d_model = encoder_d_model

        self.lstm1 = nn.LSTMCell(d_word_vec, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(hidden_size, hidden_size)

    def init_state(self, bsz, device):
        c0 = torch.randn(bsz, self.hidden_size).to(device)
        c1 = torch.randn(bsz, self.hidden_size).to(device)
        c2 = torch.randn(bsz, self.hidden_size).to(device)
        h0 = torch.randn(bsz, self.hidden_size).to(device)
        h1 = torch.randn(bsz, self.hidden_size).to(device)
        h2 = torch.randn(bsz, self.hidden_size).to(device)

        vec_h = [h0, h1, h2]
        vec_c = [c0, c1, c2]

        BOP = BOS_POSE
        BOP = np.tile(BOP, (bsz, 1))
        root = BOP[:, 2*8:2*9]
        bos = BOP - np.tile(root, (1, 25))
        bos[:, 2*8:2*9] = root
        out_frame = torch.from_numpy(bos).float().to(device)
        out_seq = torch.FloatTensor(bsz, 1).to(device)

        return (vec_h, vec_c), out_frame, out_seq

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
    def __init__(self, encoder, decoder, condition_step=10,
                 sliding_windown_size=100, lambda_v=0.01, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(decoder.hidden_size + encoder.d_model, decoder.input_size)

        self.condition_step = condition_step
        self.sliding_windown_size = sliding_windown_size
        self.lambda_v = lambda_v
        self.device = device

    def init_decoder_hidden(self, bsz):
        return self.decoder.init_state(bsz, self.device)

    # dynamic auto-condition + self-attention mask
    def forward(self, src_seq, src_pos, tgt_seq, hidden, dec_output, out_seq, epoch_i):
        bsz, seq_len, _ = tgt_seq.size()
        vec_h, vec_c = hidden

        enc_mask = get_subsequent_mask(src_seq, self.sliding_windown_size)
        enc_outputs, *_ = self.encoder(src_seq, src_pos, mask=enc_mask)

        groundtruth_mask = torch.ones(seq_len, self.condition_step)
        prediction_mask = torch.zeros(seq_len, int(epoch_i * self.lambda_v))
        mask = torch.cat([prediction_mask, groundtruth_mask], 1).view(-1)[:seq_len]  # for random

        preds = []
        for i in range(seq_len):
            dec_input = tgt_seq[:, i] if mask[i] == 1 else dec_output.detach()  # dec_output
            dec_output, vec_h, vec_c = self.decoder(dec_input, vec_h, vec_c)
            dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
            dec_output = self.linear(dec_output)
            preds.append(dec_output)

        outputs = [z.unsqueeze(1) for z in preds]
        outputs = torch.cat(outputs, dim=1)
        return outputs
