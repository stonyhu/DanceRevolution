# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


""" This module will handle the pose generation. """
import torch
import torch.nn as nn
from model import Encoder, Decoder, Model, get_subsequent_mask
import numpy as np
np.set_printoptions(threshold=np.inf)


class Generator(object):
    """ Load with trained model """
    def __init__(self, args, device):
        self.args = args
        self.device = device

        checkpoint = torch.load(args.model)
        model_args = checkpoint['args']

        print(f'Loaded model args: {model_args}')
        self.model_args = model_args

        encoder = Encoder(max_seq_len=model_args.max_seq_len,
                          input_size=model_args.d_frame_vec,
                          d_word_vec=model_args.frame_emb_size,
                          n_layers=model_args.n_layers,
                          n_head=model_args.n_head,
                          d_k=model_args.d_k,
                          d_v=model_args.d_v,
                          d_model=model_args.d_model,
                          d_inner=model_args.d_inner,
                          dropout=model_args.dropout)

        decoder = Decoder(input_size=model_args.d_pose_vec,
                          d_word_vec=model_args.pose_emb_size,
                          hidden_size=model_args.d_inner,
                          dropout=model_args.dropout)

        model = Model(encoder, decoder,
                      condition_step=model_args.condition_step,
                      sliding_windown_size=model_args.sliding_windown_size,
                      lambda_v=model_args.lambda_v,
                      device=device)

        # Data Parallel to use multi-gpu
        model = nn.DataParallel(model)

        model.load_state_dict(checkpoint['model'])
        # self.log.log.info('[Info] Trained model loaded.')
        print('[Info] Trained model loaded.')

        self.model = model.to(self.device)
        self.model.eval()

    def generate(self, src_seq, src_pos, tgt_seq=None):
        """ Generate dance pose in one batch """
        with torch.no_grad():

            # Use the pre-defined begin of pose (BOP) to generate whole sequence
            bsz, src_seq_len, _ = src_seq.size()
            # bsz, tgt_seq_len, dim = tgt_seq.size()
            tgt_seq_len = 1
            generated_frames_num = src_seq_len - tgt_seq_len

            hidden, dec_output, out_seq = self.model.module.init_decoder_hidden(bsz)
            vec_h, vec_c = hidden

            enc_mask = get_subsequent_mask(src_seq, self.model_args.sliding_windown_size)
            enc_outputs, *_ = self.model.module.encoder(src_seq, src_pos, enc_mask)

            preds = []
            for i in range(tgt_seq_len):
                # dec_input = tgt_seq[:, i]
                dec_input = dec_output
                dec_output, vec_h, vec_c = self.model.module.decoder(dec_input, vec_h, vec_c)
                dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
                dec_output = self.model.module.linear(dec_output)
                preds.append(dec_output)

            for i in range(generated_frames_num):
                dec_input = dec_output
                dec_output, vec_h, vec_c = self.model.module.decoder(dec_input, vec_h, vec_c)
                dec_output = torch.cat([dec_output, enc_outputs[:, i + tgt_seq_len]], 1)
                dec_output = self.model.module.linear(dec_output)
                preds.append(dec_output)

        outputs = [z.unsqueeze(1) for z in preds]
        outputs = torch.cat(outputs, dim=1)
        return outputs
