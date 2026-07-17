# This source code is licensed under the MIT License found in the
# LICENSE file in the root directory of this open-source project.


import torch
import torch.nn as nn
from model import Encoder, Decoder, Model
from utils.functional import printer
import numpy as np


class Generator(object):
    """ Load with trained model """
    def __init__(self, model_file, device):
        self.device = device

        checkpoint = torch.load(model_file)
        model_args = checkpoint['args']
        self.model_args = model_args
        print(f'[Info] Loading model args:')
        printer(model_args)

        encoder = Encoder(model_args)
        decoder = Decoder(model_args)
        model = Model(encoder, decoder, model_args, device=device)

        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model'])
        # self.log.log.info('[Info] Trained model loaded.')
        print('[Info] Trained model loaded.')

        # Use gpu to accelerate inference
        self.model = model.to(device)
        self.model.eval()

    # Random
    def generate(self, src_seq, src_pos):
        """ Generate dance pose in one batch """
        with torch.no_grad():

            bsz, src_seq_len, _ = src_seq.size()
            generated_frames_num = src_seq_len

            hidden, dec_output = self.model.module.init_decoder_hidden(bsz)
            vec_h, vec_c = hidden

            enc_outputs, *_ = self.model.module.encoder(src_seq, src_pos)

            preds = []
            for i in range(generated_frames_num):
                dec_input = dec_output
                dec_output, vec_h, vec_c = self.model.module.decoder(dec_input, vec_h, vec_c)
                dec_output = torch.cat([dec_output, enc_outputs[:, i]], 1)
                dec_output = self.model.module.linear(dec_output)
                preds.append(dec_output)

        outputs = [z.unsqueeze(1) for z in preds]
        outputs = torch.cat(outputs, dim=1)
        # print(f'outputs -> {outputs.size()}')
        return outputs
