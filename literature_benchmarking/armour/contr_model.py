import copy
import math

import torch
import torch.nn as nn
from bert_utils import *
from grud import *
from torch.functional import F


class ExpModel(nn.Module):
    def __init__(self, config, Y):
        super().__init__()

        self.grud = GRUD(config["ts_size"], config["ts_size"])
        self.fusion = ContrastFusionModel(config, Y)

    def forward(self, x_ts, x_txt, labels, ts_attn_mask=None, txt_attn_mask=None):

        # encode measurements
        x, mask, delta = x_ts
        x, mask, delta = x.float(), mask.float(), delta.float()

        ts_input, _ = self.grud(x, mask, delta)
        txt_input = x_txt.unsqueeze(1)
        #         print(ts_input.size(), ts_input)
        #         print(txt_input.size(), txt_input)
        if txt_attn_mask is not None:
            txt_attn_mask = txt_attn_mask.unsqueeze(1)

        # run through fusion
        logits, loss = self.fusion(
            ts_input, txt_input, labels, ts_attn_mask, txt_attn_mask
        )

        return logits, loss


def init_model(config):
    task = config["task"]

    if task == "in-hospital-mortality":
        Y = 2
    elif task == "phenotyping":
        Y = 27

    model = ExpModel(config, Y)

    return model
