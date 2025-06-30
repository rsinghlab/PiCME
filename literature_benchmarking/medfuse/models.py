import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.functional import kl_div, log_softmax, softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet34


class CXRModel(nn.Module):
    def __init__(self, projection_dim):
        super(CXRModel, self).__init__()
        self.model = getattr(torchvision.models, "resnet34")(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, projection_dim)

    def forward(self, x):
        return self.model(x)


class EHRModel(nn.Module):
    def __init__(
        self,
        input_dim=76,
        hidden_dim=256,
        batch_first=True,
        dropout=0.0,
        layers=1,
    ):
        super(EHRModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        for layer in range(layers):
            setattr(
                self,
                f"layer{layer}",
                nn.LSTM(
                    input_dim, hidden_dim, batch_first=batch_first, dropout=dropout
                ),
            )
            input_dim = hidden_dim
        self.do = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        for model in self.modules():

            if type(model) in [nn.Linear]:
                nn.init.xavier_uniform_(model.weight)
                nn.init.zeros_(model.bias)
            elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x, seq_lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths, batch_first=True, enforce_sorted=False
        )
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f"layer{layer}")(x)
        feats = ht.squeeze()
        if self.do is not None:
            feats = self.do(feats)
        return feats


class Fusion(nn.Module):
    def __init__(self, projection_dim, pred_classes, ts_dim, args):
        super(Fusion, self).__init__()

        self.ehr_model = EHRModel(ts_dim, projection_dim, layers=args.ts_layers)
        self.cxr_model = CXRModel(projection_dim)

        target_classes = pred_classes

        self.projection = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        self.lstm_fusion_layer = nn.LSTM(
            projection_dim, projection_dim, batch_first=True, dropout=0.0
        )

        self.lstm_fused_cls = nn.Sequential(
            nn.Linear(projection_dim, target_classes), nn.Sigmoid()
        )

    def forward_lstm_fused(self, x_ehr, x_cxr, seq_lengths):
        ehr_feats = self.ehr_model(x_ehr, seq_lengths)
        cxr_feats = self.cxr_model(x_cxr)

        ehr_proj = self.projection(ehr_feats)
        cxr_proj = self.projection(cxr_feats)

        feats = ehr_proj[:, None, :]
        feats = torch.cat([feats, cxr_proj[:, None, :]], dim=1)

        seq_lengths = np.array([2] * len(seq_lengths))

        feats = torch.nn.utils.rnn.pack_padded_sequence(
            feats, seq_lengths, batch_first=True, enforce_sorted=False
        )

        _, (ht, _) = self.lstm_fusion_layer(feats)
        out = ht.squeeze()
        fused_preds = self.lstm_fused_cls(out)

        return {
            "lstm": fused_preds,
            "ehr_feats": ehr_feats,
            "cxr_feats": cxr_feats,
        }
