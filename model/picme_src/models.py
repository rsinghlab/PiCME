import math

import torch
import torch.nn as nn
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from picme_src.encoders import *
from picme_src.losses import *
from .picme_utils import secure_fusion


class BaseContrastiveModel(nn.Module):
    def __init__(self, ts_input_dim, demo_input_dim, projection_dim, num_modalities):
        super(BaseContrastiveModel, self).__init__()
        self.image_encoder = ImageEncoder(projection_dim)
        self.text_encoder = TextEncoder(projection_dim)
        self.ts_encoder = TimeSeriesEncoder(ts_input_dim, projection_dim)
        self.demo_encoder = DemoEncoder(demo_input_dim, projection_dim)
        self.projection = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.num_modalities = num_modalities
        self.initialize_weights()

    def initialize_weights(self):
        for model in self.modules():
            if isinstance(model, (nn.LSTM, nn.RNN, nn.GRU)):
                nn.init.orthogonal_(model.weight_hh_l0)
                nn.init.xavier_uniform_(model.weight_ih_l0)
                nn.init.zeros_(model.bias_hh_l0)
                nn.init.zeros_(model.bias_ih_l0)

    def encode_modality(self, data, modality_type, att_mask=None, lengths=None):
        if modality_type == "img":
            features = self.image_encoder(data)
        elif modality_type in ["text_rad", "text_ds"]:
            features = self.text_encoder(data, att_mask)
        elif modality_type == "ts":
            features = self.ts_encoder(data, lengths.cpu())
        elif modality_type == "demo":
            features = self.demo_encoder(data)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")

        return self.projection(features)


class ContrastiveModel(BaseContrastiveModel):
    def __init__(self, ts_input_dim, demo_input_dim, projection_dim):
        super(ContrastiveModel, self).__init__(
            ts_input_dim, demo_input_dim, projection_dim
        )

    def forward(
        self,
        modality1,
        modality2,
        modality1_type,
        modality2_type,
        att_mask1=None,
        att_mask2=None,
        lengths=None,
    ):
        embeddings1 = self.encode_modality(
            modality1, modality1_type, att_mask1, lengths
        )
        embeddings2 = self.encode_modality(
            modality2, modality2_type, att_mask2, lengths
        )
        return embeddings1, embeddings2


class MultiModalContrastiveModel(BaseContrastiveModel):
    def __init__(
        self, ts_input_dim, demo_input_dim, projection_dim, num_modalities, ovo=False
    ):
        super(MultiModalContrastiveModel, self).__init__(
            ts_input_dim, demo_input_dim, projection_dim, num_modalities=num_modalities
        )
        self.ovo = ovo
        if self.ovo:
            self.N_weights = nn.Parameter(
                torch.ones(self.num_modalities) / self.num_modalities
            )
        else:
            num_pairs = self.num_modalities * (self.num_modalities - 1) // 2
            self.pair_weights = nn.Parameter(torch.ones(num_pairs) / num_pairs)

    def forward(self, modalities_data, modalities_type, mask_rad, mask_ds, ts_lengths):
        embeddings = [
            self.encode_modality(
                data,
                modality_type,
                (
                    mask_rad
                    if modality_type == "text_rad"
                    else mask_ds if modality_type == "text_ds" else None
                ),
                ts_lengths if modality_type == "ts" else None,
            )
            for data, modality_type in zip(modalities_data, modalities_type)
        ]
        return embeddings


class MultiModalBaseline(nn.Module):
    def __init__(
        self,
        ts_input_dim,
        demo_input_dim,
        projection_dim,
        args,  # args passed directly
        device,
    ):
        """
        Initialize encoders and the classification head.
        """
        super(MultiModalBaseline, self).__init__()
        self.device = device
        self.modalities = args.modalities
        self.num_modalities = len(self.modalities)
        self.args = args

        # Initialize encoders dynamically using ModuleDict
        self.encoders = nn.ModuleDict(
            {
                "img": ImageEncoder(projection_dim),
                "text_rad": TextEncoder(projection_dim),
                "text_ds": TextEncoder(projection_dim),
                "ts": TimeSeriesEncoder(ts_input_dim, projection_dim),
                "demo": DemoEncoder(demo_input_dim, projection_dim),
            }
        )
        self._TASK_CLASSES = {
            "mortality": 2,
            "phenotyping": 25,
        }

        # Initialize the classification head using provided args
        self.classifier_head = ClassificationHead(
            projection_dim=projection_dim,
            num_classes=self._TASK_CLASSES[args.task],
            fusion_method=args.fusion_method,
            num_modalities=self.num_modalities,
            modality_lambdas=args.modality_lambdas,
        ).to(self.device)

    def encode_modality(self, data, modality_type, att_mask=None, lengths=None):
        """
        Encode a single modality using the appropriate encoder from ModuleDict.
        """
        if modality_type not in self.encoders:
            raise ValueError(f"Unknown modality type: {modality_type}")

        if modality_type in ["text_rad", "text_ds"]:
            features = self.encoders[modality_type](data, att_mask)
        elif modality_type == "ts":
            features = self.encoders[modality_type](data, lengths.cpu())
        else:
            features = self.encoders[modality_type](data)

        return features
    
    def hooked_forward(
            self, 
            modalities_data, 
            modalities_type, 
            mask_rad=None, 
            mask_ds=None, 
            ts_lengths=None
    ):
        """
        Encode all modalities, fuse and return them for IG analysis.
        """
        # Encode all modalities and collect their embeddings
        embeddings = [
            self.encode_modality(
                data,
                modality_type,
                att_mask=(
                    mask_rad
                    if modality_type == "text_rad"
                    else mask_ds if modality_type == "text_ds" else None
                ),
                lengths=(ts_lengths if modality_type == "ts" else None),
            )
            for data, modality_type in zip(modalities_data, modalities_type)
        ]

        concatenated_embeddings = secure_fusion(
            embeddings, self.device, self.args.fusion_method
        )
        output = self.classifier_head(concatenated_embeddings)

        return concatenated_embeddings, output

    def forward(
        self,
        modalities_data,
        modalities_type,
        mask_rad=None,
        mask_ds=None,
        ts_lengths=None,
    ):
        """
        Encode all modalities, fuse them, and classify the output.
        """
        # Encode all modalities and collect their embeddings
        embeddings = [
            self.encode_modality(
                data,
                modality_type,
                att_mask=(
                    mask_rad
                    if modality_type == "text_rad"
                    else mask_ds if modality_type == "text_ds" else None
                ),
                lengths=(ts_lengths if modality_type == "ts" else None),
            )
            for data, modality_type in zip(modalities_data, modalities_type)
        ]

        concatenated_embeddings = secure_fusion(
            embeddings, self.device, self.args.fusion_method
        )
        output = self.classifier_head(concatenated_embeddings)

        return output


class ModalityEnhancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, modality_lambda=None, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_seq = []
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(device),
                torch.zeros(bs, self.hidden_size).to(device),
            )
        else:
            h_t, c_t = init_states

        hs = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            lambda_t = 1 if not modality_lambda else modality_lambda[t]

            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :hs]),  # input
                torch.sigmoid(gates[:, hs : hs * 2]),  # forget
                torch.tanh(gates[:, hs * 2 : hs * 3]),  # candidate memory
                torch.sigmoid(gates[:, hs * 3 :]),  # output
            )

            modality_scale_factor = torch.zeros_like(g_t) + lambda_t
            c_t = f_t * c_t + ((i_t * g_t) * modality_scale_factor)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        projection_dim,
        num_classes,
        fusion_method,
        num_modalities,
        modality_lambdas,
        verbose=True,
    ):
        super(ClassificationHead, self).__init__()
        self.fusion_method = fusion_method
        self.num_modalities = num_modalities
        self.modality_lambdas = modality_lambdas

        if fusion_method == "concatenation":
            if verbose:
                print(f"Building a {fusion_method} fusion classifier head.")
            self.classifier = nn.Sequential(
                nn.Linear(projection_dim * num_modalities, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
        elif fusion_method == "vanilla_lstm":
            self.vanilla_lstm = nn.LSTM(
                projection_dim, projection_dim, batch_first=True, dropout=0.0
            )
            self.classifier = nn.Sequential(
                nn.Linear(projection_dim, num_classes), nn.Sigmoid()
            )
        elif fusion_method == "modality_lstm":
            assert num_modalities == len(modality_lambdas)
            self.modality_lstm = ModalityEnhancedLSTM(projection_dim, projection_dim)
            self.classifier = nn.Sequential(
                nn.Linear(projection_dim, num_classes), nn.Sigmoid()
            )

    def forward(self, embeddings):
        if self.fusion_method == "concatenation":
            output = self.classifier(embeddings)
        elif self.fusion_method == "vanilla_lstm":
            seq_lengths = np.array([self.num_modalities] * len(embeddings))
            feats = torch.nn.utils.rnn.pack_padded_sequence(
                embeddings, seq_lengths, batch_first=True, enforce_sorted=False
            )
            _, (ht, _) = self.vanilla_lstm(feats)
            lstm_out = ht.squeeze(0)
            output = self.classifier(lstm_out)
        elif self.fusion_method == "modality_lstm":
            _, (ht, _) = self.modality_lstm(
                embeddings, modality_lambda=self.modality_lambdas
            )
            output = self.classifier(ht)

        return output
