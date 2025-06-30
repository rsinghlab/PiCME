import copy
import math

import torch
import torch.nn as nn
from torch.functional import F


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class BertAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attn_dropout,
        ctx_dim=None,
    ):
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = hidden_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attn_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(context))
        value_layer = self.transpose_for_scores(self.value(context))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # mask: 1, 0; same setup as HF
            assert attention_mask.unique().sum() == 1
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob) -> None:
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(self.dense(hidden_states))
        return hidden_states + input_tensor


class BertSelfAttnLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()

        self.attn = BertAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.output = BertAttOutput(hidden_size, attention_probs_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        output = self.attn(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertCrossAttnLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        ctx_dim=None,
    ):
        super().__init__()

        self.attn = BertAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim
        )
        self.output = BertAttOutput(hidden_size, attention_probs_dropout_prob)

    def forward(self, input_tensor, ctx_tensor, attention_mask=None):
        output = self.attn(input_tensor, ctx_tensor, attention_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(hidden_dropout_prob),
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.mlp(self.LayerNorm(input_tensor))
        return hidden_states + input_tensor


class BertEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout):
        super().__init__()

        self.mha = BertSelfAttnLayer(hidden_size, num_attention_heads, dropout)
        self.mlp = BertMLP(hidden_size, intermediate_size, dropout)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.mha(hidden_states, attention_mask)
        layer_output = self.mlp(attention_output)
        return layer_output


class CrossEncoderLayer(nn.Module):
    def __init__(self, ts_size, txt_size, num_heads, dropout):
        super().__init__()

        self.ts2txt_cross_attention = BertAttention(
            ts_size, num_heads, dropout, txt_size
        )
        self.txt2ts_cross_attention = BertAttention(
            txt_size, num_heads, dropout, ts_size
        )

        self.ts_self_encoder = BertEncoderLayer(ts_size, num_heads, ts_size, dropout)
        self.txt_self_encoder = BertEncoderLayer(txt_size, num_heads, txt_size, dropout)

    def forward(self, hidden_ts, hidden_tx, mask_ts=None, mask_tx=None):

        new_ts = self.ts2txt_cross_attention(hidden_ts, hidden_tx, mask_tx)
        new_ts = self.ts_self_encoder(new_ts, mask_ts)

        new_txt = self.txt2ts_cross_attention(hidden_tx, hidden_ts, mask_ts)
        new_txt = self.txt_self_encoder(new_txt, mask_tx)

        return new_ts, new_txt


# PyTorch implementation of Positional Encoding Layer.
class PositionalEncodingSeqFirst(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PositionalEncodingBatchFirst(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_length, d_model)
        k = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args: x: embeddings (batch_size, seq_length, d_model)
        Returns: embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class BertEncoder(nn.Module):
    def __init__(self, config, hidden_size, num_layer=1, batch_first=True):
        super().__init__()

        layer = BertEncoderLayer(
            hidden_size=hidden_size,
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=int(hidden_size * config["intermediate_multiplier"]),
            dropout=config["dropout"],
        )
        self.hidden_size = hidden_size

        self.layers = _get_clones(layer, num_layer)

        if batch_first:
            self.pe = PositionalEncodingBatchFirst(hidden_size)
        else:
            self.pe = PositionalEncodingSeqFirst(hidden_size)

    def forward(self, hidden_states, attention_mask=None):

        length = hidden_states.size(1)
        hidden_states = self.pe(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class CrossEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        layer = CrossEncoderLayer(
            ts_size=config["ts_size"],
            txt_size=config["txt_size"],
            num_heads=config["num_attention_heads"],
            dropout=config["dropout"],
        )

        self.ts_size = config["ts_size"]
        self.txt_size = config["txt_size"]

        self.layers = _get_clones(layer, config["num_layer_cross"])

    def forward(self, hidden_ts, hidden_txt, mask_ts=None, mask_txt=None):

        for layer in self.layers:
            hidden_ts, hidden_txt = layer(hidden_ts, hidden_txt, mask_ts, mask_txt)

        return hidden_ts, hidden_txt


class FusionModel(nn.Module):
    def __init__(self, config, Y) -> None:
        super().__init__()

        self._init_cls(config)
        self._init_layers(config, Y)

    def _init_layers(self, config, Y):

        ts_size = config["ts_size"]
        txt_size = config["txt_size"]

        ts_num_layer = config["num_layer_ts"]
        txt_num_layer = config["num_layer_txt"]

        # encoders
        self.ts_encoder = BertEncoder(config, ts_size, ts_num_layer)
        self.txt_encoder = BertEncoder(config, txt_size, txt_num_layer)
        self.cross_encoder = CrossEncoder(config)

        # output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(ts_size + txt_size, ts_size + txt_size),
            nn.Tanh(),
            nn.Linear(ts_size + txt_size, Y),
        )

    def _init_cls(self, config):
        # cls tokens
        cls_ts = nn.Parameter(torch.randn(1, 1, config["ts_size"]))
        cls_txt = nn.Parameter(torch.randn(1, 1, config["txt_size"]))
        torch.nn.init.xavier_uniform_(cls_ts.data)
        torch.nn.init.xavier_uniform_(cls_txt.data)
        self.register_parameter("cls_ts", cls_ts)
        self.register_parameter("cls_txt", cls_txt)

    def _append_cls(self, ts_input, txt_input, ts_attn_mask, txt_attn_mask):

        bsize = ts_input.size(0)
        cls_ts = self.cls_ts.expand(bsize, -1, -1)
        cls_txt = self.cls_txt.expand(bsize, -1, -1)
        ts_input = torch.cat([cls_ts, ts_input], 1)
        txt_input = torch.cat([cls_txt, txt_input], 1)

        mask_one = torch.ones(bsize, 1)
        if ts_attn_mask is not None:
            ts_attn_mask = torch.cat([mask_one.type_as(ts_attn_mask), ts_attn_mask], 1)
        if txt_attn_mask is not None:
            txt_attn_mask = torch.cat(
                [mask_one.type_as(txt_attn_mask), txt_attn_mask], 1
            )

        return ts_input, txt_input, ts_attn_mask, txt_attn_mask

    def forward(
        self,
        ts_input,
        txt_input,
        target,
        ts_attn_mask=None,
        txt_attn_mask=None,
        *args,
        **kwargs,
    ):
        ts_input, txt_input, ts_attn_mask, txt_attn_mask = self._append_cls(
            ts_input, txt_input, ts_attn_mask, txt_attn_mask
        )

        # encode
        hidden_ts = self.ts_encoder(ts_input, ts_attn_mask)
        hidden_txt = self.txt_encoder(txt_input, txt_attn_mask)

        hidden_ts, hidden_txt = self.cross_encoder(
            hidden_ts, hidden_txt, ts_attn_mask, txt_attn_mask
        )

        # output
        cls_ts = hidden_ts[:, 0]
        cls_txt = hidden_txt[:, 0]

        final_hidden = torch.cat([cls_ts, cls_txt], -1)
        logits = self.output_layer(final_hidden)
        loss = nn.CrossEntropyLoss()(logits, target)

        return logits, loss


class ContrastFusionModel(FusionModel):
    def __init__(self, config, Y) -> None:
        super().__init__(config, Y)

        self._init_cls(config)
        self._init_layers(config, Y)

        self._init_contrast(config)
        self._init_momentum_layers(config)

    def _init_momentum_layers(self, config):

        ts_size = config["ts_size"]
        txt_size = config["txt_size"]

        ts_num_layer = config["num_layer_ts"]
        txt_num_layer = config["num_layer_txt"]

        self.ts_encoder_m = BertEncoder(config, ts_size, ts_num_layer)
        self.txt_encoder_m = BertEncoder(config, txt_size, txt_num_layer)

        self.ts_proj_m = nn.Linear(ts_size, config["proj_dim"])
        self.txt_proj_m = nn.Linear(txt_size, config["proj_dim"])

        self.model_pairs = [
            [self.ts_encoder, self.ts_encoder_m],
            [self.txt_encoder, self.txt_encoder_m],
            [self.ts_proj, self.ts_proj_m],
            [self.txt_proj, self.txt_proj_m],
        ]

        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    def _init_contrast(self, config):

        embed_dim = config["proj_dim"]

        self.ts_proj = nn.Linear(config["ts_size"], embed_dim)
        self.txt_proj = nn.Linear(config["txt_size"], embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config["temp"])

        self.queue_size = config["queue_size"]
        self.momentum = config["momentum"]
        self.alpha = config["alpha"]

        self.train_stage = True

        # create the queue
        self.register_buffer("ts_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("txt_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.ts_queue = nn.functional.normalize(self.ts_queue, dim=0)
        self.txt_queue = nn.functional.normalize(self.txt_queue, dim=0)

    def forward(
        self, ts_input, txt_input, target, ts_attn_mask=None, txt_attn_mask=None
    ):

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # cls
        ts_input, txt_input, ts_attn_mask, txt_attn_mask = self._append_cls(
            ts_input, txt_input, ts_attn_mask, txt_attn_mask
        )

        # encode
        hidden_ts = self.ts_encoder(ts_input, ts_attn_mask)
        hidden_txt = self.txt_encoder(txt_input, txt_attn_mask)

        ts_feat = torch.functional.F.normalize(self.ts_proj(hidden_ts[:, 0]), dim=-1)
        txt_feat = torch.functional.F.normalize(self.txt_proj(hidden_txt[:, 0]), dim=-1)

        # get momentum fts
        with torch.no_grad():
            self._momentum_update()

            hidden_ts_m = self.ts_encoder_m(ts_input, ts_attn_mask)
            hidden_txt_m = self.txt_encoder_m(txt_input, txt_attn_mask)

            ts_feat_m = torch.functional.F.normalize(
                self.ts_proj_m(hidden_ts_m[:, 0]), dim=-1
            )
            txt_feat_m = torch.functional.F.normalize(
                self.txt_proj_m(hidden_txt_m[:, 0]), dim=-1
            )

            ts_feat_all = torch.cat(
                [ts_feat_m.T, self.ts_queue.clone().detach()], dim=1
            )
            txt_feat_all = torch.cat(
                [txt_feat_m.T, self.txt_queue.clone().detach()], dim=1
            )

            sim_targets = torch.zeros(ts_feat_m.size(0), ts_feat_all.size(1)).to(
                ts_feat_m.device
            )
            sim_targets.fill_diagonal_(1)

        sim_ts2txt = ts_feat @ txt_feat_all / self.temp
        sim_txt2ts = txt_feat @ ts_feat_all / self.temp

        loss_ts2txt = -torch.sum(
            F.log_softmax(sim_ts2txt, dim=1) * sim_targets, dim=1
        ).mean()
        loss_txt2ts = -torch.sum(
            F.log_softmax(sim_txt2ts, dim=1) * sim_targets, dim=1
        ).mean()

        loss_contrastive = (loss_ts2txt + loss_txt2ts) / 2

        if self.train_stage:
            # no update when forward on val and test
            self._dequeue_and_enqueue(ts_feat_m, txt_feat_m)

        # cross attn fusion and final output
        hidden_ts, hidden_txt = self.cross_encoder(
            hidden_ts, hidden_txt, ts_attn_mask, txt_attn_mask
        )

        # output
        cls_ts = hidden_ts[:, 0]
        cls_txt = hidden_txt[:, 0]

        final_hidden = torch.cat([cls_ts, cls_txt], -1)

        logits = self.output_layer(final_hidden)
        loss_sup = nn.CrossEntropyLoss()(logits, target)

        # final loss
        loss = self.alpha * loss_contrastive + (1 - self.alpha) * loss_sup

        return logits, loss

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, ts_feat, txt_feat):
        # # gather keys before updating queue
        ts_feats = ts_feat
        txt_feats = txt_feat

        batch_size = ts_feats.shape[0]

        ptr = int(self.queue_ptr)

        if self.queue_size % batch_size == 0:  # for simplicity
            self.ts_queue[:, ptr : ptr + batch_size] = ts_feats.T
            self.txt_queue[:, ptr : ptr + batch_size] = txt_feats.T
            ptr = (ptr + batch_size) % self.queue_size  # move pointer

            self.queue_ptr[0] = ptr
