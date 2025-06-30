import peft
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torchvision.models import resnet50
from transformers import AutoModel


class ImageEncoder(nn.Module):
    def __init__(self, projection_dim):
        super(ImageEncoder, self).__init__()
        self.model = resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, projection_dim)

    def forward(self, x):
        return self.model(x)


# Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, projection_dim):
        super(TextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
            lora_dropout=0.01,
            bias="none",
        )
        self.model = get_peft_model(self.model, config)
        self.fc = nn.Linear(self.model.config.hidden_size, projection_dim)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0, :])


# Time Series Encoder
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(TimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=projection_dim,
            batch_first=True,
            bidirectional=False,
            dropout=0.1,
        )

    def forward(self, x, lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (ht, _) = self.lstm(x)
        return ht.squeeze()


# Demographic Data Encoder (MLP)
class DemoEncoder(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(DemoEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        return self.model(x)
