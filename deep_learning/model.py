import pandas as pd
import torch.nn as nn
import torch
import math
import numpy as np
from timeit import default_timer as timer
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
from encoder import *


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            # (BS, F)
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            # (BS, 1)
        )

    def forward(self, x):
        return self.layers(x)


class TimeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, seq_len):
        super(TimeModel, self).__init__()

        # self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc0 = nn.Linear(input_dim, 256)

        self.fc1 = nn.Linear(256, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=1)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True, dropout=0.3)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.fc_dec = nn.Linear(hidden_dim, 256)
        self.fc_dec2 = nn.Linear(256, 32)
        self.fc = nn.Linear(32*seq_len, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # x = self.embedding(x)
        # print('x:', x[:2, :, :])
        x = self.fc0(x)
        x = torch.nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)
        # print('x:', x[:2, :, :])
        x = self.positional_encoding(x)
        # print('x:', x[:2, :, :])
        x = self.encoder(x)
        # print('x:', x[:2, :, :])
        # print('x:', x[:2, :, :])
        x = torch.nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc_dec(x)
        x = torch.nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc_dec2(x)
        x = torch.nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('x:', x.shape)
        # print('self.pe:', self.pe.shape)
        x = x + self.pe[:, :x.size(1), :]
        return x


class NewsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, seq_len):
        super(NewsModel, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=1)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.fc = nn.Linear(hidden_dim*seq_len, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x, news):
        # x = self.embedding(x)
        # print('x:', x[:2, :, :])
        news_embed = encode_text(news)
        x = self.fc0(x)
        # print('x:', x[:2, :, :])
        x = self.positional_encoding(x)
        # print('x:', x[:2, :, :])
        x = self.encoder(x)
        x = torch.concat([x, news_embed], dim=1)
        # print('x:', x[:2, :, :])
        # print('x:', x[:2, :, :])
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.nn.ReLU(x)
        x = self.fc2(x)
        return x