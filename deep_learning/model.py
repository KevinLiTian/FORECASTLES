import pandas as pd
import torch.nn as nn
import torch
import math
import numpy as np
from timeit import default_timer as timer
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
from encoder import *


class Full_Transformer(nn.Module):
    def __init__(self, input_dim, curr_state_dim, hidden_dim, num_layers, num_heads, output_dim):
        super(Full_Transformer, self).__init__()
        # decoder
        self.fc1 = nn.Linear(curr_state_dim, 32)
        self.batch_norm = nn.BatchNorm1d(32)
        self.fc_decoder_input = nn.Linear(32, hidden_dim) # input_dim is the number of features of the current state (4)
        self.decoder_positional_encoder = PositionalEncoding(hidden_dim, max_len=500)
        # self.fc_encoder_output = nn.Linear(input_dim, hidden_dim)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_heads, batch_first=True, dropout=0.0),
            num_layers,
            decoder_norm
        )
        # self.reduced_sum = ReducedSumLayer(dim=1)
        self.fc_dec0 = nn.Linear(hidden_dim, 32)
        self.fc_dec1 = nn.Linear(32, output_dim)
        # encoder
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.encoder_positional_encoder = PositionalEncoding(hidden_dim, max_len=500)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True, dropout=0.3)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, x, past_days):
        # encoder
        # print('0 semantic_goal:', semantic_goal.shape)  # (batch_size, max_goal_sent_len) -> 64 , 14
        # semantic_goal = self.embedding(semantic_goal)
        # print('1 semantic_goal:', semantic_goal.shape)  # (batch_size, max_goal_sent_len, hidden_dim) -> 64 , 14 , 1024
        x = self.fc0(x)
        x = torch.nn.ReLU()(x)
        x = self.encoder_positional_encoder(x)
        # print('2 semantic_goal:', semantic_goal.shape)  # (batch_size, max_goal_sent_len, hidden_dim) -> 64 , 14 , 1024
        x = self.encoder(x)
        # print('3 x:', x.shape)  # (batch_size, max_goal_sent_len, hidden_dim) -> 64 , 14 , 1024
        # decoder
        # print("Past days", past_days.shape)
        past_days = self.fc1(past_days)

        past_days = torch.permute(past_days, (0, 2, 1))
        past_days = self.batch_norm(past_days)
        past_days = torch.permute(past_days, (0, 2, 1))
        # print('4 current_state:', current_state.shape)  # (batch_size, num-obj, num-feature) -> 64 , 4 , 4
        past_days = self.fc_decoder_input(past_days)
        # print('5 current_state:', current_state.shape)  # (batch_size, num-obj, hidden_dim) -> 64 , 4 , 1024
        past_days = self.decoder_positional_encoder(past_days)
        # print('6 current_state:', current_state.shape)  # (batch_size, num-obj, hidden_dim) -> 64 , 4 , 1024
        decoder_mask = self._generate_square_subsequent_mask(past_days.size(1)).to(past_days.device)
        # print('7 decoder_mask:', decoder_mask.shape)  # (num-obj, num-obj) -> 4 , 4
        output = self.decoder(past_days, x, tgt_mask=decoder_mask)
        # print('8 output:', output.shape)  # (batch_size, num-obj , hidden_dim) -> 64 , 4 , 1024
        output = self.fc_dec0(output)
        output = torch.nn.ReLU()(output)
        output = self.fc_dec1(output)
        return output

    def _generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


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