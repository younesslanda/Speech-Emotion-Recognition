#this model is inspired from : https://github.com/KrishnaDN/speech-emotion-recognition-using-self-attention

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Config as cfg
    
class Speech2Emotion(nn.Module):
    def __init__(self, input_spec_size=cfg.INPUT_SPEC_SIZE, cnn_filter_size=cfg.CNN_FILTER_SIZE, lstm_hidden_size=128, num_layers_lstm=2,
                 dropout_p=0.2, bidirectional=True, rnn_cell=cfg.RNN_CELL, num_gender_class=cfg.NUM_GENDER_CLASSES,
                 num_emotion_classes=cfg.NUM_EMOTION_CLASSES):
            
        super(Speech2Emotion, self).__init__()
        self.input_spec_size = input_spec_size
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.num_layers_lstm = num_layers_lstm
        self.dropout_p = dropout_p
        self.num_emo_classes = num_emotion_classes
        self.num_gender_class = num_gender_class
        self.cnn_filter_size = cnn_filter_size
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
            
    
        outputs_channel = self.cnn_filter_size
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, outputs_channel, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(outputs_channel, outputs_channel, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        
        rnn_input_dims = int(math.floor(input_spec_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_dims = int(math.floor(rnn_input_dims + 2 * 10 - 21) / 2 + 1)
        rnn_input_dims *= outputs_channel

        self.rnn =  self.rnn_cell(rnn_input_dims, self.lstm_hidden_size,
                                  self.num_layers_lstm, dropout=self.dropout_p, bidirectional=self.bidirectional)
        self.self_attn_layer = nn.TransformerEncoderLayer(d_model=self.lstm_hidden_size*2, dim_feedforward=512,nhead=8)
        self.gender_layer  = nn.Linear(self.lstm_hidden_size*4, self.num_gender_class)
        self.emotion_layer = nn.Linear(self.lstm_hidden_size*4, self.num_emo_classes)
        

    def forward(self, input_var, input_lengths=None):
        output_lengths = self.get_seq_lens(input_lengths)
        x = input_var # (B,1,D,T)
        x, _ = self.conv(x, output_lengths) # (B, C, D, T)
        
        x_size = x.size()
        x = x.view(x_size[0], x_size[1] * x_size[2], x_size[3]) # (B, C * D, T)
        x = x.transpose(1, 2).transpose(0, 1).contiguous() # (T, B, D)
        
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=True)
        x, h_state = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        
        x = x.transpose(0, 1) # (B, T, D)
        
        x = self.self_attn_layer(x)
        
        mu = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        pooled = torch.cat((mu,std),dim=1)
        
        gen_pred = self.gender_layer(pooled)
        emo_pred = self.emotion_layer(pooled)
        
        return emo_pred, gen_pred

    def get_seq_lens(self, input_lengths):
        seq_len = input_lengths
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d :
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)

        return seq_len.int()
    
class MaskConv(nn.Module):
    def __init__(self, seq_module):
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths): #(B,1,D,T)
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths