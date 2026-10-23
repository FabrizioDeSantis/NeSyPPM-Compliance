import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from model.inputs import EventEncoder


class LSTMModel(nn.Module):
    '''
    Bidirectional LSTM over the events of a prefix. Returns P(label = 1), shape (B, 1).
    With num_rule_features > 0 the input also contains the rule truth values (see EventEncoder).
    '''

    def __init__(self, feature_names, vocab_sizes, seq_len, hidden_size=128, num_layers=2, dropout=0.1,
                 num_rule_features=0):
        super().__init__()
        self.encoder = EventEncoder(feature_names, vocab_sizes, seq_len, hidden_size, num_rule_features)
        self.lstm = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        events, padding = self.encoder(x)
        lengths = (~padding).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(events, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        # Last layer: forward state after the last event, backward state after the first one
        last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return torch.sigmoid(self.fc(last))
